"""
FastAPI backend for the market research assistant.

Endpoints:
    GET  /           — serves the web UI
    GET  /eval       — serves the evaluation dashboard
    POST /query      — runs the Scout → Mapper → Critic pipeline
    POST /compare    — runs multiple model pipelines side-by-side
    POST /eval/run   — runs the eval suite when explicitly enabled

Usage:
    uvicorn app.main:app --reload
"""

import asyncio
import json
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agents.llm_client import ModelId, MODEL_REGISTRY
from pipeline.orchestrator import run_pipeline_full

app = FastAPI(title="Market Research Analyst")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = pathlib.Path(__file__).parent / "static"
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"
COMPARE_MODEL_TIMEOUT_SECONDS = float(os.getenv("COMPARE_MODEL_TIMEOUT_SECONDS", "240"))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str
    model: ModelId | None = None


class CompareRequest(BaseModel):
    """Request body for running one query across multiple LLM pipelines."""

    query: str
    models: list[ModelId] = Field(default_factory=lambda: ["gemini", "llama", "qwen"])

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: list[ModelId]) -> list[ModelId]:
        """Validate the requested comparison model list.

        Args:
            models: Short model ids requested by the client.

        Returns:
            The validated model list.

        Raises:
            ValueError: If the list is empty or includes duplicates, because the
                response is keyed by model id and duplicate keys would be lost.
        """
        if not models:
            raise ValueError("At least one model must be requested.")
        if len(set(models)) != len(models):
            raise ValueError("Model list must not include duplicates.")
        return models


class EvalRunRequest(BaseModel):
    """Request body for gated on-demand eval runs."""

    models: list[ModelId] = Field(default_factory=lambda: ["gemini", "llama", "qwen"])

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: list[ModelId]) -> list[ModelId]:
        """Validate models requested for an eval run.

        Args:
            models: Short model ids to evaluate.

        Returns:
            The validated model list.

        Raises:
            ValueError: If no models are requested or if duplicate ids would
                overwrite result keys in the saved dashboard JSON.
        """
        if not models:
            raise ValueError("At least one model must be requested.")
        if len(set(models)) != len(models):
            raise ValueError("Model list must not include duplicates.")
        return models


class ThemeResult(BaseModel):
    theme_name: str
    companies: list[str]
    rationale: str
    citations: list[str]


class AgentStats(BaseModel):
    latency_ms: float
    docs_retrieved: int | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class PipelineStats(BaseModel):
    scout: AgentStats
    mapper: AgentStats
    critic: AgentStats
    total_cost_usd: float
    total_latency_ms: float


class QueryResponse(BaseModel):
    query: str
    themes: list[ThemeResult]
    stats: PipelineStats
    trace_id: str


class CompareModelResult(BaseModel):
    """One model's comparison payload.

    Fields:
        themes: list of ThemeResult entries with keys theme_name, companies,
            rationale, citations.
        latency_ms: Wall-clock runtime for that model's pipeline invocation.
        cost_usd: Sum of per-agent token costs from the pipeline timings.
        trace_url: Optional direct Langfuse trace link when URL metadata exists.
        error: Optional per-model error. Present when that model failed or timed
            out; other model results can still render normally.
    """

    themes: list[ThemeResult]
    latency_ms: float
    cost_usd: float
    trace_url: str | None
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/eval", response_class=HTMLResponse)
async def serve_eval_dashboard() -> HTMLResponse:
    """Serve the static evaluation dashboard page.

    Returns:
        HTML response for ``app/static/eval.html``.
    """
    html_path = STATIC_DIR / "eval.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/eval/results/latest.json")
async def latest_eval_results() -> dict:
    """Return the latest saved evaluation dashboard JSON.

    Returns:
        Dict with keys: generated_at, models. ``models`` maps model ids to
        aggregate metric dicts.

    Raises:
        HTTPException: If no eval result file exists yet.
    """
    result_path = EVAL_RESULTS_DIR / "latest.json"
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval results found. Enable ALLOW_EVAL_RUN=true and run /eval/run.",
        )
    return json.loads(result_path.read_text())


@app.post("/eval/run")
async def run_eval_endpoint(request: EvalRunRequest) -> dict:
    """Run the eval suite on demand when explicitly enabled.

    Args:
        request: Model ids to evaluate. Running multiple models can spend API
            budget, so this endpoint is disabled unless ``ALLOW_EVAL_RUN`` is
            set to ``true``.

    Returns:
        Saved dashboard JSON with keys: generated_at, models.

    Raises:
        HTTPException: If eval runs are disabled or the evaluator fails.
    """
    if os.getenv("ALLOW_EVAL_RUN", "").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Set ALLOW_EVAL_RUN=true to enable on-demand eval runs.",
        )

    try:
        payload = await asyncio.to_thread(_run_eval_models, request.models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return payload


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    model = request.model or _default_model()
    try:
        result = run_pipeline_full(request.query, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    timings = result["timings"]
    total_cost = round(
        sum(v.get("cost_usd", 0.0) for v in timings.values()), 6
    )
    total_latency = round(
        sum(v.get("latency_ms", 0.0) for v in timings.values()), 1
    )
    pipeline_stats = PipelineStats(
        scout=AgentStats(**timings.get("scout", {"latency_ms": 0})),
        mapper=AgentStats(**timings.get("mapper", {"latency_ms": 0})),
        critic=AgentStats(**timings.get("critic", {"latency_ms": 0})),
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
    )
    return QueryResponse(
        query=request.query,
        themes=[ThemeResult(**theme) for theme in result["reviewed_map"]],
        stats=pipeline_stats,
        trace_id=result["trace_id"],
    )


@app.post("/compare", response_model=dict[str, CompareModelResult])
async def compare_endpoint(request: CompareRequest) -> dict[str, CompareModelResult]:
    """Run the same query through multiple model-backed pipelines concurrently.

    Args:
        request: Query text plus a list of short model ids. The returned JSON is
            keyed by those ids; each value includes themes, wall-clock latency,
            cost, and a Langfuse trace URL when one can be constructed.

    Returns:
        Dict keyed by model id with keys: themes, latency_ms, cost_usd,
        trace_url.

    Raises:
        HTTPException: If the query is blank or any model pipeline fails.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    model_results = await asyncio.gather(
        *(_run_compare_model(request.query, model) for model in request.models)
    )
    return dict(model_results)


async def _run_compare_model(
    query: str,
    model: ModelId,
) -> tuple[str, CompareModelResult]:
    """Run one synchronous pipeline invocation in a worker thread.

    Args:
        query: Natural language market research query.
        model: Short model id selecting the Mapper/Critic LLM.

    Returns:
        Tuple of the short model id and its comparison response payload.
    """
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline_full, query, model=model),
            timeout=COMPARE_MODEL_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return (
            model,
            CompareModelResult(
                themes=[],
                latency_ms=latency_ms,
                cost_usd=0.0,
                trace_url=None,
                error=(
                    f"Timed out after {int(COMPARE_MODEL_TIMEOUT_SECONDS)}s. "
                    "Try this model by itself or retry later."
                ),
            ),
        )
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return (
            model,
            CompareModelResult(
                themes=[],
                latency_ms=latency_ms,
                cost_usd=0.0,
                trace_url=None,
                error=str(e),
            ),
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    timings = result["timings"]
    cost_usd = round(sum(v.get("cost_usd", 0.0) for v in timings.values()), 6)
    return (
        model,
        CompareModelResult(
            themes=[ThemeResult(**theme) for theme in result["reviewed_map"]],
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            trace_url=_trace_url(result["trace_id"]),
        ),
    )


def _trace_url(trace_id: str) -> str | None:
    """Build a Langfuse trace URL when deployment metadata is configured.

    Args:
        trace_id: Langfuse-compatible trace id returned by the pipeline.

    Returns:
        A direct trace URL if either ``LANGFUSE_TRACE_URL_TEMPLATE`` is set or
        both ``LANGFUSE_HOST`` and ``LANGFUSE_PROJECT_ID`` are present;
        otherwise None. Avoiding a guessed URL keeps API output honest when the
        project id is unavailable.
    """
    template = os.getenv("LANGFUSE_TRACE_URL_TEMPLATE")
    if template:
        return template.format(trace_id=trace_id)

    host = os.getenv("LANGFUSE_HOST")
    project_id = os.getenv("LANGFUSE_PROJECT_ID")
    if not host or not project_id:
        return None
    return f"{host.rstrip('/')}/project/{project_id}/traces/{trace_id}"


def _run_eval_models(models: list[ModelId]) -> dict:
    """Run evaluator aggregates for one or more models and save dashboard JSON.

    Args:
        models: Short model ids evaluated by Mapper and Critic.

    Returns:
        Dict with keys: generated_at, models. Each model value contains
        aggregate metric keys returned by ``eval.evaluator.run_eval``.
    """
    from eval.evaluator import run_eval

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    generated_at = datetime.now(timezone.utc).isoformat()
    for model in models:
        aggregate = run_eval(lambda q, m=model: run_pipeline_full(q, model=m))
        results[model] = {"aggregate": aggregate}

    payload = {"generated_at": generated_at, "models": results}
    timestamp = generated_at.replace(":", "").replace("+", "Z")
    (EVAL_RESULTS_DIR / f"latest_{timestamp}.json").write_text(
        json.dumps(payload, indent=2)
    )
    (EVAL_RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    return payload


def _default_model() -> ModelId:
    """Read the API's default LLM model from the environment.

    Returns:
        A valid short model id from ``MODEL_REGISTRY``. Defaults to ``"gemini"``
        because the project CI/demo path already uses Google AI Studio.

    Raises:
        HTTPException: If ``DEFAULT_MODEL`` is set to an unknown model id.
    """
    model = os.getenv("DEFAULT_MODEL", "gemini")
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid DEFAULT_MODEL={model!r}. "
                f"Expected one of: {', '.join(MODEL_REGISTRY)}."
            ),
        )
    return cast(ModelId, model)
