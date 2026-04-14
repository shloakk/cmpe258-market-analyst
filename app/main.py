"""
FastAPI backend for the market research assistant.

Endpoints:
    GET  /           — serves the web UI
    POST /query      — runs the Scout → Mapper → Critic pipeline

Usage:
    uvicorn app.main:app --reload
"""

import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.orchestrator import run_pipeline_full

app = FastAPI(title="Market Research Analyst")

STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str


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


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        result = run_pipeline_full(request.query)
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
    )
