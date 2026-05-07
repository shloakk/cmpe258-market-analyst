"""
Evaluation metrics and runners for the multi-agent market research pipeline.

Metrics:
- entity_precision_recall: how well the pipeline identifies gold companies
- hallucination_rate: fraction of predicted companies with no source support
- theme_coverage: fraction of gold themes covered by predicted theme names
- theme_purity: within-theme tag homogeneity from retrieved source documents

Usage:
    # Single-model eval, useful for CI and cache warming
    python eval/evaluator.py --model gemini --cache-dir eval/cache/gemini

    # Multi-model eval report input, useful for local comparison
    python eval/evaluator.py --models gpt llama gemini --write-results
"""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
import pathlib
import sys
import time
from typing import Callable, Optional

EVAL_PATH = pathlib.Path(__file__).parent.parent / "data" / "eval" / "queries.json"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DEFAULT_MODELS = ["gpt", "llama", "gemini"]

# When executed as `python eval/evaluator.py`, Python sets `sys.path[0]` to the
# `eval/` directory, which prevents importing sibling packages like `pipeline/`.
# Adding the repo root keeps imports working both locally and in CI.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables for local runs. GitHub Actions injects secrets as
# environment variables, but loading `.env` keeps local CLI commands simple.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    # If python-dotenv is unavailable, fall back to the existing environment.
    pass


def entity_precision_recall(
    gold_entities: list[str], predicted_entities: list[str]
) -> dict:
    """
    Compute precision and recall between gold and predicted company sets.

    Matching is case-insensitive substring matching so "YC" can match
    "Y Combinator" and partial company names do not unfairly count as misses.

    Args:
        gold_entities: Hand-labeled expected company/entity names.
        predicted_entities: Company/entity names produced by the pipeline.

    Returns:
        Dict with keys precision, recall, f1.
    """
    gold_lower = {e.lower() for e in gold_entities}
    pred_lower = {e.lower() for e in predicted_entities}

    true_positives = sum(
        1
        for p in pred_lower
        if any(p in g or g in p for g in gold_lower)
    )

    precision = true_positives / len(pred_lower) if pred_lower else 0.0
    recall = true_positives / len(gold_lower) if gold_lower else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def theme_coverage(gold_themes: list[str], predicted_themes: list[str]) -> float:
    """
    Compute the fraction of gold themes covered by predicted theme names.

    Args:
        gold_themes: Hand-labeled theme names for the query.
        predicted_themes: Theme names produced by the pipeline.

    Returns:
        Float between 0.0 and 1.0. Matching is case-insensitive substring
        matching, e.g. "RAG" matches "RAG and retrieval".
    """
    if not gold_themes:
        return 1.0
    gold_lower = [t.lower() for t in gold_themes]
    pred_lower = [t.lower() for t in predicted_themes]
    covered = sum(
        1 for g in gold_lower
        if any(g in p or p in g for p in pred_lower)
    )
    return round(covered / len(gold_lower), 3)


def theme_purity(reviewed_map: list[dict], retrieved_docs: list[dict]) -> float:
    """
    Estimate cluster quality using retrieved-document tag homogeneity.

    Each predicted theme is linked back to retrieved documents using citations
    first and company-name mentions as a fallback. The theme score is the most
    common tag count divided by total tag count. Themes supported by docs that
    mostly share one tag are treated as more coherent.

    Args:
        reviewed_map: Predicted themes. Keys per entry: theme_name, companies,
            rationale, citations.
        retrieved_docs: Scout documents. Keys per entry: title, source_url,
            tags, publish_date, snippet, score.

    Returns:
        Average theme purity between 0.0 and 1.0. Themes with no matched tags
        are ignored because they provide no corpus-tag signal.
    """
    theme_scores = []
    for theme in reviewed_map:
        citations = {c.lower() for c in theme.get("citations", [])}
        companies = [c.lower() for c in theme.get("companies", [])]
        matched_tags: list[str] = []

        for doc in retrieved_docs:
            title = doc.get("title", "").lower()
            snippet = doc.get("snippet", "").lower()
            cited = title in citations
            mentions_company = any(c in title or c in snippet for c in companies)
            if cited or mentions_company:
                matched_tags.extend(tag.lower() for tag in doc.get("tags", []))

        if not matched_tags:
            continue

        tag_counts = {
            tag: matched_tags.count(tag)
            for tag in set(matched_tags)
        }
        theme_scores.append(max(tag_counts.values()) / len(matched_tags))

    if not theme_scores:
        return 0.0
    return round(sum(theme_scores) / len(theme_scores), 3)


def hallucination_rate(
    reviewed_map: list[dict], retrieved_docs: list[dict]
) -> float:
    """
    Compute the fraction of predicted companies with no retrieved-doc support.

    Args:
        reviewed_map: Predicted themes. Keys per entry: theme_name, companies,
            rationale, citations.
        retrieved_docs: Scout documents. Keys per entry: title, source_url,
            tags, publish_date, snippet, score.

    Returns:
        Float between 0.0 (no hallucinations) and 1.0 (all companies are
        unsupported by retrieved docs).
    """
    all_companies = [
        company
        for theme in reviewed_map
        for company in theme.get("companies", [])
    ]
    if not all_companies:
        return 0.0

    doc_text = " ".join(
        (d.get("snippet", "") + " " + d.get("title", "")).lower()
        for d in retrieved_docs
    )
    hallucinated = sum(
        1 for c in all_companies if c.lower() not in doc_text
    )
    return round(hallucinated / len(all_companies), 3)


def _total_cost(timings: dict) -> float:
    """
    Sum LLM cost across the per-agent timings dict.

    Args:
        timings: Per-agent stats dict keyed by scout, mapper, critic.

    Returns:
        Total USD cost reported by LLM-backed agents.
    """
    return round(sum(v.get("cost_usd", 0.0) for v in timings.values()), 6)


def _total_latency(timings: dict) -> float:
    """
    Sum per-agent latency across the timings dict in milliseconds.

    Args:
        timings: Per-agent stats dict keyed by scout, mapper, critic.

    Returns:
        Total latency in milliseconds.
    """
    return round(sum(v.get("latency_ms", 0.0) for v in timings.values()), 1)


def _aggregate_results(results: list[dict]) -> dict:
    """
    Compute aggregate metrics from per-query eval rows.

    Args:
        results: Per-query dicts with keys precision, recall, f1,
            hallucination_rate, theme_coverage, theme_purity, cost_usd,
            latency_ms, query_id.

    Returns:
        Dict containing mean quality metrics plus total/mean cost and latency.
    """
    n = len(results)
    if n == 0:
        raise ValueError("Cannot aggregate an empty eval result set.")

    return {
        "mean_precision": round(sum(r["precision"] for r in results) / n, 3),
        "mean_recall": round(sum(r["recall"] for r in results) / n, 3),
        "mean_f1": round(sum(r["f1"] for r in results) / n, 3),
        "mean_hallucination_rate": round(
            sum(r["hallucination_rate"] for r in results) / n, 3
        ),
        "mean_theme_coverage": round(
            sum(r["theme_coverage"] for r in results) / n, 3
        ),
        "mean_theme_purity": round(
            sum(r["theme_purity"] for r in results) / n, 3
        ),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1),
        "num_queries": n,
    }


def _save_eval_result(result: dict, results_dir: pathlib.Path = RESULTS_DIR) -> pathlib.Path:
    """
    Persist one model's eval output to ``eval/results/<model>_<timestamp>.json``.

    Args:
        result: Dict with keys model, timestamp, queries_path, per_query,
            aggregate.
        results_dir: Directory where result files should be written.

    Returns:
        Path to the JSON file written.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{result['model']}_{result['timestamp']}.json"
    output_path.write_text(json.dumps(result, indent=2))
    return output_path


def _invoke_pipeline(pipeline_fn: Callable[..., dict], query: str, model: str) -> dict:
    """
    Call either a model-aware or legacy single-argument pipeline function.

    Args:
        pipeline_fn: Callable accepting either ``(query, model=...)`` or just
            ``(query)``.
        query: Natural language eval query.
        model: Short model id requested by the eval runner.

    Returns:
        Pipeline result dict with keys reviewed_map, retrieved_docs, and
        optionally timings.
    """
    signature = inspect.signature(pipeline_fn)
    accepts_model = (
        "model" in signature.parameters
        or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    )
    if accepts_model:
        return pipeline_fn(query, model=model)
    return pipeline_fn(query)


def _parse_rate_limit_wait_seconds(error_message: str) -> float:
    """
    Parse provider 429 wait hints such as ``Please try again in 2m41.56s``.

    Args:
        error_message: Raw exception message from the provider wrapper.

    Returns:
        Seconds to sleep. Defaults to 180 seconds if no provider hint is found.
    """
    wait_s = 180.0
    marker = "Please try again in "
    if marker not in error_message:
        return wait_s

    try:
        tail = error_message.split(marker, 1)[1]
        num = ""
        for ch in tail:
            if ch.isdigit() or ch in ".m s":
                num += ch
            else:
                break
        num = num.strip()
        if "m" in num:
            m_part, s_part = num.split("m", 1)
            wait_s = float(m_part.strip()) * 60.0 + float(
                s_part.replace("s", "").strip()
            )
        elif num.endswith("s"):
            wait_s = float(num[:-1])
    except Exception:
        return 180.0
    return max(1.0, wait_s)


def run_eval(
    pipeline_fn: Callable[..., dict],
    queries_path: pathlib.Path = EVAL_PATH,
    model: str = "llama",
    cache_dir: Optional[pathlib.Path] = None,
    sleep_on_rate_limit: bool = False,
    save_results: bool = False,
    results_dir: pathlib.Path = RESULTS_DIR,
) -> dict:
    """
    Run the pipeline on every query in the eval set for one model.

    When ``cache_dir`` is provided, pipeline results are persisted as
    ``<cache_dir>/<query_id>.json`` on cache miss and read on later runs. This
    keeps CI from repeatedly spending provider tokens for unchanged eval rows.

    Args:
        pipeline_fn: Callable that accepts ``query`` and optionally ``model``
            and returns keys ``reviewed_map`` (list[dict]), ``retrieved_docs``
            (list[dict]), and optionally ``timings`` (dict).
        queries_path: Path to the JSON eval query file.
        model: Short model id to pass into the pipeline.
        cache_dir: Optional directory for per-query pipeline cache files.
        sleep_on_rate_limit: If True, sleep and retry on provider 429 errors.
        save_results: Whether to write ``eval/results/<model>_<timestamp>.json``.
        results_dir: Directory used when ``save_results`` is True.

    Returns:
        Dict with keys model, timestamp, queries_path, per_query, aggregate, and
        optionally output_path.
    """
    with open(queries_path) as f:
        queries = json.load(f)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for item in queries:
        query = item["query"]
        gold_entities = item["gold_entities"]
        gold_themes = item.get("gold_themes", [])
        print(f"[{model}] Running: {query[:60]}...")

        # keys: reviewed_map, retrieved_docs, timings. Cache stores raw
        # pipeline output so metric changes can be recomputed without model
        # calls, while prompt/model changes can invalidate by using a new dir.
        cache_file = (cache_dir / f"{item['id']}.json") if cache_dir else None

        while True:
            try:
                if cache_file and cache_file.exists():
                    pipeline_result = json.loads(cache_file.read_text())
                    print("  (cached)")
                else:
                    pipeline_result = _invoke_pipeline(pipeline_fn, query, model)
                    if cache_file:
                        cache_file.write_text(json.dumps(pipeline_result))

                reviewed_map = pipeline_result["reviewed_map"]
                retrieved_docs = pipeline_result["retrieved_docs"]
                timings = pipeline_result.get("timings", {})

                predicted_entities = [
                    company
                    for theme in reviewed_map
                    for company in theme.get("companies", [])
                ]
                predicted_themes = [
                    theme.get("theme_name", "") for theme in reviewed_map
                ]

                metrics = entity_precision_recall(gold_entities, predicted_entities)
                h_rate = hallucination_rate(reviewed_map, retrieved_docs)
                t_coverage = theme_coverage(gold_themes, predicted_themes)
                t_purity = theme_purity(reviewed_map, retrieved_docs)
                cost_usd = _total_cost(timings)
                latency_ms = _total_latency(timings)
                error = None
                break
            except Exception as e:
                msg = str(e)
                is_rate_limit = (
                    "rate_limit_exceeded" in msg
                    or "Error code: 429" in msg
                    or "429" in msg
                )
                if is_rate_limit and sleep_on_rate_limit:
                    wait_s = _parse_rate_limit_wait_seconds(msg)
                    print(
                        f"  Rate limited (429). Sleeping {round(wait_s, 1)}s "
                        "then retrying..."
                    )
                    time.sleep(wait_s)
                    continue

                print(f"  ERROR: {e}")
                metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                h_rate = 1.0
                t_coverage = 0.0
                t_purity = 0.0
                cost_usd = 0.0
                latency_ms = 0.0
                error = str(e)
                break

        results.append({
            **metrics,
            "hallucination_rate": h_rate,
            "theme_coverage": t_coverage,
            "theme_purity": t_purity,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "query_id": item["id"],
            "model": model,
            "error": error,
        })
        print(
            f"  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
            f"F1={metrics['f1']:.2f}  Halluc={h_rate:.2f}  "
            f"ThemeCov={t_coverage:.2f}  ThemePurity={t_purity:.2f}  "
            f"Cost=${cost_usd:.5f}  Latency={latency_ms / 1000:.1f}s"
        )

    aggregate = _aggregate_results(results)
    output = {
        "model": model,
        "timestamp": timestamp,
        "queries_path": str(queries_path),
        "per_query": results,
        "aggregate": aggregate,
    }
    if save_results:
        output_path = _save_eval_result(output, results_dir=results_dir)
        output["output_path"] = str(output_path)

    print(f"\n=== Aggregate Results: {model} ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    if save_results:
        print(f"  output_path: {output['output_path']}")
    return output


def run_eval_for_models(
    pipeline_fn: Callable[..., dict],
    models: list[str] | None = None,
    queries_path: pathlib.Path = EVAL_PATH,
    results_dir: pathlib.Path = RESULTS_DIR,
    cache_root: pathlib.Path | None = None,
    sleep_on_rate_limit: bool = False,
) -> dict:
    """
    Run the eval suite for multiple models and write comparison-ready JSON.

    Args:
        pipeline_fn: Callable that accepts ``query`` and ``model`` and returns
            keys reviewed_map, retrieved_docs, and timings.
        models: Model ids to evaluate. Defaults to gpt, llama, gemini.
        queries_path: Path to the gold eval query set.
        results_dir: Directory where JSON result files should be written.
        cache_root: Optional root directory for per-model cache dirs. If set,
            each model uses ``<cache_root>/<model>``.
        sleep_on_rate_limit: Whether to sleep/retry on provider 429 errors.

    Returns:
        Dict with keys timestamp, models, results, output_paths, latest_path.
    """
    selected_models = models or DEFAULT_MODELS
    all_results = {}
    output_paths = {}

    for model in selected_models:
        cache_dir = cache_root / model if cache_root else None
        result = run_eval(
            pipeline_fn,
            queries_path=queries_path,
            model=model,
            cache_dir=cache_dir,
            sleep_on_rate_limit=sleep_on_rate_limit,
            save_results=False,
        )
        output_path = _save_eval_result(result, results_dir=results_dir)
        result["output_path"] = str(output_path)
        all_results[model] = result
        output_paths[model] = str(output_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    latest_path = results_dir / "latest.json"
    latest = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "models": selected_models,
        "results": all_results,
        "output_paths": output_paths,
    }
    latest_path.write_text(json.dumps(latest, indent=2))
    latest["latest_path"] = str(latest_path)
    print(f"\nWrote latest multi-model summary to {latest_path}")
    return latest


if __name__ == "__main__":
    import argparse
    from agents.llm_client import ModelId
    from pipeline.orchestrator import run_pipeline_full

    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument(
        "--model",
        choices=["claude", "gpt", "llama", "gemini"],
        default="llama",
        help=(
            "LLM used by Mapper and Critic for a single-model eval. Defaults "
            "to 'llama' (Groq)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["claude", "gpt", "llama", "gemini"],
        default=None,
        help=(
            "Run a multi-model eval for the listed models and write "
            "eval/results/latest.json. Example: --models gpt llama gemini"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for caching single-model per-query pipeline results. "
            "Example: eval/cache/llama"
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help=(
            "Root cache directory for multi-model evals; each model uses "
            "<cache-root>/<model>."
        ),
    )
    parser.add_argument(
        "--sleep-on-429",
        action="store_true",
        help=(
            "If set, automatically sleeps and retries when the LLM provider "
            "returns a 429 rate-limit error. Useful locally; avoid in CI."
        ),
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help=(
            "Write eval/results/<model>_<timestamp>.json for single-model "
            "runs. Multi-model runs always write result files."
        ),
    )
    args = parser.parse_args()

    if args.models:
        cache_root = pathlib.Path(args.cache_root) if args.cache_root else None
        run_eval_for_models(
            run_pipeline_full,
            models=args.models,
            cache_root=cache_root,
            sleep_on_rate_limit=bool(args.sleep_on_429),
        )
    else:
        model: ModelId = args.model
        cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else None
        run_eval(
            lambda q: run_pipeline_full(q, model=model),
            model=model,
            cache_dir=cache_dir,
            sleep_on_rate_limit=bool(args.sleep_on_429),
            save_results=bool(args.write_results),
        )
