"""
Evaluation metrics for the multi-agent market research pipeline.

Metrics:
- entity_precision_recall: how well the pipeline identifies the gold companies
- hallucination_rate: fraction of predicted companies with no source document support
- theme_coverage: fraction of gold themes covered by predicted theme names
- run_eval: runs all queries in the eval set and prints aggregate results

Usage:
    python eval/evaluator.py [--model gemini] [--cache-dir eval/cache/gemini]
"""

import json
import pathlib
import sys
import time
from typing import Callable, Optional

EVAL_PATH = pathlib.Path(__file__).parent.parent / "data" / "eval" / "queries.json"

# When executed as `python eval/evaluator.py`, Python sets `sys.path[0]` to the
# `eval/` directory, which prevents importing sibling packages like `pipeline/`.
# Adding the repo root keeps imports working both locally and in CI.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables (notably GROQ_API_KEY) for local runs.
# In GitHub Actions we also rely on environment variables, but loading `.env`
# locally makes `python eval/evaluator.py` work out of the box.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    # If python-dotenv isn't installed, we fall back to the existing environment.
    pass


def entity_precision_recall(
    gold_entities: list[str], predicted_entities: list[str]
) -> dict:
    """
    Compute precision and recall between gold and predicted company sets.
    Matching is case-insensitive substring match (handles partial names).

    Returns:
        {"precision": float, "recall": float, "f1": float}
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
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def theme_coverage(gold_themes: list[str], predicted_themes: list[str]) -> float:
    """
    Fraction of gold themes covered by at least one predicted theme name.
    Matching is case-insensitive substring (e.g. "RAG" matches "RAG and retrieval").

    Returns:
        Float between 0.0 (no themes covered) and 1.0 (all gold themes covered).
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


def hallucination_rate(
    reviewed_map: list[dict], retrieved_docs: list[dict]
) -> float:
    """
    Fraction of companies in the reviewed map that appear in NO retrieved document.
    A company is considered grounded if its name appears (case-insensitive) in any
    document's snippet or title.

    Returns:
        Float between 0.0 (no hallucinations) and 1.0 (all hallucinated).
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


def run_eval(
    pipeline_fn: Callable[[str], dict],
    queries_path: pathlib.Path = EVAL_PATH,
    cache_dir: Optional[pathlib.Path] = None,
    sleep_on_rate_limit: bool = False,
) -> dict:
    """
    Run the pipeline on every query in the eval set and compute aggregate metrics.

    When ``cache_dir`` is provided, pipeline results are persisted as
    ``<cache_dir>/<query_id>.json`` on the first call and read back on
    subsequent calls, eliminating redundant LLM API calls across CI runs.
    Cache files should be committed to the repo so CI never hits rate limits.

    Args:
        pipeline_fn: Callable that takes a query string and returns a dict with
            keys ``reviewed_map`` (list[dict]) and ``retrieved_docs`` (list[dict]).
        queries_path: Path to the JSON eval query file.
        cache_dir: Optional directory for caching per-query pipeline results.
            If None, caching is disabled and the pipeline is always called live.

    Returns:
        Dict with mean precision, recall, f1, hallucination rate, and theme coverage.
    """
    with open(queries_path) as f:
        queries = json.load(f)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for item in queries:
        query = item["query"]
        gold_entities = item["gold_entities"]
        gold_themes = item.get("gold_themes", [])
        print(f"Running: {query[:60]}...")

        # Cache is keyed by query_id so renaming a query (same id) correctly
        # invalidates the entry, while reordering queries does not.
        cache_file = (cache_dir / f"{item['id']}.json") if cache_dir else None

        while True:
            try:
                if cache_file and cache_file.exists():
                    pipeline_result = json.loads(cache_file.read_text())
                    print("  (cached)")
                else:
                    pipeline_result = pipeline_fn(query)
                    if cache_file:
                        cache_file.write_text(json.dumps(pipeline_result))

                reviewed_map = pipeline_result["reviewed_map"]
                retrieved_docs = pipeline_result["retrieved_docs"]

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
                break
            except Exception as e:
                msg = str(e)
                is_rate_limit = ("rate_limit_exceeded" in msg) or ("Error code: 429" in msg)
                if is_rate_limit and sleep_on_rate_limit:
                    # Groq errors often include: "Please try again in 2m41.56s"
                    wait_s = 180.0
                    try:
                        marker = "Please try again in "
                        if marker in msg:
                            tail = msg.split(marker, 1)[1]
                            num = ""
                            for ch in tail:
                                if ch.isdigit() or ch in ".m s":
                                    num += ch
                                else:
                                    break
                            num = num.strip()
                            if "m" in num:
                                m_part, s_part = num.split("m", 1)
                                wait_s = float(m_part.strip()) * 60.0 + float(s_part.replace("s", "").strip())
                            elif num.endswith("s"):
                                wait_s = float(num[:-1])
                    except Exception:
                        pass

                    wait_s = max(1.0, wait_s)
                    print(f"  Rate limited (429). Sleeping {round(wait_s, 1)}s then retrying...")
                    time.sleep(wait_s)
                    continue

                print(f"  ERROR: {e}")
                metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                h_rate = 1.0
                t_coverage = 0.0
                break

        results.append({
            **metrics,
            "hallucination_rate": h_rate,
            "theme_coverage": t_coverage,
            "query_id": item["id"],
        })
        print(
            f"  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
            f"F1={metrics['f1']:.2f}  Halluc={h_rate:.2f}  "
            f"ThemeCov={t_coverage:.2f}"
        )

    n = len(results)
    aggregate = {
        "mean_precision": round(sum(r["precision"] for r in results) / n, 3),
        "mean_recall": round(sum(r["recall"] for r in results) / n, 3),
        "mean_f1": round(sum(r["f1"] for r in results) / n, 3),
        "mean_hallucination_rate": round(
            sum(r["hallucination_rate"] for r in results) / n, 3
        ),
        "mean_theme_coverage": round(
            sum(r["theme_coverage"] for r in results) / n, 3
        ),
        "num_queries": n,
    }
    print("\n=== Aggregate Results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    return aggregate


if __name__ == "__main__":
    import argparse
    from pipeline.orchestrator import run_pipeline_full
    from agents.llm_client import ModelId

    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument(
        "--model",
        choices=["gemini", "llama", "qwen", "nemotron"],
        default="gemini",
        help=(
            "LLM used by Mapper and Critic. Defaults to 'gemini' because the "
            "local demo path uses Google AI Studio. Use --model llama (Groq) "
            "or --model qwen/--model nemotron (OpenRouter) for free "
            "open-weight comparison."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for caching per-query pipeline results. Results are "
            "written on cache miss and read on cache hit, so repeated runs "
            "(e.g. in CI) never re-call the LLM. Commit the cache directory "
            "to avoid hitting daily token limits across runs. "
            "Example: eval/cache/gemini"
        ),
    )
    parser.add_argument(
        "--sleep-on-429",
        action="store_true",
        help=(
            "If set, automatically sleeps and retries when the LLM provider "
            "returns a 429 rate-limit error. Useful for local runs; avoid in CI."
        ),
    )
    args = parser.parse_args()
    model: ModelId = args.model
    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else None
    run_eval(
        lambda q: run_pipeline_full(q, model=model),
        cache_dir=cache_dir,
        sleep_on_rate_limit=bool(args.sleep_on_429),
    )
