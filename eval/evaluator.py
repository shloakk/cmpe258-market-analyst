"""
Evaluation metrics for the multi-agent market research pipeline.

Metrics:
- entity_precision_recall: how well the pipeline identifies the gold companies
- hallucination_rate: fraction of predicted companies with no source document support
- run_eval: runs all queries in the eval set and prints aggregate results

Usage:
    python eval/evaluator.py
"""

import json
import pathlib
from typing import Callable

EVAL_PATH = pathlib.Path(__file__).parent.parent / "data" / "eval" / "queries.json"


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
    pipeline_fn: Callable[[str], list[dict]],
    queries_path: pathlib.Path = EVAL_PATH,
) -> dict:
    """
    Run the pipeline on every query in the eval set and compute aggregate metrics.

    Args:
        pipeline_fn: Callable that takes a query string and returns a reviewed_map.
        queries_path: Path to the JSON eval query file.

    Returns:
        Dict with mean precision, recall, f1, and hallucination rate.
    """
    with open(queries_path) as f:
        queries = json.load(f)

    results = []
    for item in queries:
        query = item["query"]
        gold_entities = item["gold_entities"]
        print(f"Running: {query[:60]}...")

        try:
            reviewed_map = pipeline_fn(query)
            predicted_entities = [
                company
                for theme in reviewed_map
                for company in theme.get("companies", [])
            ]
            metrics = entity_precision_recall(gold_entities, predicted_entities)
            h_rate = hallucination_rate(reviewed_map, [])
        except Exception as e:
            print(f"  ERROR: {e}")
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            h_rate = 1.0

        results.append({**metrics, "hallucination_rate": h_rate, "query_id": item["id"]})
        print(
            f"  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
            f"F1={metrics['f1']:.2f}  Halluc={h_rate:.2f}"
        )

    n = len(results)
    aggregate = {
        "mean_precision": round(sum(r["precision"] for r in results) / n, 3),
        "mean_recall": round(sum(r["recall"] for r in results) / n, 3),
        "mean_f1": round(sum(r["f1"] for r in results) / n, 3),
        "mean_hallucination_rate": round(
            sum(r["hallucination_rate"] for r in results) / n, 3
        ),
        "num_queries": n,
    }
    print("\n=== Aggregate Results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    return aggregate


if __name__ == "__main__":
    from pipeline.orchestrator import run_pipeline
    run_eval(run_pipeline)
