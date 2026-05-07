"""
Generate eval comparison artifacts from per-model result JSON files.

Inputs:
    eval/results/latest.json, written by ``eval/evaluator.py``.

Outputs:
    eval/report.md  - compact model comparison table and per-model result paths.
    eval/report.png - scatter chart of accuracy vs. cost, with latency labels.
"""

from __future__ import annotations

import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "latest.json"
DEFAULT_REPORT_MD = pathlib.Path(__file__).parent / "report.md"
DEFAULT_REPORT_PNG = pathlib.Path(__file__).parent / "report.png"


def load_latest_results(results_path: pathlib.Path = DEFAULT_RESULTS_PATH) -> dict:
    """
    Load the latest multi-model eval summary JSON.

    Args:
        results_path: Path to ``latest.json`` produced by ``run_eval_for_models``.

    Returns:
        Dict with keys timestamp, models, results, output_paths.

    Raises:
        FileNotFoundError: If the eval runner has not produced latest.json yet.
        ValueError: If the file does not contain the expected result keys.
    """
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} does not exist. Run `python eval/evaluator.py` first."
        )

    data = json.loads(results_path.read_text())
    required_keys = {"timestamp", "models", "results", "output_paths"}
    missing = required_keys - set(data)
    if missing:
        raise ValueError(f"{results_path} is missing required keys: {sorted(missing)}")
    return data


def render_markdown(latest: dict) -> str:
    """
    Render a markdown comparison report from eval results.

    Args:
        latest: Dict returned by ``load_latest_results``. Expected keys:
            timestamp, models, results, output_paths.

    Returns:
        Markdown string containing a model comparison table and source files.
    """
    lines = [
        "# Multi-Model Evaluation Report",
        "",
        f"Generated from eval run: `{latest['timestamp']}`",
        "",
        "## Model Comparison",
        "",
        "| Model | F1 | Precision | Recall | Hallucination | Theme Coverage | Theme Purity | Cost | Mean Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model in latest["models"]:
        aggregate = latest["results"][model]["aggregate"]
        lines.append(
            "| "
            f"{model} | "
            f"{aggregate['mean_f1']:.3f} | "
            f"{aggregate['mean_precision']:.3f} | "
            f"{aggregate['mean_recall']:.3f} | "
            f"{aggregate['mean_hallucination_rate']:.3f} | "
            f"{aggregate['mean_theme_coverage']:.3f} | "
            f"{aggregate['mean_theme_purity']:.3f} | "
            f"${aggregate['total_cost_usd']:.6f} | "
            f"{aggregate['mean_latency_ms'] / 1000:.1f}s |"
        )

    lines.extend([
        "",
        "## Source Result Files",
        "",
    ])
    for model in latest["models"]:
        lines.append(f"- `{model}`: `{latest['output_paths'][model]}`")

    lines.extend([
        "",
        "## Notes",
        "",
        "- `Hallucination` is lower-is-better.",
        "- `Theme Purity` is corpus-tag homogeneity inside predicted themes; higher is better.",
        "- Cost includes Mapper/Critic LLM calls and any JSON retry attempts.",
    ])
    return "\n".join(lines) + "\n"


def render_chart(latest: dict, output_path: pathlib.Path = DEFAULT_REPORT_PNG) -> None:
    """
    Render an accuracy-vs-cost chart to PNG.

    Args:
        latest: Dict returned by ``load_latest_results``.
        output_path: Destination PNG path.

    Raises:
        ImportError: If matplotlib is not installed. This is a hard failure
            because the project plan requires a PNG chart artifact.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required to generate eval/report.png. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from e

    models = latest["models"]
    f1_scores = [
        latest["results"][model]["aggregate"]["mean_f1"]
        for model in models
    ]
    costs = [
        latest["results"][model]["aggregate"]["total_cost_usd"]
        for model in models
    ]
    latencies = [
        latest["results"][model]["aggregate"]["mean_latency_ms"] / 1000
        for model in models
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(costs, f1_scores, s=120)
    for model, cost, f1, latency in zip(models, costs, f1_scores, latencies):
        plt.annotate(
            f"{model}\n{latency:.1f}s",
            (cost, f1),
            textcoords="offset points",
            xytext=(8, 8),
        )
    plt.xlabel("Total eval cost (USD)")
    plt.ylabel("Mean entity F1")
    plt.title("Accuracy vs. Cost by Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_report(
    results_path: pathlib.Path = DEFAULT_RESULTS_PATH,
    report_md_path: pathlib.Path = DEFAULT_REPORT_MD,
    report_png_path: pathlib.Path = DEFAULT_REPORT_PNG,
) -> dict:
    """
    Generate markdown and PNG eval reports from the latest result JSON.

    Args:
        results_path: Path to latest multi-model result JSON.
        report_md_path: Destination markdown report path.
        report_png_path: Destination PNG chart path.

    Returns:
        Dict with keys report_md_path and report_png_path.
    """
    latest = load_latest_results(results_path)
    report_md_path.write_text(render_markdown(latest))
    render_chart(latest, report_png_path)
    return {
        "report_md_path": str(report_md_path),
        "report_png_path": str(report_png_path),
    }


if __name__ == "__main__":
    paths = generate_report()
    print(f"Wrote {paths['report_md_path']}")
    print(f"Wrote {paths['report_png_path']}")
