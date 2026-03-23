"""Module 9: Evaluation metrics and plots."""

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results_log: List[Dict], window_size: int = 50) -> Dict:
    """Compute evaluation metrics from a per-problem results log.

    Args:
        results_log: List of per-problem dicts from the agent/baseline runs.
        window_size: Window for rolling accuracy computation.

    Returns:
        Dict of computed metrics.
    """
    n = len(results_log)
    if n == 0:
        return {}

    successes = [r["success"] for r in results_log]
    overall_accuracy = sum(successes) / n

    # Rolling accuracy
    rolling_acc = []
    for i in range(n):
        start = max(0, i - window_size + 1)
        window = successes[start:i + 1]
        rolling_acc.append(sum(window) / len(window))

    # Accuracy by difficulty bucket
    buckets = {
        "800-1199": [],
        "1200-1599": [],
        "1600-1999": [],
        "2000+": [],
    }
    for r in results_log:
        d = r.get("difficulty", 0)
        if d < 1200:
            buckets["800-1199"].append(r["success"])
        elif d < 1600:
            buckets["1200-1599"].append(r["success"])
        elif d < 2000:
            buckets["1600-1999"].append(r["success"])
        else:
            buckets["2000+"].append(r["success"])

    bucket_accuracy = {
        k: (sum(v) / len(v) if v else None) for k, v in buckets.items()
    }

    # Accuracy by method
    method_counts: Dict[str, Dict] = {}
    for r in results_log:
        m = r.get("method", "unknown")
        method_counts.setdefault(m, {"total": 0, "success": 0})
        method_counts[m]["total"] += 1
        if r["success"]:
            method_counts[m]["success"] += 1

    method_accuracy = {
        m: v["success"] / v["total"] for m, v in method_counts.items()
    }

    # Retrieval precision: for successful adapted runs, did the retrieved entry
    # share any algorithm tag with the problem?
    retrieval_precision_values = []
    for r in results_log:
        if not r.get("method", "").startswith("adapted_"):
            continue
        gt_tags = set(r.get("ground_truth_tags", []))
        # We don't store per-attempt retrieved tags in the log directly,
        # but approximate with whether method succeeded.
        # Full precision requires the retrieved entry's strategy tags — computed
        # externally if the full entries are available.
        retrieval_precision_values.append(1 if r["success"] else 0)

    retrieval_precision = (
        sum(retrieval_precision_values) / len(retrieval_precision_values)
        if retrieval_precision_values else None
    )

    return {
        "overall_accuracy": overall_accuracy,
        "rolling_accuracy": rolling_acc,
        "bucket_accuracy": bucket_accuracy,
        "method_accuracy": method_accuracy,
        "method_counts": method_counts,
        "retrieval_precision_proxy": retrieval_precision,
        "total_problems": n,
        "total_successes": sum(successes),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curves(
    logs: Dict[str, List[Dict]],
    window_size: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """Figure 1: rolling accuracy over time for multiple runs.

    Args:
        logs: Dict of {label: results_log} — one entry per method/baseline.
        window_size: Rolling window size.
        save_path: If provided, save to this path. Otherwise display.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, results_log in logs.items():
        metrics = compute_metrics(results_log, window_size=window_size)
        x = list(range(1, len(results_log) + 1))
        ax.plot(x, metrics["rolling_accuracy"], label=label, linewidth=2)

    ax.set_xlabel("Problems Seen", fontsize=13)
    ax.set_ylabel(f"Rolling Accuracy (window={window_size})", fontsize=13)
    ax.set_title("Learning Curve: Rolling Accuracy Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved learning curves to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_difficulty_accuracy(
    logs: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
) -> None:
    """Figure 2: accuracy by difficulty bucket, one bar group per method."""
    buckets = ["800-1199", "1200-1599", "1600-1999", "2000+"]
    labels = list(logs.keys())
    x = np.arange(len(buckets))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        metrics = compute_metrics(logs[label])
        values = [
            metrics["bucket_accuracy"].get(b) or 0.0 for b in buckets
        ]
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=label)

    ax.set_xlabel("Difficulty Rating", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Accuracy by Difficulty Bucket", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved difficulty accuracy to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_method_breakdown(
    results_log: List[Dict],
    window_size: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """Figure 3: stacked area chart of fraction solved by each method over time."""
    methods = ["adapted_1", "adapted_2", "adapted_3", "free", "failed"]
    n = len(results_log)
    if n == 0:
        return

    # Build windowed fractions
    x = list(range(window_size, n + 1))
    fractions: Dict[str, List[float]] = {m: [] for m in methods}

    for i in range(window_size - 1, n):
        window = results_log[max(0, i - window_size + 1):i + 1]
        total = len(window)
        for m in methods:
            count = sum(1 for r in window if r.get("method") == m)
            fractions[m].append(count / total)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(x))
    colors = ["#2ecc71", "#27ae60", "#16a085", "#3498db", "#e74c3c"]
    for m, color in zip(methods, colors):
        vals = np.array(fractions[m])
        ax.fill_between(x, bottoms, bottoms + vals, label=m, alpha=0.8, color=color)
        bottoms += vals

    ax.set_xlabel("Problems Seen", fontsize=13)
    ax.set_ylabel("Fraction of Problems", fontsize=13)
    ax.set_title("Method Breakdown Over Time", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved method breakdown to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Convenience: generate all plots from saved JSON logs
# ---------------------------------------------------------------------------

def generate_all_plots(
    main_log_path: str,
    baseline_log_paths: Dict[str, str],
    output_dir: str = "plots",
) -> None:
    """Load result logs from disk and produce all paper-ready figures.

    Args:
        main_log_path: Path to the main agent's final_results.json.
        baseline_log_paths: Dict of {label: path_to_final_results.json}.
        output_dir: Directory to save figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load_log(path: str) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        return data.get("per_problem_log", [])

    all_logs: Dict[str, List[Dict]] = {}
    all_logs["Strategy Adaptation (ours)"] = load_log(main_log_path)
    for label, path in baseline_log_paths.items():
        all_logs[label] = load_log(path)

    plot_learning_curves(all_logs, save_path=str(out / "fig1_learning_curves.png"))
    plot_difficulty_accuracy(all_logs, save_path=str(out / "fig2_difficulty_accuracy.png"))

    main_log = load_log(main_log_path)
    plot_method_breakdown(main_log, save_path=str(out / "fig3_method_breakdown.png"))

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Method':<30} {'Acc':>8} {'N':>6}")
    print("-" * 46)
    for label, log in all_logs.items():
        m = compute_metrics(log)
        print(f"{label:<30} {m['overall_accuracy']:>7.1%} {m['total_problems']:>6}")
