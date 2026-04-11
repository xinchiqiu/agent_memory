#!/usr/bin/env python3
"""Experiment runner — orchestrates all experiments for the NeurIPS paper.

Usage
-----
# Run all experiments sequentially:
    python run_experiments.py --all --dataset_dir dataset_cc/

# Run a specific experiment:
    python run_experiments.py --exp learning_curve --dataset_dir dataset_cc/

# Run granularity ablation only:
    python run_experiments.py --exp granularity --dataset_dir dataset_cc/

# Run with a specific backend:
    python run_experiments.py --all --dataset_dir dataset_cc/ --backend vllm

# Run with a specific seed count (for quick testing):
    python run_experiments.py --exp learning_curve --dataset_dir dataset_cc/ --seed_count 50 --eval_count 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG
from data_structures import Problem
from data_collection.dataset_utils import (
    load_problem, load_split, load_all_problems, dict_to_problem, dict_to_seed_tuples,
)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _method_completed(log_dir: str) -> bool:
    """Check if a method already has final results (for resume support)."""
    return Path(log_dir).joinpath("final_results.json").exists()


def _load_completed_results(log_dir: str) -> Optional[Dict]:
    """Load previously completed results from a method's log dir."""
    path = Path(log_dir) / "final_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_method_results(log_dir: str, results: Dict) -> None:
    """Save results for a single method so it can be skipped on resume."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(log_dir) / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_splits(dataset_dir: str) -> Dict[str, List[str]]:
    """Load seed/eval/test splits as lists of problem IDs."""
    splits = {}
    for name in ("seed", "eval", "test"):
        ids = load_split(name, dataset_dir)
        splits[name] = ids
        logging.info(f"  {name}: {len(ids)} problem IDs")
    return splits


def load_problems_for_ids(
    problem_ids: List[str],
    dataset_dir: str,
    require_statement: bool = True,
    require_solutions: bool = False,
    max_count: Optional[int] = None,
) -> List[dict]:
    """Load problem dicts for a list of IDs, filtering by data quality."""
    problems = []
    for pid in problem_ids:
        d = load_problem(pid, dataset_dir)
        if d is None:
            continue
        if require_statement and not d.get("statement"):
            continue
        if require_solutions and not d.get("reference_solutions"):
            continue
        problems.append(d)
        if max_count and len(problems) >= max_count:
            break
    return problems


def prepare_seed_tuples(
    problem_dicts: List[dict],
    max_count: Optional[int] = None,
) -> List[Tuple]:
    """Convert problem dicts to (Problem, code, language) tuples for seeding."""
    tuples = []
    for d in problem_dicts:
        ts = dict_to_seed_tuples(d)
        if ts:
            tuples.append(ts[0])  # Use first reference solution
            if max_count and len(tuples) >= max_count:
                break
    return tuples


def prepare_eval_problems(
    problem_dicts: List[dict],
    max_count: Optional[int] = None,
) -> List[Problem]:
    """Convert problem dicts to Problem objects for evaluation."""
    problems = []
    for d in problem_dicts:
        if not d.get("sample_tests"):
            continue
        problems.append(dict_to_problem(d))
        if max_count and len(problems) >= max_count:
            break
    return problems


# ---------------------------------------------------------------------------
# Experiment 1: Learning Curve
# ---------------------------------------------------------------------------

def run_learning_curve(
    dataset_dir: str,
    seed_count: int = 200,
    eval_count: int = 500,
    num_seeds: int = 3,
    backend: Optional[str] = None,
    output_dir: str = "results/exp1_learning_curve",
):
    """Experiment 1: Learning curve — does the agent improve over time?

    Runs: Full system, No Memory, Random Retrieval, Full History, Tag Oracle.
    Each with `num_seeds` random seeds for statistical rigor.
    """
    from llm_client import create_llm_client
    from encoder import create_encoder
    from agent import StrategyAdaptationAgent
    from baselines import (
        NoMemoryBaseline, RandomRetrievalBaseline,
        FullHistoryBaseline, TagOracleBaseline,
    )
    from evaluation import compute_metrics, plot_learning_curves

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 60)
    logging.info("EXPERIMENT 1: Learning Curve")
    logging.info("=" * 60)

    # Load data
    splits = load_dataset_splits(dataset_dir)
    seed_dicts = load_problems_for_ids(
        splits["seed"], dataset_dir, require_solutions=True, max_count=seed_count,
    )
    eval_dicts = load_problems_for_ids(
        splits["eval"], dataset_dir, require_statement=True, max_count=eval_count,
    )
    seed_tuples = prepare_seed_tuples(seed_dicts, max_count=seed_count)
    eval_problems = prepare_eval_problems(eval_dicts, max_count=eval_count)

    logging.info(f"Seed problems with solutions: {len(seed_tuples)}")
    logging.info(f"Eval problems: {len(eval_problems)}")

    client = create_llm_client(backend)
    encoder = create_encoder()

    all_logs: Dict[str, List[Dict]] = {}

    def _run_or_resume(label, runner_fn, log_subdir):
        """Run a method, or load previous results if already completed."""
        method_dir = str(out / log_subdir)
        prev = _load_completed_results(method_dir)
        if prev is not None:
            logging.info(f"\n--- {label} --- SKIPPED (already completed)")
            all_logs[label] = prev.get("per_problem_log", prev.get("results_log", []))
            return
        logging.info(f"\n--- {label} ---")
        results = runner_fn(method_dir)
        all_logs[label] = results["per_problem_log"]
        _save_method_results(method_dir, results)

    for seed_idx in range(num_seeds):
        logging.info(f"\n--- Seed {seed_idx + 1}/{num_seeds} ---")
        import random
        random.seed(42 + seed_idx)

        # 1. Full system (Strategy Adaptation Agent)
        def _run_full(d, _st=seed_tuples, _ep=eval_problems):
            a = StrategyAdaptationAgent(client, encoder=encoder, log_dir=d)
            a.seed_memory(_st)
            return a.run(_ep)
        _run_or_resume(
            f"Strategy Adaptation (seed {seed_idx})",
            _run_full, f"full_system_seed{seed_idx}",
        )

        # 2. No Memory baseline
        if seed_idx == 0:
            def _run_nm(d, _ep=eval_problems):
                nm = NoMemoryBaseline(client, log_dir=d)
                return nm.run(_ep)
            _run_or_resume("No Memory", _run_nm, "no_memory")

        # 3. Random Retrieval baseline
        def _run_rr(d, _st=seed_tuples, _ep=eval_problems):
            rr = RandomRetrievalBaseline(client, encoder=encoder, log_dir=d)
            rr.seed_memory(_st)
            return rr.run(_ep)
        _run_or_resume(
            f"Random Retrieval (seed {seed_idx})",
            _run_rr, f"random_retrieval_seed{seed_idx}",
        )

        # 4. Full History baseline
        def _run_fh(d, _st=seed_tuples, _ep=eval_problems):
            fh = FullHistoryBaseline(client, encoder=encoder, log_dir=d)
            fh.seed_memory(_st)
            return fh.run(_ep)
        _run_or_resume(
            f"Full History (seed {seed_idx})",
            _run_fh, f"full_history_seed{seed_idx}",
        )

        # 5. Tag Oracle baseline
        def _run_to(d, _st=seed_tuples, _ep=eval_problems):
            to_b = TagOracleBaseline(client, encoder=encoder, log_dir=d)
            to_b.seed_memory(_st)
            return to_b.run(_ep)
        _run_or_resume(
            f"Tag Oracle (seed {seed_idx})",
            _run_to, f"tag_oracle_seed{seed_idx}",
        )

    # Save all logs
    with open(out / "all_logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    # Generate learning curve plot (average across seeds)
    avg_logs = _average_across_seeds(all_logs, num_seeds)
    plot_learning_curves(avg_logs, save_path=str(out / "fig1_learning_curves.png"))

    logging.info(f"Experiment 1 complete. Results saved to {out}")


# ---------------------------------------------------------------------------
# Experiment 2: Granularity Ablation (KEY EXPERIMENT)
# ---------------------------------------------------------------------------

def run_granularity_ablation(
    dataset_dir: str,
    seed_count: int = 200,
    eval_count: int = 300,
    backend: Optional[str] = None,
    output_dir: str = "results/exp2_granularity",
):
    """Experiment 2: Granularity ablation — is strategy-level the right abstraction?

    Runs G1-G6 variants on the same eval set:
      G1: No retrieval (free generation)
      G2: Tag hints only
      G3: Strategy only (DEFAULT — our method)
      G4: Strategy + code snippet
      G5: Full solution
      G6: 3 full solutions (single-step)
    """
    from llm_client import create_llm_client
    from encoder import create_encoder
    from agent import StrategyAdaptationAgent
    from evaluation import compute_metrics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 60)
    logging.info("EXPERIMENT 2: Granularity Ablation")
    logging.info("=" * 60)

    # Load data
    splits = load_dataset_splits(dataset_dir)
    seed_dicts = load_problems_for_ids(
        splits["seed"], dataset_dir, require_solutions=True, max_count=seed_count,
    )
    eval_dicts = load_problems_for_ids(
        splits["eval"], dataset_dir, require_statement=True, max_count=eval_count,
    )
    seed_tuples = prepare_seed_tuples(seed_dicts, max_count=seed_count)
    eval_problems = prepare_eval_problems(eval_dicts, max_count=eval_count)

    client = create_llm_client(backend)
    encoder = create_encoder()

    granularity_modes = ["G1", "G2", "G3", "G4", "G5", "G6"]
    all_results = {}

    for mode in granularity_modes:
        mode_dir = str(out / f"granularity_{mode}")

        # Resume: skip completed modes
        prev = _load_completed_results(mode_dir)
        if prev is not None:
            logging.info(f"\n--- Granularity mode: {mode} --- SKIPPED (already completed)")
            all_results[mode] = prev
            continue

        logging.info(f"\n--- Granularity mode: {mode} ---")
        agent = StrategyAdaptationAgent(
            client, encoder=encoder,
            log_dir=mode_dir,
            granularity_mode=mode,
        )
        # G1 doesn't need memory, but seed anyway for fair comparison
        agent.seed_memory(seed_tuples)
        results = agent.run(eval_problems)
        all_results[mode] = results

        # Save per-mode results for resume
        _save_method_results(mode_dir, results)

        metrics = compute_metrics(results["per_problem_log"])
        logging.info(
            f"  {mode}: accuracy={metrics['overall_accuracy']:.3f} "
            f"({metrics['total_successes']}/{metrics['total_problems']})"
        )

    # Save results
    summary = {}
    for mode, results in all_results.items():
        metrics = compute_metrics(results["per_problem_log"])
        summary[mode] = {
            "overall_accuracy": metrics["overall_accuracy"],
            "total_successes": metrics["total_successes"],
            "total_problems": metrics["total_problems"],
            "bucket_accuracy": metrics.get("bucket_accuracy", {}),
        }

    with open(out / "granularity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 60)
    print("GRANULARITY ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Mode':<8} {'Description':<35} {'Accuracy':>10}")
    print("-" * 55)
    descriptions = {
        "G1": "No retrieval (free generation)",
        "G2": "Tag hints only (~20 tokens)",
        "G3": "Strategy only (~200 tokens) [OURS]",
        "G4": "Strategy + code snippet (~500 tokens)",
        "G5": "Full solution (~1000 tokens)",
        "G6": "3 full solutions (~3000 tokens)",
    }
    for mode in granularity_modes:
        acc = summary[mode]["overall_accuracy"]
        desc = descriptions.get(mode, "")
        marker = " <--" if mode == "G3" else ""
        print(f"{mode:<8} {desc:<35} {acc:>9.1%}{marker}")

    logging.info(f"Experiment 2 complete. Results saved to {out}")


# ---------------------------------------------------------------------------
# Experiment 3: Retrieval Quality
# ---------------------------------------------------------------------------

def run_retrieval_quality(
    dataset_dir: str,
    seed_count: int = 200,
    eval_count: int = 300,
    backend: Optional[str] = None,
    output_dir: str = "results/exp3_retrieval",
):
    """Experiment 3: Does technique-aware retrieval matter?

    Compares retrieval precision AND downstream solve rate for:
      - Base encoder (all-MiniLM-L6-v2)
      - Fine-tuned encoder (models/technique_encoder)
      - Random retrieval
      - Tag oracle (ceiling)
    """
    from llm_client import create_llm_client
    from encoder import ProblemEncoder, create_encoder
    from agent import StrategyAdaptationAgent
    from baselines import RandomRetrievalBaseline, TagOracleBaseline
    from retriever import Retriever
    from evaluation import compute_metrics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 60)
    logging.info("EXPERIMENT 3: Retrieval Quality")
    logging.info("=" * 60)

    splits = load_dataset_splits(dataset_dir)
    seed_dicts = load_problems_for_ids(
        splits["seed"], dataset_dir, require_solutions=True, max_count=seed_count,
    )
    eval_dicts = load_problems_for_ids(
        splits["eval"], dataset_dir, require_statement=True, max_count=eval_count,
    )
    seed_tuples = prepare_seed_tuples(seed_dicts, max_count=seed_count)
    eval_problems = prepare_eval_problems(eval_dicts, max_count=eval_count)

    client = create_llm_client(backend)

    methods_results = {}

    def _run_retrieval_method(label, log_subdir, make_runner):
        """Run or resume a retrieval method."""
        method_dir = str(out / log_subdir)
        prev = _load_completed_results(method_dir)
        if prev is not None:
            logging.info(f"\n--- Retrieval: {label} --- SKIPPED (already completed)")
            methods_results[label] = prev
            return
        logging.info(f"\n--- Retrieval: {label} ---")
        results = make_runner(method_dir)
        methods_results[label] = results
        _save_method_results(method_dir, results)

    # Base encoder
    base_encoder = ProblemEncoder()
    def _run_base(d):
        a = StrategyAdaptationAgent(client, encoder=base_encoder, log_dir=d)
        a.seed_memory(seed_tuples)
        return a.run(eval_problems)
    _run_retrieval_method("base_encoder", "base_encoder", _run_base)

    # Fine-tuned encoder (if available)
    ft_encoder = create_encoder(prefer_finetuned=True)
    if ft_encoder.model_name != base_encoder.model_name:
        def _run_ft(d):
            a = StrategyAdaptationAgent(client, encoder=ft_encoder, log_dir=d)
            a.seed_memory(seed_tuples)
            return a.run(eval_problems)
        _run_retrieval_method("finetuned_encoder", "finetuned_encoder", _run_ft)
    else:
        logging.warning("Fine-tuned encoder not available. Skipping.")

    # Random retrieval
    def _run_random(d):
        rr = RandomRetrievalBaseline(client, encoder=base_encoder, log_dir=d)
        rr.seed_memory(seed_tuples)
        return rr.run(eval_problems)
    _run_retrieval_method("random", "random_retrieval", _run_random)

    # Tag oracle
    def _run_oracle(d):
        to = TagOracleBaseline(client, encoder=base_encoder, log_dir=d)
        to.seed_memory(seed_tuples)
        return to.run(eval_problems)
    _run_retrieval_method("tag_oracle", "tag_oracle", _run_oracle)

    # Summary
    summary = {}
    for label, res in methods_results.items():
        per_problem = res.get("per_problem_log", [])
        if not per_problem:
            continue
        m = compute_metrics(per_problem)
        summary[label] = {
            "overall_accuracy": m["overall_accuracy"],
            "total_successes": m["total_successes"],
            "total_problems": m["total_problems"],
        }

    with open(out / "retrieval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("RETRIEVAL QUALITY RESULTS")
    print("=" * 50)
    print(f"{'Method':<25} {'Accuracy':>10}")
    print("-" * 37)
    for label, s in summary.items():
        print(f"{label:<25} {s['overall_accuracy']:>9.1%}")

    logging.info(f"Experiment 3 complete. Results saved to {out}")


# ---------------------------------------------------------------------------
# Experiment 4: Two-step vs Single-step Adaptation
# ---------------------------------------------------------------------------

def run_adaptation_ablation(
    dataset_dir: str,
    seed_count: int = 200,
    eval_count: int = 300,
    backend: Optional[str] = None,
    output_dir: str = "results/exp4_adaptation",
):
    """Experiment 4: Does explicit alignment reasoning help?

    Compares two-step (alignment + code gen) vs single-step (direct code gen with strategy).
    """
    from llm_client import create_llm_client
    from encoder import create_encoder
    from agent import StrategyAdaptationAgent
    from evaluation import compute_metrics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 60)
    logging.info("EXPERIMENT 4: Adaptation Ablation (Two-step vs Single-step)")
    logging.info("=" * 60)

    splits = load_dataset_splits(dataset_dir)
    seed_dicts = load_problems_for_ids(
        splits["seed"], dataset_dir, require_solutions=True, max_count=seed_count,
    )
    eval_dicts = load_problems_for_ids(
        splits["eval"], dataset_dir, require_statement=True, max_count=eval_count,
    )
    seed_tuples = prepare_seed_tuples(seed_dicts, max_count=seed_count)
    eval_problems = prepare_eval_problems(eval_dicts, max_count=eval_count)

    client = create_llm_client(backend)
    encoder = create_encoder()

    # Two-step (default G3)
    logging.info("\n--- Two-step adaptation (G3, default) ---")
    agent_two = StrategyAdaptationAgent(
        client, encoder=encoder,
        log_dir=str(out / "two_step"),
        granularity_mode="G3",
    )
    agent_two.seed_memory(seed_tuples)
    results_two = agent_two.run(eval_problems)

    # Single-step: use G6-style (inject strategy directly into code gen, skip alignment)
    # G6 already does single-step with full solutions; we make a variant with strategy-only
    logging.info("\n--- Single-step adaptation (strategy direct to code gen) ---")
    agent_single = StrategyAdaptationAgent(
        client, encoder=encoder,
        log_dir=str(out / "single_step"),
        granularity_mode="G6",  # Single-step variant
    )
    agent_single.seed_memory(seed_tuples)
    results_single = agent_single.run(eval_problems)

    summary = {}
    for label, res in [("two_step", results_two), ("single_step", results_single)]:
        m = compute_metrics(res["per_problem_log"])
        summary[label] = {
            "overall_accuracy": m["overall_accuracy"],
            "total_successes": m["total_successes"],
            "total_problems": m["total_problems"],
        }

    with open(out / "adaptation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("ADAPTATION ABLATION RESULTS")
    print("=" * 50)
    for label, s in summary.items():
        print(f"  {label}: {s['overall_accuracy']:.1%}")

    logging.info(f"Experiment 4 complete. Results saved to {out}")


# ---------------------------------------------------------------------------
# Experiment 5: Breakdown Analysis
# ---------------------------------------------------------------------------

def run_breakdown_analysis(
    results_dir: str = "results/exp1_learning_curve",
    output_dir: str = "results/exp5_breakdown",
):
    """Experiment 5: Where does memory help most?

    Analyzes existing results from Experiment 1 by:
      - Difficulty bucket
      - Method breakdown over time
      - Technique-seen vs technique-unseen
    """
    from evaluation import (
        compute_metrics, plot_difficulty_accuracy,
        plot_method_breakdown, plot_learning_curves,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 60)
    logging.info("EXPERIMENT 5: Breakdown Analysis")
    logging.info("=" * 60)

    # Load results from Experiment 1
    logs_path = Path(results_dir) / "all_logs.json"
    if not logs_path.exists():
        logging.error(f"No results found at {logs_path}. Run Experiment 1 first.")
        return

    with open(logs_path) as f:
        all_logs = json.load(f)

    # Average across seeds for the main system
    avg_logs = _average_across_seeds(all_logs, num_seeds=3)

    # Difficulty breakdown
    plot_difficulty_accuracy(avg_logs, save_path=str(out / "fig2_difficulty_accuracy.png"))

    # Method breakdown for main system
    main_key = [k for k in avg_logs if "Strategy" in k]
    if main_key:
        main_log = avg_logs[main_key[0]]
        plot_method_breakdown(main_log, save_path=str(out / "fig3_method_breakdown.png"))

    # Technique-seen vs technique-unseen analysis
    if main_key:
        _technique_novelty_analysis(avg_logs[main_key[0]], out)

    logging.info(f"Experiment 5 complete. Results saved to {out}")


def _technique_novelty_analysis(results_log: List[Dict], output_dir: Path):
    """Analyze accuracy for problems whose technique family was seen vs unseen in memory."""
    seen_tags = set()
    seen_correct, seen_total = 0, 0
    unseen_correct, unseen_total = 0, 0

    for r in results_log:
        gt_tags = set(r.get("ground_truth_tags", []))
        if gt_tags & seen_tags:
            # At least one technique was seen before
            seen_total += 1
            if r["success"]:
                seen_correct += 1
        else:
            unseen_total += 1
            if r["success"]:
                unseen_correct += 1
        # Add this problem's tags to seen set (regardless of success)
        seen_tags.update(gt_tags)

    analysis = {
        "technique_seen": {
            "accuracy": seen_correct / seen_total if seen_total else 0,
            "count": seen_total,
        },
        "technique_unseen": {
            "accuracy": unseen_correct / unseen_total if unseen_total else 0,
            "count": unseen_total,
        },
    }
    with open(output_dir / "technique_novelty.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nTechnique Novelty Analysis:")
    print(f"  Seen:   {analysis['technique_seen']['accuracy']:.1%} "
          f"({analysis['technique_seen']['count']} problems)")
    print(f"  Unseen: {analysis['technique_unseen']['accuracy']:.1%} "
          f"({analysis['technique_unseen']['count']} problems)")


# ---------------------------------------------------------------------------
# Generate all paper figures
# ---------------------------------------------------------------------------

def generate_paper_figures(
    results_base: str = "results",
    output_dir: str = "figures",
):
    """Generate all paper-ready figures from saved experiment results."""
    from evaluation import (
        compute_metrics, plot_learning_curves,
        plot_difficulty_accuracy, plot_method_breakdown,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: Learning curves (from Experiment 1)
    exp1_logs = Path(results_base) / "exp1_learning_curve" / "all_logs.json"
    if exp1_logs.exists():
        with open(exp1_logs) as f:
            all_logs = json.load(f)
        avg_logs = _average_across_seeds(all_logs, num_seeds=3)
        plot_learning_curves(avg_logs, save_path=str(out / "fig1_learning_curves.png"))
        plot_difficulty_accuracy(avg_logs, save_path=str(out / "fig2_difficulty_accuracy.png"))

    # Figure 4: Granularity comparison (from Experiment 2)
    exp2_summary = Path(results_base) / "exp2_granularity" / "granularity_summary.json"
    if exp2_summary.exists():
        _plot_granularity_bar_chart(exp2_summary, out / "fig4_granularity_ablation.png")

    logging.info(f"Paper figures saved to {out}")


def _plot_granularity_bar_chart(summary_path: Path, save_path: Path):
    """Plot a bar chart of accuracy by granularity mode."""
    import matplotlib.pyplot as plt
    import numpy as np

    with open(summary_path) as f:
        summary = json.load(f)

    modes = ["G1", "G2", "G3", "G4", "G5", "G6"]
    labels = [
        "No retrieval", "Tag hints", "Strategy\n(ours)",
        "Strategy +\nsnippet", "Full\nsolution", "3 full\nsolutions",
    ]
    accuracies = [summary.get(m, {}).get("overall_accuracy", 0) for m in modes]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95a5a6"] * 6
    colors[2] = "#2ecc71"  # Highlight G3 (our method)

    bars = ax.bar(range(len(modes)), accuracies, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Granularity Ablation: Information Level vs. Accuracy", fontsize=14)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(accuracies) * 1.15 if accuracies else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _average_across_seeds(all_logs: Dict, num_seeds: int) -> Dict[str, List[Dict]]:
    """Average results across random seeds for each method.

    Groups logs by method name (ignoring the seed suffix), then picks seed 0
    as the representative log (since per-problem order is the same).
    """
    averaged = {}
    seen_methods = set()

    for label, log in all_logs.items():
        # Strip " (seed N)" suffix
        base_label = label
        if "(seed " in label:
            base_label = label.rsplit(" (seed", 1)[0]

        if base_label not in seen_methods:
            seen_methods.add(base_label)
            averaged[base_label] = log  # Use first seed as representative

    return averaged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run NeurIPS experiments")
    parser.add_argument("--dataset_dir", type=str, default="dataset_cc/",
                        help="Path to the dataset directory")
    parser.add_argument("--exp", type=str, default=None,
                        choices=["learning_curve", "granularity", "retrieval",
                                 "adaptation", "breakdown", "figures"],
                        help="Run a specific experiment")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments sequentially")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["hf_local", "vllm", "openai", "anthropic"],
                        help="Override LLM backend")
    parser.add_argument("--seed_count", type=int, default=200,
                        help="Number of seed problems for memory initialization")
    parser.add_argument("--eval_count", type=int, default=500,
                        help="Number of eval problems")
    parser.add_argument("--num_seeds", type=int, default=3,
                        help="Number of random seeds for statistical rigor")
    parser.add_argument("--output_base", type=str, default="results",
                        help="Base directory for results")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.exp == "learning_curve" or args.all:
        run_learning_curve(
            args.dataset_dir,
            seed_count=args.seed_count,
            eval_count=args.eval_count,
            num_seeds=args.num_seeds,
            backend=args.backend,
            output_dir=f"{args.output_base}/exp1_learning_curve",
        )

    if args.exp == "granularity" or args.all:
        run_granularity_ablation(
            args.dataset_dir,
            seed_count=args.seed_count,
            eval_count=min(args.eval_count, 300),
            backend=args.backend,
            output_dir=f"{args.output_base}/exp2_granularity",
        )

    if args.exp == "retrieval" or args.all:
        run_retrieval_quality(
            args.dataset_dir,
            seed_count=args.seed_count,
            eval_count=min(args.eval_count, 300),
            backend=args.backend,
            output_dir=f"{args.output_base}/exp3_retrieval",
        )

    if args.exp == "adaptation" or args.all:
        run_adaptation_ablation(
            args.dataset_dir,
            seed_count=args.seed_count,
            eval_count=min(args.eval_count, 300),
            backend=args.backend,
            output_dir=f"{args.output_base}/exp4_adaptation",
        )

    if args.exp == "breakdown" or args.all:
        run_breakdown_analysis(
            results_dir=f"{args.output_base}/exp1_learning_curve",
            output_dir=f"{args.output_base}/exp5_breakdown",
        )

    if args.exp == "figures" or args.all:
        generate_paper_figures(
            results_base=args.output_base,
            output_dir="figures",
        )

    if not args.exp and not args.all:
        parser.print_help()


if __name__ == "__main__":
    main()
