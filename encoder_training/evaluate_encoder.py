"""Encoder evaluation: Precision@k for technique-aware retrieval.

Compares:
  1. Base sentence transformer (all-MiniLM-L6-v2 off-the-shelf)
  2. Fine-tuned technique encoder
  3. Random retrieval lower bound

Usage
-----
python encoder_training/evaluate_encoder.py \
    --dataset_dir dataset/ \
    --base_model all-MiniLM-L6-v2 \
    --finetuned_model models/technique_encoder

Output
------
A table like:

  Encoder                    P@1     P@3     P@5     P@10
  Random baseline           0.143   0.143   0.143   0.143
  Base (all-MiniLM-L6-v2)  0.412   0.371   0.349   0.318
  Fine-tuned                0.631   0.587   0.561   0.523
  Improvement               +0.219  +0.216  +0.212  +0.205
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.dataset_utils import load_all_problems, load_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval encoder quality")
    p.add_argument("--dataset_dir",      default="dataset/")
    p.add_argument("--base_model",       default="all-MiniLM-L6-v2")
    p.add_argument("--finetuned_model",  default="models/technique_encoder",
                   help="Path to fine-tuned model, or HF identifier")
    p.add_argument("--eval_split",       default="eval",
                   help="Which split to use as queries (default: eval)")
    p.add_argument("--k_values",         default="1,3,5,10")
    p.add_argument("--batch_size",       type=int, default=256)
    p.add_argument("--output_json",      default=None,
                   help="Optional path to save results as JSON")
    p.add_argument("--max_eval",         type=int, default=None,
                   help="Limit number of eval queries (for fast testing)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_encoder(
    model,
    all_problems: List[dict],
    eval_problems: List[dict],
    k_values: List[int] = (1, 3, 5, 10),
    batch_size: int = 256,
) -> Dict[str, float]:
    """Compute Precision@k for technique-aware retrieval.

    For each eval problem:
      1. Find its k nearest neighbours among all_problems.
      2. Count how many share at least one algorithm tag with the query.

    Returns:
        Dict "precision@k" -> mean precision over all eval queries.
    """
    # Encode all problems
    all_stmts = [p["statement"] for p in all_problems]
    logging.info(f"Encoding {len(all_stmts)} problems…")

    all_embs_list = []
    for start in range(0, len(all_stmts), batch_size):
        batch = all_stmts[start: start + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs_list.append(np.array(embs, dtype=np.float32))
    all_embs = np.concatenate(all_embs_list, axis=0)  # (N, D)

    pid_to_idx  = {p["problem_id"]: i for i, p in enumerate(all_problems)}
    pid_to_tags = {p["problem_id"]: set(p.get("tags", [])) for p in all_problems}

    results: Dict[str, List[float]] = {f"precision@{k}": [] for k in k_values}

    for eval_p in eval_problems:
        eid = eval_p["problem_id"]
        if eid not in pid_to_idx:
            continue
        eval_tags = pid_to_tags.get(eid, set())
        if not eval_tags:
            continue

        ei = pid_to_idx[eid]
        sims = all_embs @ all_embs[ei]   # dot product (normalised = cosine sim)
        sims[ei] = -2.0                  # exclude self

        sorted_indices = np.argsort(sims)[::-1]

        for k in k_values:
            top_k = sorted_indices[:k]
            matches = sum(
                1 for j in top_k
                if pid_to_tags.get(all_problems[j]["problem_id"], set()) & eval_tags
            )
            results[f"precision@{k}"].append(matches / k)

    return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in results.items()}


def random_baseline_precision(
    all_problems: List[dict],
    eval_problems: List[dict],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute expected Precision@k for random retrieval.

    E[P@k] ≈ probability that a randomly chosen problem shares a tag
             with the query = (mean problems sharing any tag) / N
    """
    pid_to_tags = {p["problem_id"]: set(p.get("tags", [])) for p in all_problems}
    n = len(all_problems)
    precisions = []

    for eval_p in eval_problems:
        eval_tags = pid_to_tags.get(eval_p["problem_id"], set())
        if not eval_tags:
            continue
        share = sum(1 for p in all_problems
                    if p["problem_id"] != eval_p["problem_id"]
                    and pid_to_tags.get(p["problem_id"], set()) & eval_tags)
        precisions.append(share / max(n - 1, 1))

    mean_p = float(np.mean(precisions)) if precisions else 0.0
    return {f"precision@{k}": mean_p for k in k_values}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    k_values = [int(k) for k in args.k_values.split(",")]

    # Load problems
    all_problems = [
        p for p in load_all_problems(args.dataset_dir)
        if p.get("tags") and p.get("statement")
    ]
    eval_ids = set(load_split(args.eval_split, args.dataset_dir))
    eval_problems = [p for p in all_problems if p["problem_id"] in eval_ids]

    if args.max_eval:
        import random; random.seed(42)
        eval_problems = random.sample(eval_problems, min(args.max_eval, len(eval_problems)))

    logging.info(f"All problems: {len(all_problems)}  Eval queries: {len(eval_problems)}")

    from sentence_transformers import SentenceTransformer

    encoders_to_eval = {}
    encoders_to_eval["Base"] = SentenceTransformer(args.base_model)
    if Path(args.finetuned_model).exists() or "/" not in args.finetuned_model:
        encoders_to_eval["Fine-tuned"] = SentenceTransformer(args.finetuned_model)
    else:
        logging.warning(f"Fine-tuned model not found at {args.finetuned_model}, skipping")

    # Random baseline
    rand_results = random_baseline_precision(all_problems, eval_problems, k_values)

    all_results: Dict[str, dict] = {"Random baseline": rand_results}
    for name, model in encoders_to_eval.items():
        logging.info(f"Evaluating {name}…")
        res = evaluate_encoder(model, all_problems, eval_problems, k_values, args.batch_size)
        all_results[name] = res

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    k_headers = "  ".join(f"P@{k:>2}" for k in k_values)
    print(f"\n{'Encoder':<32}  {k_headers}")
    print("-" * (34 + 8 * len(k_values)))
    for name, res in all_results.items():
        vals = "  ".join(f"{res[f'precision@{k}']:.3f}" for k in k_values)
        print(f"{name:<32}  {vals}")

    # Improvement row
    if "Base" in all_results and "Fine-tuned" in all_results:
        base = all_results["Base"]
        ft   = all_results["Fine-tuned"]
        diffs = "  ".join(
            f"{ft[f'precision@{k}'] - base[f'precision@{k}']:+.3f}" for k in k_values
        )
        print(f"{'Improvement':<32}  {diffs}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
