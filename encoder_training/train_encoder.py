"""Contrastive fine-tuning of the retrieval encoder.

Trains a sentence transformer to cluster problems by algorithmic technique
rather than surface-level semantic similarity.

Usage
-----
# Full training pipeline (data must be collected first):
python encoder_training/train_encoder.py --dataset_dir dataset/ --output_dir models/technique_encoder

# Quick smoke test (small subset):
python encoder_training/train_encoder.py --dataset_dir dataset/ --output_dir models/test --num_pairs 1000 --epochs 1

Training loss options
---------------------
mnrl      MultipleNegativesRankingLoss — fast, in-batch negatives, recommended default
triplet   TripletLoss — uses explicit (a, p, n) triplets; slower but uses hard negatives
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.dataset_utils import load_all_problems, load_split
from encoder_training.pair_generation import (
    generate_training_pairs,
    mine_hard_negatives,
    to_mnrl_examples,
    to_triplet_examples,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune retrieval encoder")
    p.add_argument("--dataset_dir",  default="dataset/",          help="Root dataset directory")
    p.add_argument("--output_dir",   default="models/technique_encoder")
    p.add_argument("--base_model",   default="all-MiniLM-L6-v2",  help="HF model to fine-tune")
    p.add_argument("--loss",         default="mnrl",               choices=["mnrl", "triplet"])
    p.add_argument("--num_pairs",    type=int, default=50_000,     help="Training pairs to generate")
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--hard_neg_ratio", type=float, default=0.3,
                   help="Fraction of training data from hard-negative mining")
    p.add_argument("--hard_pos_ratio", type=float, default=0.5,
                   help="Fraction of positives that share primary tag")
    p.add_argument("--no_hard_negatives", action="store_true",
                   help="Skip hard negative mining (faster but lower quality)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--eval_fraction", type=float, default=0.1,
                   help="Fraction of eval problems to use for mid-training eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logging.info("Loading problems…")
    all_problems = load_all_problems(args.dataset_dir)
    seed_ids  = set(load_split("seed",  args.dataset_dir))
    eval_ids  = set(load_split("eval",  args.dataset_dir))

    train_problems = [p for p in all_problems if p["problem_id"] in seed_ids
                      and p.get("tags") and p.get("statement")]
    eval_problems  = [p for p in all_problems if p["problem_id"] in eval_ids
                      and p.get("tags") and p.get("statement")]

    logging.info(f"Train: {len(train_problems)}  Eval: {len(eval_problems)}")

    if len(train_problems) < 50:
        logging.error("Too few training problems — collect more data first")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Import sentence-transformers (lazy to avoid slow import on top level)
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer, losses, evaluation
    from torch.utils.data import DataLoader

    # ------------------------------------------------------------------
    # Generate training data
    # ------------------------------------------------------------------
    logging.info("Generating training pairs…")
    pairs = generate_training_pairs(
        train_problems,
        num_pairs=args.num_pairs,
        hard_positive_ratio=args.hard_pos_ratio,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    logging.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    # ------------------------------------------------------------------
    # Hard negative mining (uses the BASE model before any training)
    # ------------------------------------------------------------------
    hard_neg_pairs = []
    if not args.no_hard_negatives:
        num_hard = int(len(pairs) * args.hard_neg_ratio)
        logging.info(f"Mining {num_hard} hard negatives…")
        hard_neg_pairs = mine_hard_negatives(
            train_problems,
            base_encoder=model,
            max_pairs=num_hard,
            seed=args.seed,
        )

    # ------------------------------------------------------------------
    # Build training examples and DataLoader
    # ------------------------------------------------------------------
    logging.info(f"Building training examples (loss={args.loss})…")
    if args.loss == "triplet":
        train_examples = to_triplet_examples(pairs, hard_neg_pairs or None)
        train_loss = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.5,
        )
    else:  # mnrl
        # MNRL only needs (anchor, positive) — in-batch negatives are used automatically
        mnrl_pairs = pairs + [
            {
                "anchor":   hn["anchor"],
                "positive": _find_any_positive(hn["anchor_tags"], pairs),
            }
            for hn in hard_neg_pairs
            if _find_any_positive(hn["anchor_tags"], pairs) is not None
        ]
        train_examples = to_mnrl_examples(mnrl_pairs)
        train_loss = losses.MultipleNegativesRankingLoss(model)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Evaluator (runs on a subset of eval problems between epochs)
    # ------------------------------------------------------------------
    evaluator = None
    if eval_problems:
        import random
        random.seed(args.seed)
        n_eval = max(50, int(len(eval_problems) * args.eval_fraction))
        eval_subset = random.sample(eval_problems, min(n_eval, len(eval_problems)))
        eval_pairs = generate_training_pairs(eval_subset, num_pairs=500, seed=args.seed + 1)
        if eval_pairs:
            anchors   = [p["anchor"]   for p in eval_pairs]
            positives = [p["positive"] for p in eval_pairs]
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                anchors, positives,
                scores=[1.0] * len(eval_pairs),  # all are positive pairs
                name="technique_similarity",
            )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logging.info(
        f"Training: {total_steps} total steps, {warmup_steps} warmup, "
        f"lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Save training metadata
    # ------------------------------------------------------------------
    meta = {
        "base_model":       args.base_model,
        "loss":             args.loss,
        "num_train_pairs":  len(train_examples),
        "num_hard_negatives": len(hard_neg_pairs),
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "train_problems":   len(train_problems),
        "eval_problems":    len(eval_problems),
    }
    with open(Path(args.output_dir) / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Model saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _find_any_positive(anchor_tags: list, pairs: list) -> str | None:
    """Find a positive statement for an anchor tag set from the pair pool."""
    import random
    candidates = [
        p["positive"] for p in pairs
        if set(p["anchor_tags"]) & set(anchor_tags)
    ]
    return random.choice(candidates) if candidates else None


if __name__ == "__main__":
    main()
