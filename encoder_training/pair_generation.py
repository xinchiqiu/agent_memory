"""Training pair generation for contrastive encoder fine-tuning.

Generates three kinds of training data:
  1. Random positive pairs  — share at least one algorithm tag
  2. Hard positive pairs    — share the primary (first) tag; harder to confuse
  3. Hard negative pairs    — embedding-close but technique-different
                              (mined from a base encoder)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tag index helpers
# ---------------------------------------------------------------------------

def build_tag_index(problems: List[dict]) -> Dict[str, List[str]]:
    """Build an inverted index: canonical_tag -> [problem_id, ...]."""
    idx: Dict[str, List[str]] = defaultdict(list)
    for p in problems:
        for tag in p.get("tags", []):
            idx[tag].append(p["problem_id"])
    return idx


# ---------------------------------------------------------------------------
# Positive / negative pair generation
# ---------------------------------------------------------------------------

def generate_training_pairs(
    problems: List[dict],
    num_pairs: int = 50_000,
    hard_positive_ratio: float = 0.5,
    seed: int = 42,
) -> List[dict]:
    """Generate (anchor, positive, negative) training triplets.

    Args:
        problems: List of problem dicts with "problem_id", "tags", "statement".
        num_pairs: Target number of triplets.
        hard_positive_ratio: Fraction of positives that share the PRIMARY tag
                             (vs. any shared tag).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts:
            anchor, positive, negative: str (problem statements)
            anchor_id, positive_id, negative_id: str
            anchor_tags, positive_tags, negative_tags: List[str]
            pair_type: "hard_positive" | "soft_positive"
    """
    random.seed(seed)

    pid_to_problem = {p["problem_id"]: p for p in problems}
    tag_index = build_tag_index(problems)
    problem_list = [p for p in problems if p.get("tags") and p.get("statement")]

    if not problem_list:
        raise ValueError("No problems with both tags and statement")

    pairs = []
    max_attempts = num_pairs * 10
    attempts = 0

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        anchor = random.choice(problem_list)
        anchor_tags = anchor.get("tags", [])
        if not anchor_tags:
            continue

        # --- Select positive ---
        use_hard_positive = random.random() < hard_positive_ratio
        if use_hard_positive:
            # Share the primary tag (first tag in the list)
            primary = anchor_tags[0]
            candidates = [
                pid for pid in tag_index.get(primary, [])
                if pid != anchor["problem_id"]
            ]
        else:
            # Share ANY tag
            shared_tag = random.choice(anchor_tags)
            candidates = [
                pid for pid in tag_index.get(shared_tag, [])
                if pid != anchor["problem_id"]
            ]

        if not candidates:
            continue
        positive = pid_to_problem[random.choice(candidates)]

        # --- Select random negative (no shared tags) ---
        anchor_tag_set = set(anchor_tags)
        negative = None
        for _ in range(30):
            cand = random.choice(problem_list)
            if cand["problem_id"] == anchor["problem_id"]:
                continue
            if not set(cand.get("tags", [])) & anchor_tag_set:
                negative = cand
                break
        if negative is None:
            continue

        pairs.append({
            "anchor":        anchor["statement"],
            "positive":      positive["statement"],
            "negative":      negative["statement"],
            "anchor_id":     anchor["problem_id"],
            "positive_id":   positive["problem_id"],
            "negative_id":   negative["problem_id"],
            "anchor_tags":   anchor_tags,
            "positive_tags": positive.get("tags", []),
            "negative_tags": negative.get("tags", []),
            "pair_type":     "hard_positive" if use_hard_positive else "soft_positive",
        })

    logging.info(
        f"Generated {len(pairs)} pairs from {attempts} attempts "
        f"({sum(1 for p in pairs if p['pair_type']=='hard_positive')} hard positives)"
    )
    return pairs


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    problems: List[dict],
    base_encoder,
    k_candidates: int = 30,
    max_pairs: int = 10_000,
    batch_size: int = 256,
    seed: int = 42,
) -> List[dict]:
    """Find (anchor, hard_negative) pairs using a base encoder.

    Hard negatives are problems that look similar in embedding space
    but require different algorithmic techniques.

    Args:
        problems: List of problem dicts (need "tags" and "statement").
        base_encoder: Any object with .encode(List[str]) -> np.ndarray.
        k_candidates: Number of nearest neighbors to examine per anchor.
        max_pairs: Maximum hard negative pairs to return.
        batch_size: Encoding batch size.
        seed: Random seed.

    Returns:
        List of dicts: anchor, hard_negative, similarity, anchor_tags, negative_tags,
                       anchor_id, negative_id.
    """
    random.seed(seed)
    filtered = [p for p in problems if p.get("tags") and p.get("statement")]
    if not filtered:
        return []

    statements = [p["statement"] for p in filtered]
    logging.info(f"Encoding {len(statements)} problems for hard negative mining…")

    # Encode in batches to avoid OOM
    all_embs = []
    for start in range(0, len(statements), batch_size):
        batch = statements[start: start + batch_size]
        embs = base_encoder.encode(batch)
        if hasattr(embs, "numpy"):
            embs = embs.numpy()
        all_embs.append(np.array(embs, dtype=np.float32))
    embeddings = np.concatenate(all_embs, axis=0)

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    hard_pairs = []
    indices = list(range(len(filtered)))
    random.shuffle(indices)

    for i in indices:
        if len(hard_pairs) >= max_pairs:
            break

        p = filtered[i]
        p_tags = set(p.get("tags", []))
        if not p_tags:
            continue

        # Cosine similarities against all other problems
        sims = embeddings @ embeddings[i]
        sims[i] = -1.0  # exclude self

        top_k = np.argsort(sims)[::-1][:k_candidates]

        for j in top_k:
            neg = filtered[j]
            neg_tags = set(neg.get("tags", []))
            if not neg_tags & p_tags:  # no shared tags
                hard_pairs.append({
                    "anchor":        p["statement"],
                    "hard_negative": neg["statement"],
                    "similarity":    float(sims[j]),
                    "anchor_id":     p["problem_id"],
                    "negative_id":   neg["problem_id"],
                    "anchor_tags":   list(p_tags),
                    "negative_tags": list(neg_tags),
                })
                break  # one hard negative per anchor

    logging.info(f"Mined {len(hard_pairs)} hard negatives")
    return hard_pairs


# ---------------------------------------------------------------------------
# Merge pairs into sentence-transformers InputExample format
# ---------------------------------------------------------------------------

def to_triplet_examples(
    pairs: List[dict],
    hard_neg_pairs: Optional[List[dict]] = None,
):
    """Convert pair dicts to sentence_transformers InputExample triplets.

    Each example has texts=[anchor, positive, negative].
    For hard-negative pairs, we sample a random positive from normal pairs.

    Returns a list of InputExample objects ready for TripletLoss.
    """
    from sentence_transformers import InputExample

    examples = []

    # Normal (anchor, positive, negative) triplets
    for p in pairs:
        examples.append(InputExample(
            texts=[p["anchor"], p["positive"], p["negative"]]
        ))

    # Hard negatives: pair each with a random positive from normal pairs
    if hard_neg_pairs:
        pos_by_tag: Dict[str, List[str]] = defaultdict(list)
        for p in pairs:
            for t in p["anchor_tags"]:
                pos_by_tag[t].append(p["positive"])

        for hn in hard_neg_pairs:
            anchor_tags = hn["anchor_tags"]
            # Find a positive that shares any anchor tag
            positives = []
            for t in anchor_tags:
                positives.extend(pos_by_tag.get(t, []))
            if not positives:
                continue
            positive_text = random.choice(positives)
            examples.append(InputExample(
                texts=[hn["anchor"], positive_text, hn["hard_negative"]]
            ))

    logging.info(f"Created {len(examples)} InputExample triplets")
    return examples


def to_mnrl_examples(pairs: List[dict]):
    """Convert pairs to (anchor, positive) InputExamples for MultipleNegativesRankingLoss."""
    from sentence_transformers import InputExample

    return [
        InputExample(texts=[p["anchor"], p["positive"]])
        for p in pairs
    ]
