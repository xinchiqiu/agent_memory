"""Module 3 (part 1): Problem encoder using sentence-transformers.

Supports both the off-the-shelf base model and a fine-tuned technique encoder.
Use create_encoder() to get the right model based on config / available files.

Encoding pipeline:
  1. (Optional) LLM generates an "algorithmic fingerprint" — a short summary
     capturing the core technique, data structure, and complexity.
  2. The fingerprint (or compressed statement) is encoded by the sentence
     transformer into a normalized embedding vector.
  3. Cosine similarity between embeddings is used for retrieval.

Using LLM fingerprints avoids embedding collapse: raw CP statements are too
similar at the surface level for small encoders to differentiate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG

# Default path where train_encoder.py saves the fine-tuned model
_DEFAULT_FINETUNED_PATH = "models/technique_encoder_v3"

# Cache directory for LLM-generated fingerprints
_FINGERPRINT_CACHE_DIR = Path("cache/fingerprints")

FINGERPRINT_PROMPT = """\
You are an expert competitive programmer. Given a problem statement, produce a \
short algorithmic fingerprint that captures WHAT technique is needed and WHY.

## Problem Statement
{statement}

## Instructions
Write exactly 2-3 sentences covering:
1. The core algorithmic technique (e.g., "DP on intervals", "BFS on implicit graph", "greedy with sorting")
2. The key structural property that makes this technique work
3. The complexity class (e.g., "O(n log n)", "O(n²)")

Be specific about the algorithm. Do NOT repeat the problem story.
Output ONLY the fingerprint sentences, nothing else.
"""


class ProblemEncoder:
    """Encodes problem statements into normalized embedding vectors.

    Supports two modes:
      - Direct encoding: compress statement → encode (fast, lower quality)
      - LLM fingerprint: LLM summarizes → encode (slower, much better retrieval)

    Use encode_with_fingerprint() when an LLM client is available (seeding,
    eval). Falls back to encode() for lightweight/offline usage.
    """

    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name or CONFIG["encoder_model"]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._fingerprint_cache: Dict[str, str] = {}
        self._load_fingerprint_cache()
        logging.info(f"ProblemEncoder loaded: {model_name}")

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode text (statement or fingerprint) into a normalized vector.

        Returns a unit-normalized float32 vector of shape (embedding_dim,).
        """
        preprocessed = _compress_statement(text)
        vec = self.model.encode(preprocessed, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_with_fingerprint(self, problem_statement: str,
                                llm_client=None) -> np.ndarray:
        """Generate an LLM fingerprint and encode it.

        If an LLM client is provided, generates an algorithmic fingerprint
        (cached on disk). Otherwise falls back to direct encoding.
        """
        if llm_client is None:
            return self.encode(problem_statement)

        fingerprint = self._get_fingerprint(problem_statement, llm_client)
        vec = self.model.encode(fingerprint, normalize_embeddings=True)
        return vec.astype(np.float32)

    def batch_encode(self, statements: List[str],
                     show_progress: bool = False,
                     batch_size: int = 256) -> np.ndarray:
        """Encode multiple statements efficiently (no LLM, uses compression)."""
        preprocessed = [_compress_statement(s) for s in statements]
        vecs = self.model.encode(
            preprocessed,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )
        return vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # Fingerprint generation + caching
    # ------------------------------------------------------------------

    def _get_fingerprint(self, statement: str, llm_client) -> str:
        """Get or generate an algorithmic fingerprint for a problem."""
        cache_key = _statement_hash(statement)

        # Check in-memory cache
        if cache_key in self._fingerprint_cache:
            return self._fingerprint_cache[cache_key]

        # Generate via LLM
        prompt = FINGERPRINT_PROMPT.format(statement=statement[:2000])
        fingerprint = llm_client.generate(
            prompt,
            temperature=0.0,
            max_tokens=200,
            thinking=False,
        ).strip()

        # Validate — should be 1-3 sentences, not code or JSON
        if len(fingerprint) < 10 or fingerprint.startswith("{") or fingerprint.startswith("```"):
            logging.warning(f"Bad fingerprint (len={len(fingerprint)}), falling back to compression")
            fingerprint = _compress_statement(statement)

        # Cache
        self._fingerprint_cache[cache_key] = fingerprint
        self._save_fingerprint(cache_key, fingerprint)
        return fingerprint

    def _load_fingerprint_cache(self):
        """Load cached fingerprints from disk."""
        cache_dir = _FINGERPRINT_CACHE_DIR
        if not cache_dir.exists():
            return
        cache_file = cache_dir / "fingerprints.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._fingerprint_cache = json.load(f)
                logging.info(f"Loaded {len(self._fingerprint_cache)} cached fingerprints")
            except Exception:
                pass

    def _save_fingerprint(self, key: str, fingerprint: str):
        """Append a fingerprint to the disk cache."""
        cache_dir = _FINGERPRINT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "fingerprints.json"
        # Atomic-ish write: load, update, save
        try:
            existing = {}
            if cache_file.exists():
                with open(cache_file) as f:
                    existing = json.load(f)
            existing[key] = fingerprint
            with open(cache_file, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save fingerprint cache: {e}")


# ---------------------------------------------------------------------------
# Statement compression — fit into 256-token encoder window
# ---------------------------------------------------------------------------

def _compress_statement(statement: str) -> str:
    """Compress a problem statement to its algorithmic core.

    The encoder has a 256-token (~600 char) limit. Raw statements are
    400-700 tokens. We need to drop boilerplate and keep what distinguishes
    problems algorithmically.

    Strategy: aggressively remove everything except the task description
    and constraints. Drop I/O format, examples, notes, and narrative.
    """
    if not statement:
        return ""

    text = statement

    # Remove everything after "Example", "Sample", "Note" sections
    text = re.sub(r"(?i)\n\s*(example|sample|note)\s*\n.*", "", text, flags=re.DOTALL)

    # Split into lines
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    kept = []
    for line in lines:
        ll = line.lower()

        # Drop section headers
        if re.match(r"^(input|output|examples?|note|constraints?)\s*$", ll):
            continue

        # Drop I/O format lines
        if re.match(r"^(the )?(first|second|third|next|each|single|only|last) line", ll):
            continue
        if re.match(r"^(print|output) ", ll) and re.search(r"(single|one|each|the) (line|number|integer)", ll):
            continue
        if re.match(r"^it is guaranteed", ll):
            # Keep constraint guarantees
            kept.append(line)
            continue

        kept.append(line)

    result = " ".join(kept)

    # Extract constraints and prepend them (highest signal for algorithmic structure)
    constraints = re.findall(
        r"(\d\s*[≤<=]\s*\w[^,.\n)]{0,60})", text
    )
    if constraints:
        constraint_str = "; ".join(constraints[:4])
        result = "Constraints: " + constraint_str + " | " + result

    # Hard limit
    if len(result) > 600:
        result = result[:600]

    return result


def _statement_hash(statement: str) -> str:
    """Stable hash for caching fingerprints."""
    return hashlib.md5(statement.encode("utf-8")).hexdigest()[:16]


def create_encoder(prefer_finetuned: bool = True,
                   finetuned_path: Optional[str] = None) -> ProblemEncoder:
    """Factory that returns the best available encoder.

    Priority:
      1. Fine-tuned technique encoder (if available at finetuned_path)
      2. Base sentence-transformer from config

    Args:
        prefer_finetuned: If True, use the fine-tuned model when available.
        finetuned_path:   Path to the fine-tuned model directory.
                          Defaults to "models/technique_encoder".

    Returns:
        A ProblemEncoder instance.

    Example
    -------
    # After running encoder_training/train_encoder.py:
    encoder = create_encoder()   # automatically picks fine-tuned model

    # Force base model (e.g., before fine-tuning):
    encoder = create_encoder(prefer_finetuned=False)
    """
    if prefer_finetuned:
        path = finetuned_path or _DEFAULT_FINETUNED_PATH
        if Path(path).exists():
            logging.info(f"Using fine-tuned encoder: {path}")
            return ProblemEncoder(model_name=path)
        else:
            logging.info(
                f"Fine-tuned encoder not found at '{path}'. "
                f"Using base model '{CONFIG['encoder_model']}'. "
                f"Run encoder_training/train_encoder.py to train it."
            )
    return ProblemEncoder()
