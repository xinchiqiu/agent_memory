"""Module 3 (part 1): Problem encoder using sentence-transformers.

Supports both the off-the-shelf base model and a fine-tuned technique encoder.
Use create_encoder() to get the right model based on config / available files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG

# Default path where train_encoder.py saves the fine-tuned model
_DEFAULT_FINETUNED_PATH = "models/technique_encoder"


class ProblemEncoder:
    """Encodes problem statements into normalized embedding vectors."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model name or local path.
                        Falls back to CONFIG["encoder_model"] if None.
        """
        model_name = model_name or CONFIG["encoder_model"]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logging.info(f"ProblemEncoder loaded: {model_name}")

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, problem_statement: str) -> np.ndarray:
        """Encode a single problem statement.

        Returns a unit-normalized float32 vector of shape (embedding_dim,).
        """
        preprocessed = self._preprocess(problem_statement)
        vec = self.model.encode(preprocessed, normalize_embeddings=True)
        return vec.astype(np.float32)

    def batch_encode(self, statements: List[str],
                     show_progress: bool = False,
                     batch_size: int = 256) -> np.ndarray:
        """Encode multiple problem statements efficiently.

        Returns array of shape (n, embedding_dim), unit-normalized.
        """
        preprocessed = [self._preprocess(s) for s in statements]
        vecs = self.model.encode(
            preprocessed,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )
        return vecs.astype(np.float32)

    def _preprocess(self, statement: str) -> str:
        """Preprocess a problem statement before encoding.

        V1: return raw text.
        V2 option: prepend an LLM-generated structural summary to help the
        encoder focus on algorithmic content rather than narrative flavor text.
        """
        return statement


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
