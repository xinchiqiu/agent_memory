"""Module 3 (part 1): Problem encoder using sentence-transformers."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG


class ProblemEncoder:
    """Encodes problem statements into normalized embedding vectors."""

    def __init__(self, model_name: str = None):
        model_name = model_name or CONFIG["encoder_model"]
        self.model = SentenceTransformer(model_name)

    def encode(self, problem_statement: str) -> np.ndarray:
        """Encode a single problem statement.

        Returns a unit-normalized float32 vector.
        """
        preprocessed = self._preprocess(problem_statement)
        vec = self.model.encode(preprocessed, normalize_embeddings=True)
        return vec.astype(np.float32)

    def batch_encode(self, statements: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode multiple problem statements.

        Returns array of shape (n, embedding_dim), unit-normalized.
        """
        preprocessed = [self._preprocess(s) for s in statements]
        vecs = self.model.encode(
            preprocessed,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def _preprocess(self, statement: str) -> str:
        """Preprocess a problem statement before encoding.

        V1: return raw text.
        V2 option: prepend an LLM-generated structural summary to help the
        encoder focus on algorithmic content rather than narrative flavor text.
        """
        return statement
