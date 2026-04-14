"""Module 3 (part 2): Retriever — similarity-based memory lookup."""

import numpy as np
from typing import List, Tuple, Optional
import random

from data_structures import Problem, Memory, EpisodicEntry
from encoder import ProblemEncoder
from config import CONFIG


class Retriever:
    """Retrieves the most relevant verified entries from memory for a new problem."""

    def __init__(self, encoder: ProblemEncoder, top_k: int = None,
                 llm_client=None):
        self.encoder = encoder
        self.top_k = top_k or CONFIG["top_k"]
        self.llm_client = llm_client

    def retrieve(self,
                 new_problem: Problem,
                 memory: Memory) -> List[Tuple[EpisodicEntry, float]]:
        """Find the top-k most relevant entries for new_problem.

        Only considers entries where verification_passed is True.
        Uses LLM fingerprints when llm_client is available.

        Returns:
            List of (entry, cosine_similarity) sorted descending.
        """
        candidates = self._get_verified_candidates(memory)
        if not candidates:
            return []

        query_vec = self.encoder.encode_with_fingerprint(
            new_problem.statement, self.llm_client
        )
        scored = []
        for entry in candidates:
            sim = float(np.dot(query_vec, entry.embedding))
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]

    def _get_verified_candidates(self, memory: Memory) -> List[EpisodicEntry]:
        return [
            e for e in memory.entries.values()
            if e.verification_passed and e.embedding is not None
        ]


class RandomRetriever(Retriever):
    """Baseline: randomly sample k entries instead of similarity search."""

    def retrieve(self,
                 new_problem: Problem,
                 memory: Memory) -> List[Tuple[EpisodicEntry, float]]:
        candidates = self._get_verified_candidates(memory)
        if not candidates:
            return []
        sampled = random.sample(candidates, min(self.top_k, len(candidates)))
        # Assign uniform score of 0.5 to indicate "random"
        return [(e, 0.5) for e in sampled]


class TagOracleRetriever(Retriever):
    """Oracle baseline: retrieve entries that share ground-truth algorithm tags.

    Uses information the real agent does not have. Represents the ceiling
    of what perfect retrieval could achieve.
    """

    def retrieve(self,
                 new_problem: Problem,
                 memory: Memory) -> List[Tuple[EpisodicEntry, float]]:
        candidates = self._get_verified_candidates(memory)
        if not candidates:
            return []

        query_tags = set(new_problem.algorithm_tags)
        scored = []
        for entry in candidates:
            entry_tags = set(entry.strategy.algorithm_tags)
            overlap = len(query_tags & entry_tags) / max(len(query_tags | entry_tags), 1)
            scored.append((entry, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]
