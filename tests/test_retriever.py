"""Tests for retriever.py — similarity-based memory retrieval."""

import numpy as np
import pytest
from data_structures import Strategy, EpisodicEntry, Memory, Problem
from retriever import Retriever, RandomRetriever, TagOracleRetriever


class MockEncoder:
    """Minimal encoder mock that returns a fixed or keyword-based vector."""

    def encode(self, statement: str) -> np.ndarray:
        rng = np.random.RandomState(hash(statement) % 2**31)
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


@pytest.fixture
def encoder():
    return MockEncoder()


@pytest.fixture
def query_problem():
    return Problem(
        problem_id="Q1",
        contest_id=999,
        index="A",
        title="Query",
        statement="Sort an array and find the median.",  # Similar to entry e_0
        difficulty_rating=1000,
        algorithm_tags=["sorting"],
        test_cases=[],
    )


class TestRetriever:
    def test_retrieve_returns_top_k(self, encoder, sample_memory, query_problem):
        r = Retriever(encoder, top_k=2)
        results = r.retrieve(query_problem, sample_memory)
        assert len(results) == 2

    def test_retrieve_empty_memory(self, encoder, query_problem):
        r = Retriever(encoder, top_k=3)
        results = r.retrieve(query_problem, Memory())
        assert results == []

    def test_retrieve_filters_unverified(self, encoder, sample_memory, query_problem):
        # Mark one entry as unverified
        sample_memory.entries["e_0"].verification_passed = False
        r = Retriever(encoder, top_k=3)
        results = r.retrieve(query_problem, sample_memory)
        entry_ids = [e.entry_id for e, _ in results]
        assert "e_0" not in entry_ids

    def test_retrieve_filters_no_embedding(self, encoder, sample_memory, query_problem):
        sample_memory.entries["e_1"].embedding = None
        r = Retriever(encoder, top_k=3)
        results = r.retrieve(query_problem, sample_memory)
        entry_ids = [e.entry_id for e, _ in results]
        assert "e_1" not in entry_ids

    def test_retrieve_ordering(self, encoder, sample_memory, query_problem):
        r = Retriever(encoder, top_k=3)
        results = r.retrieve(query_problem, sample_memory)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_fewer_than_k(self, encoder, query_problem):
        """When memory has fewer entries than top_k, return all available."""
        memory = Memory()
        strat = Strategy(technique_chain=["a"], key_insight="b", preconditions=[])
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        entry = EpisodicEntry(
            entry_id="only",
            problem_id="P0",
            problem_statement="solo",
            strategy=strat,
            solution_code="pass",
            difficulty_rating=800,
            verification_passed=True,
            embedding=emb,
        )
        memory.entries["only"] = entry
        r = Retriever(encoder, top_k=5)
        results = r.retrieve(query_problem, memory)
        assert len(results) == 1


class TestRandomRetriever:
    def test_returns_k_entries(self, encoder, sample_memory, query_problem):
        r = RandomRetriever(encoder, top_k=2)
        results = r.retrieve(query_problem, sample_memory)
        assert len(results) == 2

    def test_uniform_score(self, encoder, sample_memory, query_problem):
        r = RandomRetriever(encoder, top_k=3)
        results = r.retrieve(query_problem, sample_memory)
        for _, score in results:
            assert score == 0.5

    def test_empty_memory(self, encoder, query_problem):
        r = RandomRetriever(encoder, top_k=3)
        results = r.retrieve(query_problem, Memory())
        assert results == []


class TestTagOracleRetriever:
    def test_prefers_matching_tags(self, encoder, sample_memory, query_problem):
        """Query has ["sorting"] tag; entry e_0 has ["greedy", "sorting"]."""
        r = TagOracleRetriever(encoder, top_k=3)
        results = r.retrieve(query_problem, sample_memory)
        # e_0 should score highest because it shares "sorting" tag
        assert results[0][0].entry_id == "e_0"
        assert results[0][1] > 0

    def test_empty_memory(self, encoder, query_problem):
        r = TagOracleRetriever(encoder, top_k=3)
        results = r.retrieve(query_problem, Memory())
        assert results == []
