"""Shared pytest fixtures for the SAGE test suite."""

import sys
import os
import json
import numpy as np
import pytest

# Ensure the repo root is on sys.path so bare imports like `from config import ...` work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_structures import Strategy, EpisodicEntry, Memory, Problem, FailureAnnotation
from llm_client import BaseLLMClient


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------

class MockLLMClient(BaseLLMClient):
    """Deterministic LLM client that returns canned responses based on prompt keywords."""

    def generate(self, prompt: str, model=None, temperature=0.0,
                 max_tokens=4096, thinking=None) -> str:
        p = prompt.lower()

        if "extract" in p and "strategy" in p:
            return json.dumps({
                "technique_chain": ["read input", "sort the array", "use greedy selection"],
                "key_insight": "Sorting enables greedy selection by processing elements in optimal order.",
                "preconditions": ["input fits in memory", "comparison-based ordering exists"],
                "complexity": "O(n log n) time, O(1) space",
                "algorithm_tags": ["greedy", "sorting"],
            })

        if "adapt" in p or "alignment" in p or "match analysis" in p:
            return (
                "### MATCH ANALYSIS\n"
                "Strategy 1 matches well because the problem requires sorting.\n\n"
                "### KEY DIFFERENCES\n"
                "The new problem also needs a stack.\n\n"
                "### ADAPTATIONS NEEDED\n"
                "Add a stack-based pass after sorting.\n\n"
                "### ADAPTED PLAN\n"
                "1. Read input\n"
                "2. Sort the array\n"
                "3. Use a stack to track elements\n"
                "4. Output the result\n"
            )

        if "solve" in p or "implement" in p or "solution" in p or "code" in p:
            return (
                "```python\n"
                "a, b = map(int, input().split())\n"
                "print(a + b)\n"
                "```"
            )

        if "diagnos" in p:
            return (
                "The greedy strategy failed because the problem requires "
                "dynamic programming to handle overlapping subproblems."
            )

        return "Mock response for unknown prompt type."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def sample_strategy():
    return Strategy(
        technique_chain=["read input", "sort array", "greedy selection"],
        key_insight="Sorting enables greedy processing.",
        preconditions=["elements are comparable"],
        complexity="O(n log n)",
        algorithm_tags=["greedy", "sorting"],
    )


@pytest.fixture
def sample_problem():
    return Problem(
        problem_id="TEST001",
        contest_id=1000,
        index="A",
        title="Sum of Two",
        statement="Read two integers a and b, and print their sum.",
        difficulty_rating=800,
        algorithm_tags=["implementation"],
        test_cases=[
            {"input": "2 3", "expected_output": "5"},
            {"input": "0 0", "expected_output": "0"},
            {"input": "-1 1", "expected_output": "0"},
        ],
        editorial=None,
        reference_solutions=["a,b=map(int,input().split())\nprint(a+b)"],
        contest_date="2020-01-01",
    )


@pytest.fixture
def sample_episodic_entry(sample_strategy):
    rng = np.random.RandomState(42)
    emb = rng.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return EpisodicEntry(
        entry_id="e_1",
        problem_id="SEED001",
        problem_statement="Sort an array and find the median.",
        strategy=sample_strategy,
        solution_code="arr = sorted(map(int, input().split()))\nprint(arr[len(arr)//2])",
        difficulty_rating=1200,
        verification_passed=True,
        embedding=emb,
        created_at=1,
        source="seed",
    )


@pytest.fixture
def sample_memory(sample_strategy):
    """Memory with 3 pre-populated entries, each with a different embedding."""
    rng = np.random.RandomState(42)
    memory = Memory()

    tags_list = [
        ["greedy", "sorting"],
        ["dp"],
        ["graph_dfs", "trees"],
    ]
    statements = [
        "Sort an array and find the median.",
        "Find the longest increasing subsequence.",
        "Find the diameter of a tree.",
    ]

    for i in range(3):
        emb = rng.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        strat = Strategy(
            technique_chain=[f"step_{i}_1", f"step_{i}_2"],
            key_insight=f"Insight for entry {i}",
            preconditions=[f"precondition_{i}"],
            complexity="O(n)",
            algorithm_tags=tags_list[i],
        )
        entry = EpisodicEntry(
            entry_id=f"e_{i}",
            problem_id=f"P{i:03d}",
            problem_statement=statements[i],
            strategy=strat,
            solution_code=f"print({i})",
            difficulty_rating=800 + i * 400,
            verification_passed=True,
            embedding=emb,
            created_at=i,
            source="seed",
        )
        memory.entries[entry.entry_id] = entry

    # Set timestep high enough to avoid entry_id collisions in update tests
    memory.current_timestep = 100
    return memory


@pytest.fixture
def verifier():
    from verifier import Verifier
    return Verifier(timeout_seconds=5, max_tests=10)
