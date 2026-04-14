"""Tests for memory_update.py — memory state management."""

import numpy as np
import pytest
from data_structures import Memory, EpisodicEntry, Strategy, Problem
from memory_update import update_memory, _handle_success, _handle_failure


class MockEncoder:
    def encode(self, statement: str) -> np.ndarray:
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


@pytest.fixture
def encoder():
    return MockEncoder()


@pytest.fixture
def success_result():
    return {
        "success": True,
        "code": "a,b=map(int,input().split())\nprint(a+b)",
        "method": "adapted_1",
        "alignment_analysis": "analysis",
        "adapted_plan": "plan",
        "attempts": [
            {
                "method": "adapted_1",
                "retrieved_entry_id": "e_0",
                "similarity_score": 0.9,
                "code": "a,b=map(int,input().split())\nprint(a+b)",
                "passed": True,
                "error_info": "",
            }
        ],
    }


@pytest.fixture
def failure_result():
    return {
        "success": False,
        "code": "print('wrong')",
        "method": "failed",
        "alignment_analysis": "analysis",
        "adapted_plan": "plan",
        "attempts": [
            {
                "method": "adapted_1",
                "retrieved_entry_id": "e_0",
                "similarity_score": 0.9,
                "code": "print('wrong')",
                "passed": False,
                "error_info": "wrong answer",
            },
            {
                "method": "free",
                "retrieved_entry_id": None,
                "similarity_score": None,
                "code": "print('also wrong')",
                "passed": False,
                "error_info": "wrong answer",
            },
        ],
    }


class TestUpdateMemorySuccess:
    def test_adds_entry(self, mock_llm_client, encoder, sample_memory, sample_problem, success_result):
        initial_count = len(sample_memory.entries)
        update_memory(sample_memory, sample_problem, success_result, encoder, mock_llm_client)
        assert len(sample_memory.entries) == initial_count + 1

    def test_increments_timestep(self, mock_llm_client, encoder, sample_memory, sample_problem, success_result):
        old_ts = sample_memory.current_timestep
        update_memory(sample_memory, sample_problem, success_result, encoder, mock_llm_client)
        assert sample_memory.current_timestep == old_ts + 1

    def test_updates_usage_stats(self, mock_llm_client, encoder, sample_memory, sample_problem, success_result):
        old_retrieved = sample_memory.entries["e_0"].times_retrieved
        old_success = sample_memory.entries["e_0"].times_led_to_success
        update_memory(sample_memory, sample_problem, success_result, encoder, mock_llm_client)
        assert sample_memory.entries["e_0"].times_retrieved == old_retrieved + 1
        assert sample_memory.entries["e_0"].times_led_to_success == old_success + 1

    def test_new_entry_has_embedding(self, mock_llm_client, encoder, sample_memory, sample_problem, success_result):
        update_memory(sample_memory, sample_problem, success_result, encoder, mock_llm_client)
        # Find the new entry (highest timestep)
        new_entries = [e for e in sample_memory.entries.values() if e.source == "solved"]
        assert len(new_entries) == 1
        assert new_entries[0].embedding is not None


class TestUpdateMemoryFailure:
    def test_adds_failure_annotation(self, mock_llm_client, encoder, sample_memory, sample_problem, failure_result):
        initial_failures = len(sample_memory.failures)
        update_memory(sample_memory, sample_problem, failure_result, encoder, mock_llm_client)
        assert len(sample_memory.failures) == initial_failures + 1

    def test_does_not_add_entry(self, mock_llm_client, encoder, sample_memory, sample_problem, failure_result):
        initial_count = len(sample_memory.entries)
        update_memory(sample_memory, sample_problem, failure_result, encoder, mock_llm_client)
        assert len(sample_memory.entries) == initial_count

    def test_updates_failure_stats(self, mock_llm_client, encoder, sample_memory, sample_problem, failure_result):
        old_failure = sample_memory.entries["e_0"].times_led_to_failure
        update_memory(sample_memory, sample_problem, failure_result, encoder, mock_llm_client)
        assert sample_memory.entries["e_0"].times_led_to_failure == old_failure + 1

    def test_skips_free_attempts(self, mock_llm_client, encoder, sample_memory, sample_problem, failure_result):
        """Free generation attempts should not create FailureAnnotations."""
        update_memory(sample_memory, sample_problem, failure_result, encoder, mock_llm_client)
        # Only 1 annotation for adapted_1, not for free
        annotations_for_problem = [
            fa for fa in sample_memory.failures
            if fa.problem_id == sample_problem.problem_id
        ]
        assert len(annotations_for_problem) == 1
