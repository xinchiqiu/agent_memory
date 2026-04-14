"""Tests for data_structures.py — core dataclasses."""

import numpy as np
from data_structures import Strategy, EpisodicEntry, Memory, Problem, FailureAnnotation


class TestStrategy:
    def test_creation(self):
        s = Strategy(
            technique_chain=["step 1", "step 2"],
            key_insight="Key idea.",
            preconditions=["input is sorted"],
        )
        assert s.technique_chain == ["step 1", "step 2"]
        assert s.key_insight == "Key idea."
        assert s.complexity is None
        assert s.algorithm_tags == []

    def test_with_optional_fields(self):
        s = Strategy(
            technique_chain=["x"],
            key_insight="y",
            preconditions=[],
            complexity="O(n)",
            algorithm_tags=["dp", "greedy"],
        )
        assert s.complexity == "O(n)"
        assert s.algorithm_tags == ["dp", "greedy"]


class TestEpisodicEntry:
    def test_defaults(self):
        s = Strategy(technique_chain=["a"], key_insight="b", preconditions=[])
        e = EpisodicEntry(
            entry_id="e1",
            problem_id="P1",
            problem_statement="stmt",
            strategy=s,
            solution_code="print(1)",
            difficulty_rating=1000,
            verification_passed=True,
        )
        assert e.times_retrieved == 0
        assert e.times_led_to_success == 0
        assert e.times_led_to_failure == 0
        assert e.embedding is None
        assert e.source == "solved"
        assert e.created_at == 0

    def test_with_embedding(self):
        s = Strategy(technique_chain=["a"], key_insight="b", preconditions=[])
        emb = np.zeros(384, dtype=np.float32)
        e = EpisodicEntry(
            entry_id="e1",
            problem_id="P1",
            problem_statement="stmt",
            strategy=s,
            solution_code="print(1)",
            difficulty_rating=1000,
            verification_passed=True,
            embedding=emb,
        )
        assert e.embedding is not None
        assert e.embedding.shape == (384,)


class TestMemory:
    def test_empty(self):
        m = Memory()
        assert len(m.entries) == 0
        assert len(m.failures) == 0
        assert m.current_timestep == 0

    def test_add_entry(self, sample_episodic_entry):
        m = Memory()
        m.entries[sample_episodic_entry.entry_id] = sample_episodic_entry
        assert "e_1" in m.entries
        assert m.entries["e_1"].problem_id == "SEED001"

    def test_add_failure(self, sample_strategy):
        m = Memory()
        fa = FailureAnnotation(
            problem_id="P1",
            problem_statement="stmt",
            retrieved_entry_id="e_1",
            attempted_strategy=sample_strategy,
            failure_code="print('wrong')",
            error_info="wrong answer",
            diagnosis="greedy doesn't work here",
        )
        m.failures.append(fa)
        assert len(m.failures) == 1


class TestProblem:
    def test_creation(self, sample_problem):
        assert sample_problem.problem_id == "TEST001"
        assert sample_problem.contest_id == 1000
        assert sample_problem.difficulty_rating == 800
        assert len(sample_problem.test_cases) == 3
        assert sample_problem.editorial is None

    def test_minimal(self):
        p = Problem(
            problem_id="X",
            contest_id=1,
            index="A",
            title="T",
            statement="S",
            difficulty_rating=800,
            algorithm_tags=[],
            test_cases=[],
        )
        assert p.reference_solutions == []
        assert p.contest_date == ""


class TestFailureAnnotation:
    def test_creation(self, sample_strategy):
        fa = FailureAnnotation(
            problem_id="P1",
            problem_statement="stmt",
            retrieved_entry_id="e_1",
            attempted_strategy=sample_strategy,
            failure_code="code",
            error_info="TLE",
            diagnosis="too slow",
        )
        assert fa.problem_id == "P1"
        assert fa.diagnosis == "too slow"
