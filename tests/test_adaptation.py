"""Tests for adaptation.py — alignment, code generation, fallback chain."""

import pytest
from data_structures import Problem, EpisodicEntry, Strategy
from adaptation import (
    extract_python_code,
    extract_adapted_plan,
    extract_thinking,
    format_retrieved_strategies,
    format_tag_hints,
    adapt_and_solve,
    free_generation,
    solve_with_fallbacks,
)


class TestExtractPythonCode:
    def test_markdown_python_fence(self):
        resp = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert extract_python_code(resp) == "print('hello')"

    def test_plain_fence(self):
        resp = "```\nprint(42)\n```"
        assert extract_python_code(resp) == "print(42)"

    def test_solution_section(self):
        resp = "### Thinking\nblah\n\n### Solution\nprint(1)\nprint(2)\n"
        code = extract_python_code(resp)
        assert "print(1)" in code

    def test_no_fence(self):
        resp = "print(99)"
        assert extract_python_code(resp) == "print(99)"


class TestExtractAdaptedPlan:
    def test_extracts_plan(self):
        resp = (
            "### MATCH ANALYSIS\nstuff\n\n"
            "### ADAPTED PLAN\n1. Step one\n2. Step two\n\n"
            "### NEXT SECTION\nmore\n"
        )
        plan = extract_adapted_plan(resp)
        assert "Step one" in plan
        assert "Step two" in plan

    def test_no_plan_section(self):
        resp = "Just some text without a plan section."
        plan = extract_adapted_plan(resp)
        assert plan == resp.strip()

    def test_plan_at_end(self):
        resp = "### ADAPTED PLAN\n1. Only step"
        plan = extract_adapted_plan(resp)
        assert "Only step" in plan


class TestExtractThinking:
    def test_extracts(self):
        resp = "### Thinking\nMy reasoning here.\n\n### Solution\ncode"
        assert "My reasoning here" in extract_thinking(resp)

    def test_no_thinking(self):
        assert extract_thinking("no thinking section") == ""


class TestFormatRetrievedStrategies:
    def test_formats(self, sample_episodic_entry):
        result = format_retrieved_strategies([(sample_episodic_entry, 0.85)])
        assert "Strategy 1" in result
        assert "0.850" in result
        assert "greedy" in result


class TestFormatTagHints:
    def test_formats(self, sample_episodic_entry):
        result = format_tag_hints([(sample_episodic_entry, 0.9)])
        assert "Similar Problem 1" in result
        assert "greedy" in result


class TestAdaptAndSolve:
    def test_returns_code(self, mock_llm_client, sample_problem, sample_episodic_entry):
        code, analysis, plan = adapt_and_solve(
            sample_problem,
            [(sample_episodic_entry, 0.9)],
            mock_llm_client,
        )
        assert len(code) > 0
        assert "ADAPTED PLAN" in analysis or "MATCH ANALYSIS" in analysis


class TestFreeGeneration:
    def test_returns_code(self, mock_llm_client, sample_problem):
        code, response, thinking = free_generation(sample_problem, mock_llm_client)
        assert len(code) > 0


class TestSolveWithFallbacks:
    def test_first_succeeds(self, mock_llm_client, sample_problem, sample_episodic_entry, verifier):
        """Mock LLM generates a+b code; sample_problem test cases expect a+b."""
        result = solve_with_fallbacks(
            sample_problem,
            [(sample_episodic_entry, 0.9)],
            mock_llm_client,
            verifier,
        )
        assert result["success"] is True
        assert result["method"] == "adapted_1"
        assert len(result["attempts"]) >= 1

    def test_falls_back_to_free(self, mock_llm_client, verifier):
        """When test cases don't match the mock output, all adapted fail → free gen."""
        hard_problem = Problem(
            problem_id="HARD",
            contest_id=1,
            index="A",
            title="Hard",
            statement="Print 'impossible'.",
            difficulty_rating=3000,
            algorithm_tags=["constructive"],
            test_cases=[{"input": "", "expected_output": "impossible"}],
        )
        strat = Strategy(
            technique_chain=["x"], key_insight="y", preconditions=[],
            algorithm_tags=["constructive"],
        )
        import numpy as np
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        entry = EpisodicEntry(
            entry_id="e_hard",
            problem_id="P_HARD",
            problem_statement="Hard problem.",
            strategy=strat,
            solution_code="pass",
            difficulty_rating=3000,
            verification_passed=True,
            embedding=emb,
        )
        result = solve_with_fallbacks(
            hard_problem,
            [(entry, 0.5)],
            mock_llm_client,
            verifier,
        )
        # The mock generates a+b code which won't print "impossible"
        # Both adapted and free should fail
        assert result["method"] in ("free", "failed")
        assert len(result["attempts"]) >= 2  # at least adapted + free
