"""Tests for data_collection/dataset_utils.py — tag normalization, splits, I/O."""

import sys
import os
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_collection"))

from dataset_utils import normalize_tags, validate_problem, dict_to_problem, _build_full_statement, create_splits


class TestNormalizeTags:
    def test_basic_mapping(self):
        tags = normalize_tags(["greedy", "dp", "sortings"])
        assert "greedy" in tags
        assert "dp" in tags
        assert "sorting" in tags

    def test_unknown_tags_dropped(self):
        tags = normalize_tags(["*special problem", "totally_fake_tag"])
        assert len(tags) == 0

    def test_graph_tags(self):
        tags = normalize_tags(["dfs and similar", "shortest paths"])
        assert "graph_dfs" in tags
        assert "graph_shortest_path" in tags

    def test_math_tags(self):
        tags = normalize_tags(["number theory", "combinatorics"])
        assert "number_theory" in tags
        assert "combinatorics" in tags

    def test_dedup(self):
        tags = normalize_tags(["greedy", "greedy"])
        assert tags.count("greedy") == 1


class TestValidateProblem:
    def test_valid(self):
        p = {
            "problem_id": "1A",
            "statement": "Some problem text here.",
            "tags": ["greedy"],
            "sample_tests": [{"input": "1", "output": "2"}],
            "reference_solutions": [{"code": "print(2)", "language": "Python 3"}],
        }
        assert validate_problem(p) == []

    def test_missing_statement(self):
        p = {
            "problem_id": "1A",
            "statement": "",
            "tags": ["greedy"],
            "sample_tests": [{"input": "1", "output": "2"}],
            "reference_solutions": [{"code": "print(2)", "language": "Python 3"}],
        }
        issues = validate_problem(p)
        assert len(issues) > 0
        assert "missing statement" in issues

    def test_missing_tags(self):
        p = {
            "problem_id": "1A",
            "statement": "Some text",
            "tags": [],
            "sample_tests": [{"input": "1", "output": "2"}],
            "reference_solutions": [{"code": "print(2)", "language": "Python 3"}],
        }
        issues = validate_problem(p)
        assert len(issues) > 0
        assert "no canonical tags" in issues


class TestDictToProblem:
    def test_converts_correctly(self):
        d = {
            "problem_id": "42A",
            "contest_id": 42,
            "index": "A",
            "title": "Test Problem",
            "statement": "Do something.",
            "rating": 1200,
            "tags": ["greedy", "math"],
            "sample_tests": [{"input": "1", "output": "2"}],
            "all_tests": [{"input": "1", "output": "2"}],
            "reference_solutions": [{"code": "print(2)", "language": "Python 3"}],
        }
        prob = dict_to_problem(d)
        assert prob.problem_id == "42A"
        assert prob.difficulty_rating == 1200
        assert "greedy" in prob.algorithm_tags


class TestCreateSplits:
    def test_temporal_ordering(self):
        problems = [
            {"contest_id": 100, "problem_id": "100A"},
            {"contest_id": 200, "problem_id": "200A"},
            {"contest_id": 300, "problem_id": "300A"},
            {"contest_id": 400, "problem_id": "400A"},
        ]
        contest_dates = {
            100: {"date": "2022-01-01"},
            200: {"date": "2023-01-01"},
            300: {"date": "2023-09-01"},
            400: {"date": "2025-01-01"},
        }
        splits = create_splits(
            problems, contest_dates,
            seed_before="2023-07-01",
            eval_before="2024-07-01",
        )
        assert splits["seed"] == ["100A", "200A"]
        assert splits["eval"] == ["300A"]
        assert splits["test"] == ["400A"]

    def test_empty_input(self):
        splits = create_splits([], {})
        assert splits["seed"] == []
        assert splits["eval"] == []
        assert splits["test"] == []


class TestBuildFullStatement:
    def test_all_sections(self):
        p = {
            "statement": "Main statement.",
            "input_spec": "One integer.",
            "output_spec": "Print result.",
            "note": "See examples.",
        }
        full = _build_full_statement(p)
        assert "Main statement." in full
        assert "One integer." in full
        assert "Print result." in full

    def test_minimal(self):
        p = {"statement": "Just this."}
        full = _build_full_statement(p)
        assert "Just this." in full
