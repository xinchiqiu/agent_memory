"""Tests for strategy_extraction.py — AST analysis and LLM extraction."""

import json
import pytest
from strategy_extraction import (
    extract_code_structure,
    _max_loop_depth,
    format_ast_features,
    parse_json_response,
    validate_strategy,
    _is_python,
    extract_strategy,
)
from data_structures import Strategy


class TestExtractCodeStructure:
    def test_simple_loop(self):
        code = "for i in range(10):\n    print(i)"
        features = extract_code_structure(code)
        assert features["loop_depth"] == 1
        assert features["main_loop_structure"] == "single_for"
        assert features["has_recursion"] is False

    def test_recursion(self):
        code = (
            "def solve(n):\n"
            "    if n <= 0: return 0\n"
            "    return solve(n - 1) + 1\n"
            "solve(10)"
        )
        features = extract_code_structure(code)
        assert features["has_recursion"] is True
        assert features["main_loop_structure"] == "recursion"
        assert features["num_functions"] == 1

    def test_dp_pattern(self):
        code = (
            "n = int(input())\n"
            "dp = [0] * (n + 1)\n"
            "for i in range(1, n + 1):\n"
            "    dp[i] = dp[i - 1] + 1\n"
        )
        features = extract_code_structure(code)
        assert features["has_dp_pattern"] is True

    def test_nested_loops(self):
        code = (
            "for i in range(10):\n"
            "    for j in range(10):\n"
            "        pass\n"
        )
        features = extract_code_structure(code)
        assert features["loop_depth"] == 2
        assert features["main_loop_structure"] == "nested_for"

    def test_sorting_detection(self):
        code = "arr = sorted([3, 1, 2])"
        features = extract_code_structure(code)
        assert features["has_sorting"] is True

    def test_binary_search_import(self):
        code = "import bisect\nx = bisect.bisect_left([1,2,3], 2)"
        features = extract_code_structure(code)
        assert features["has_binary_search"] is True

    def test_binary_search_pattern(self):
        code = (
            "lo, hi = 0, 100\n"
            "while lo <= hi:\n"
            "    mid = (lo + hi) // 2\n"
            "    lo = mid + 1\n"
        )
        features = extract_code_structure(code)
        assert features["has_binary_search"] is True

    def test_heap_usage(self):
        code = "import heapq\nheapq.heappush([], 1)"
        features = extract_code_structure(code)
        assert features["uses_heap"] is True

    def test_modular_arithmetic(self):
        code = "x = (a + b) % 1000000007"
        features = extract_code_structure(code)
        assert features["has_modular_arithmetic"] is True

    def test_invalid_code(self):
        features = extract_code_structure("this is not valid python {{{")
        assert features["loop_depth"] == 0
        assert features["has_recursion"] is False

    def test_dict_usage(self):
        code = "d = dict()\nd['a'] = 1"
        features = extract_code_structure(code)
        assert features["uses_dict"] is True

    def test_set_usage(self):
        code = "s = set()\ns.add(1)"
        features = extract_code_structure(code)
        assert features["uses_set"] is True


class TestFormatAstFeatures:
    def test_formats_all_keys(self):
        features = {"has_recursion": True, "loop_depth": 2}
        result = format_ast_features(features)
        assert "has_recursion: True" in result
        assert "loop_depth: 2" in result


class TestIsPython:
    def test_valid_python(self):
        assert _is_python("print('hello')") is True

    def test_cpp_code(self):
        assert _is_python("#include <iostream>\nint main() { return 0; }") is False

    def test_java_code(self):
        assert _is_python("public class Main { public static void main(String[] args) {} }") is False


class TestParseJsonResponse:
    def test_plain_json(self):
        resp = '{"key": "value"}'
        assert parse_json_response(resp) == {"key": "value"}

    def test_markdown_fenced(self):
        resp = "Here is the result:\n```json\n{\"key\": \"value\"}\n```\nDone."
        assert parse_json_response(resp) == {"key": "value"}

    def test_no_json(self):
        with pytest.raises(ValueError):
            parse_json_response("no json here")

    def test_nested_json(self):
        resp = '{"outer": {"inner": 42}}'
        result = parse_json_response(resp)
        assert result["outer"]["inner"] == 42


class TestValidateStrategy:
    def test_valid(self):
        d = {
            "technique_chain": ["step1"],
            "key_insight": "This is the insight.",
            "preconditions": ["pre1"],
            "algorithm_tags": ["greedy"],
        }
        assert validate_strategy(d) == []

    def test_missing_technique_chain(self):
        d = {
            "technique_chain": [],
            "key_insight": "insight",
            "preconditions": [],
            "algorithm_tags": ["greedy"],
        }
        issues = validate_strategy(d)
        assert len(issues) > 0
        assert "technique_chain" in issues[0]

    def test_short_insight(self):
        d = {
            "technique_chain": ["x"],
            "key_insight": "ab",
            "preconditions": [],
            "algorithm_tags": [],
        }
        issues = validate_strategy(d)
        assert any("key_insight" in i for i in issues)

    def test_bad_tags(self):
        d = {
            "technique_chain": ["x"],
            "key_insight": "Valid insight here",
            "preconditions": [],
            "algorithm_tags": ["not_a_real_tag"],
        }
        issues = validate_strategy(d)
        assert any("Unknown" in i for i in issues)


class TestExtractStrategy:
    def test_with_mock_llm(self, mock_llm_client, sample_problem):
        result = extract_strategy(
            sample_problem,
            "a,b=map(int,input().split())\nprint(a+b)",
            mock_llm_client,
        )
        assert isinstance(result, Strategy)
        assert len(result.technique_chain) > 0
        assert len(result.key_insight) > 0
        assert all(tag in {"greedy", "sorting"} for tag in result.algorithm_tags)
