"""Tests for verifier.py — code execution and verification."""

import pytest
from verifier import Verifier


@pytest.fixture
def v():
    return Verifier(timeout_seconds=5, max_tests=10)


class TestVerifier:
    def test_correct_solution(self, v):
        code = "a, b = map(int, input().split())\nprint(a + b)"
        tests = [
            {"input": "2 3", "expected_output": "5"},
            {"input": "10 20", "expected_output": "30"},
        ]
        passed, err = v.verify(code, tests)
        assert passed is True
        assert err == ""

    def test_wrong_answer(self, v):
        code = "a, b = map(int, input().split())\nprint(a - b)"
        tests = [{"input": "2 3", "expected_output": "5"}]
        passed, err = v.verify(code, tests)
        assert passed is False
        assert "wrong answer" in err.lower()

    def test_runtime_error(self, v):
        code = "x = 1 / 0"
        tests = [{"input": "", "expected_output": "0"}]
        passed, err = v.verify(code, tests)
        assert passed is False
        assert "runtime error" in err.lower() or "Error" in err

    def test_timeout(self):
        v = Verifier(timeout_seconds=1, max_tests=1)
        code = "while True: pass"
        tests = [{"input": "", "expected_output": ""}]
        passed, err = v.verify(code, tests)
        assert passed is False
        assert "time limit" in err.lower() or "timeout" in err.lower()

    def test_empty_test_cases(self, v):
        passed, err = v.verify("print('hello')", [])
        assert passed is True

    def test_output_normalization(self, v):
        code = "print('hello  ')\nprint()"
        tests = [{"input": "", "expected_output": "hello"}]
        passed, _ = v.verify(code, tests)
        assert passed is True

    def test_multiple_test_cases_all_pass(self, v):
        code = "print(int(input()) * 2)"
        tests = [
            {"input": "1", "expected_output": "2"},
            {"input": "5", "expected_output": "10"},
            {"input": "0", "expected_output": "0"},
        ]
        passed, err = v.verify(code, tests)
        assert passed is True

    def test_multiple_test_cases_second_fails(self, v):
        code = "print(int(input()) + 1)"
        tests = [
            {"input": "0", "expected_output": "1"},
            {"input": "5", "expected_output": "10"},  # will fail: gets 6
        ]
        passed, err = v.verify(code, tests)
        assert passed is False
        assert "Test case 2" in err

    def test_empty_code(self, v):
        passed, err = v.verify("", [{"input": "", "expected_output": "hello"}])
        assert passed is False

    def test_multiline_output(self, v):
        code = "print(1)\nprint(2)\nprint(3)"
        tests = [{"input": "", "expected_output": "1\n2\n3"}]
        passed, _ = v.verify(code, tests)
        assert passed is True
