"""Module 5: Verifier — runs generated code against test cases in a subprocess."""

import subprocess
import tempfile
import os
import logging
from typing import List, Dict, Tuple

from config import CONFIG


class Verifier:
    """Runs generated Python code against test cases in a sandboxed subprocess."""

    def __init__(self, timeout_seconds: int = None):
        self.timeout = timeout_seconds or CONFIG["timeout_seconds"]

    def verify(self, code: str, test_cases: List[Dict]) -> Tuple[bool, str]:
        """Run code against all test cases.

        Args:
            code: Python source code string.
            test_cases: List of {"input": str, "expected_output": str}.

        Returns:
            (all_passed, error_info) — error_info is "" if all passed,
            otherwise describes the first failure.
        """
        if not test_cases:
            return True, ""

        for i, tc in enumerate(test_cases):
            passed, actual_output, error = self._run_single(code, tc["input"])

            if not passed:
                snippet = lambda s: s[:300]
                return False, (
                    f"Test case {i+1} runtime error.\n"
                    f"Input: {snippet(tc['input'])}\n"
                    f"Expected: {snippet(tc['expected_output'])}\n"
                    f"Got: {snippet(actual_output)}\n"
                    f"Error: {snippet(error)}"
                )

            if not self._outputs_match(actual_output, tc["expected_output"]):
                snippet = lambda s: s[:300]
                return False, (
                    f"Test case {i+1}: wrong answer.\n"
                    f"Input: {snippet(tc['input'])}\n"
                    f"Expected: {snippet(tc['expected_output'])}\n"
                    f"Got: {snippet(actual_output)}"
                )

        return True, ""

    def _run_single(self, code: str, input_data: str) -> Tuple[bool, str, str]:
        """Run code with given input in a temporary file subprocess.

        Returns (success, stdout, stderr).
        """
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        try:
            tmp.write(code)
            tmp.flush()
            tmp.close()
            result = subprocess.run(
                ['python3', tmp.name],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Time limit exceeded"
        except Exception as e:
            return False, "", str(e)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _outputs_match(self, actual: str, expected: str) -> bool:
        """Compare outputs tolerantly (strip each line, ignore trailing blank lines)."""
        def normalise(s: str) -> List[str]:
            return [line.strip() for line in s.strip().split('\n')]
        return normalise(actual) == normalise(expected)
