"""Module 2: Strategy extraction from solved problems.

Two-stage pipeline:
  1. AST-based structural analysis (deterministic)
  2. LLM-based semantic extraction
"""

import ast
import json
import logging
import re
from typing import Optional

from data_structures import Problem, Strategy
from config import CONFIG, ALLOWED_ALGORITHM_TAGS


# ---------------------------------------------------------------------------
# Stage 1: AST structural analysis
# ---------------------------------------------------------------------------

def extract_code_structure(code: str) -> dict:
    """Parse Python code and extract structural features from the AST.

    Returns a dict with boolean/integer features describing the code's shape.
    """
    features = {
        "has_recursion": False,
        "has_sorting": False,
        "has_binary_search": False,
        "loop_depth": 0,
        "uses_dict": False,
        "uses_heap": False,
        "uses_deque": False,
        "uses_set": False,
        "has_dp_pattern": False,
        "has_graph_pattern": False,
        "has_modular_arithmetic": False,
        "num_functions": 0,
        "main_loop_structure": "none",
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features

    # Collect function names defined in the code
    defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    for node in ast.walk(tree):
        # Recursion: function calls itself
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name and func_name in defined_funcs:
                features["has_recursion"] = True
            if func_name in ("sort", "sorted"):
                features["has_sorting"] = True

        # Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                if "heapq" in name:
                    features["uses_heap"] = True
                if "bisect" in name:
                    features["has_binary_search"] = True
                if "collections" in name:
                    pass  # checked below via usage

        # Dict / set / deque usage
        if isinstance(node, ast.Dict):
            features["uses_dict"] = True
        if isinstance(node, ast.Set):
            features["uses_set"] = True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "deque":
                features["uses_deque"] = True
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                features["uses_dict"] = True
            if isinstance(node.func, ast.Name) and node.func.id == "set":
                features["uses_set"] = True

        # Modular arithmetic: large modulus constant
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                if node.right.value > 1_000_000:
                    features["has_modular_arithmetic"] = True

        # Function count
        if isinstance(node, ast.FunctionDef):
            features["num_functions"] += 1

    # Loop depth (maximum nesting)
    features["loop_depth"] = _max_loop_depth(tree)

    # Manual binary search pattern: while lo <= hi with mid calculation
    if re.search(r'while\s+\w+\s*[<>]=?\s*\w+', code) and re.search(r'mid\s*=', code):
        features["has_binary_search"] = True

    # DP pattern: array subscript assigned using a previous index
    if re.search(r'dp\s*\[', code) or re.search(r'\w+\[i\]\s*=\s*\w+\[i\s*[+-]', code):
        features["has_dp_pattern"] = True

    # Graph pattern: adjacency list construction
    if re.search(r'(adj|graph|neighbors?)\s*\[', code) or re.search(r'\.append\(', code):
        features["has_graph_pattern"] = True

    # Main loop structure
    if features["has_recursion"]:
        features["main_loop_structure"] = "recursion"
    elif features["loop_depth"] >= 2:
        features["main_loop_structure"] = "nested_for"
    elif features["loop_depth"] == 1:
        # Distinguish while vs for
        if re.search(r'\bwhile\b', code):
            features["main_loop_structure"] = "while"
        else:
            features["main_loop_structure"] = "single_for"

    return features


def _max_loop_depth(tree: ast.AST) -> int:
    """Return the maximum nesting depth of for/while loops in the AST."""
    def depth(node, current=0):
        if isinstance(node, (ast.For, ast.While)):
            current += 1
        max_d = current
        for child in ast.iter_child_nodes(node):
            max_d = max(max_d, depth(child, current))
        return max_d
    return depth(tree)


def format_ast_features(features: dict) -> str:
    """Convert AST feature dict into a human-readable string for prompt injection."""
    lines = []
    for k, v in features.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2: LLM semantic extraction
# ---------------------------------------------------------------------------

STRATEGY_EXTRACTION_PROMPT = """\
You are an expert competitive programmer analyzing a solved problem.
Given the problem statement, the working solution code, and some structural \
features of the code, extract the solution strategy.

## Problem Statement
{problem_statement}

## Working Solution Code
{solution_code}

## Code Structural Features
{ast_features_formatted}

## Your Task

Extract the solution strategy by filling in the following JSON structure.
Be precise and specific. The strategy should be useful for someone solving a \
SIMILAR (but different) problem — capture the transferable reasoning, not the \
problem-specific details.

Return ONLY valid JSON, no other text:

{{
    "technique_chain": [
        "step 1 ...",
        "step 2 ..."
    ],
    "key_insight": "The single most important idea. Explain WHY, not just WHAT. 1-3 sentences.",
    "preconditions": [
        "precondition 1 ...",
        "precondition 2 ..."
    ],
    "complexity": "O(...) time, O(...) space",
    "algorithm_tags": ["tag1", "tag2"]
}}

Allowed algorithm_tags: greedy, dp, binary_search, graph_bfs, graph_dfs, \
graph_dijkstra, graph_mst, segment_tree, binary_indexed_tree, two_pointers, \
sliding_window, divide_and_conquer, math_number_theory, math_combinatorics, \
string_hashing, string_kmp, union_find, topological_sort, network_flow, \
geometry, brute_force, constructive, implementation, sorting, stack, priority_queue
"""

STRATEGY_EXTRACTION_RETRY_PROMPT = """\
The previous extraction had the following issues: {issues}

Please re-extract the strategy and fix these issues.
Return ONLY valid JSON with no other text.

## Problem Statement
{problem_statement}

## Working Solution Code
{solution_code}

## Code Structural Features
{ast_features_formatted}

Return the corrected JSON:
"""


def parse_json_response(response: str) -> dict:
    """Extract and parse the first JSON object from a response string."""
    # Strip markdown fences if present
    response = re.sub(r'```(?:json)?\s*', '', response)
    response = response.strip().rstrip('`').strip()
    # Try to find the outermost {...}
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {response[:200]}")


def validate_strategy(strategy_dict: dict) -> list:
    """Return a list of validation issues (empty = OK)."""
    issues = []
    tc = strategy_dict.get("technique_chain", [])
    if not isinstance(tc, list) or not (3 <= len(tc) <= 7):
        issues.append(f"technique_chain must be a list of 3–7 steps, got {len(tc) if isinstance(tc, list) else type(tc)}")

    ki = strategy_dict.get("key_insight", "")
    if not isinstance(ki, str) or not (20 <= len(ki) <= 600):
        issues.append(f"key_insight must be 20–600 chars, got {len(ki)}")

    pc = strategy_dict.get("preconditions", [])
    if not isinstance(pc, list) or not (2 <= len(pc) <= 5):
        issues.append(f"preconditions must be a list of 2–5 items, got {len(pc) if isinstance(pc, list) else type(pc)}")

    tags = strategy_dict.get("algorithm_tags", [])
    if not isinstance(tags, list):
        issues.append("algorithm_tags must be a list")
    else:
        bad = [t for t in tags if t not in ALLOWED_ALGORITHM_TAGS]
        if bad:
            issues.append(f"Unknown algorithm_tags: {bad}")

    return issues


def _is_python(code: str) -> bool:
    """Heuristic check: does this code look like Python (vs C++/Java)?"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_strategy(problem: Problem, solution_code: str, llm_client,
                     solution_language: str = "") -> Strategy:
    """Extract a Strategy from a solved problem using AST + LLM.

    Accepts solution code in any language. AST structural analysis is only
    applied for Python code; for other languages (C++, Java, etc.) the LLM
    extracts strategy directly from the source text.

    Args:
        problem: The solved problem.
        solution_code: Working solution code (any language).
        llm_client: LLM client with a .generate() method.
        solution_language: Language name string (e.g. "C++17", "Python 3").
                           Used only for display; auto-detected if empty.

    Returns:
        Strategy object.
    """
    # AST analysis only for Python
    is_py = _is_python(solution_code) or "python" in solution_language.lower()
    if is_py:
        ast_features = extract_code_structure(solution_code)
        ast_features_str = format_ast_features(ast_features)
    else:
        lang_label = solution_language or "non-Python"
        ast_features_str = f"  language: {lang_label}\n  (AST analysis not available for this language)"

    prompt = STRATEGY_EXTRACTION_PROMPT.format(
        problem_statement=problem.statement,
        solution_code=solution_code,
        ast_features_formatted=ast_features_str,
    )

    response = llm_client.generate(
        prompt,
        model=CONFIG.get("extraction_model"),
        temperature=0.0,
        max_tokens=1024,
    )

    try:
        strategy_dict = parse_json_response(response)
        issues = validate_strategy(strategy_dict)
    except Exception as e:
        issues = [f"JSON parse error: {e}"]
        strategy_dict = {}

    if issues:
        logging.warning(f"Strategy extraction issues for {problem.problem_id}: {issues}. Retrying…")
        retry_prompt = STRATEGY_EXTRACTION_RETRY_PROMPT.format(
            issues="; ".join(issues),
            problem_statement=problem.statement,
            solution_code=solution_code,
            ast_features_formatted=ast_features_str,
        )
        response = llm_client.generate(retry_prompt, temperature=0.0, max_tokens=1024)
        try:
            strategy_dict = parse_json_response(response)
            issues = validate_strategy(strategy_dict)
            if issues:
                logging.warning(f"Retry still has issues: {issues}. Using best-effort result.")
        except Exception as e:
            logging.error(f"Retry parse failed: {e}. Falling back to empty strategy.")
            strategy_dict = {
                "technique_chain": ["solve the problem"],
                "key_insight": "No insight extracted.",
                "preconditions": ["generic problem"],
                "complexity": "Unknown",
                "algorithm_tags": ["implementation"],
            }

    # Clamp tags to allowed set
    tags = [t for t in strategy_dict.get("algorithm_tags", []) if t in ALLOWED_ALGORITHM_TAGS]
    if not tags:
        tags = ["implementation"]

    return Strategy(
        technique_chain=strategy_dict.get("technique_chain", []),
        key_insight=strategy_dict.get("key_insight", ""),
        preconditions=strategy_dict.get("preconditions", []),
        complexity=strategy_dict.get("complexity"),
        algorithm_tags=tags,
    )
