"""Dataset assembly, tag normalisation, splits, and dataset loading utilities."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Tag normalisation
# ---------------------------------------------------------------------------

# Codeforces tag → our canonical taxonomy
TAG_MAPPING: Dict[str, str] = {
    "greedy":                       "greedy",
    "dp":                           "dp",
    "dynamic programming":          "dp",
    "binary search":                "binary_search",
    "brute force":                  "brute_force",
    "constructive algorithms":      "constructive",
    "data structures":              "implementation",   # broad; keep as implementation
    "dfs and similar":              "graph_dfs",
    "graphs":                       "graph_dfs",        # generic graph → dfs by default
    "shortest paths":               "graph_dijkstra",
    "trees":                        "graph_dfs",
    "flows":                        "network_flow",
    "dsu":                          "union_find",
    "implementation":               "implementation",
    "math":                         "math_number_theory",
    "number theory":                "math_number_theory",
    "combinatorics":                "math_combinatorics",
    "geometry":                     "geometry",
    "sortings":                     "sorting",
    "two pointers":                 "two_pointers",
    "bitmasks":                     "implementation",
    "divide and conquer":           "divide_and_conquer",
    "hashing":                      "string_hashing",
    "string suffix structures":     "string_kmp",
    "strings":                      "string_hashing",
    "games":                        "constructive",
    "probabilities":                "math_combinatorics",
    "interactive":                  "implementation",
    "ternary search":               "binary_search",
    "meet-in-the-middle":           "divide_and_conquer",
    "fft":                          "math_number_theory",
    "2-sat":                        "graph_dfs",
    "expression parsing":           "implementation",
    "matrices":                     "dp",
    "schedules":                    "greedy",
    "chinese remainder theorem":    "math_number_theory",
}

# These are the tags our system actually uses — everything gets mapped into this set
CANONICAL_TAGS = {
    "greedy", "dp", "binary_search", "graph_bfs", "graph_dfs",
    "graph_dijkstra", "graph_mst", "segment_tree", "binary_indexed_tree",
    "two_pointers", "sliding_window", "divide_and_conquer",
    "math_number_theory", "math_combinatorics", "string_hashing",
    "string_kmp", "union_find", "topological_sort", "network_flow",
    "geometry", "brute_force", "constructive", "implementation",
    "sorting", "stack", "priority_queue",
}


def normalize_tags(raw_tags: List[str]) -> List[str]:
    """Map Codeforces raw tags to canonical taxonomy, dropping unknowns."""
    seen = set()
    result = []
    for tag in raw_tags:
        canonical = TAG_MAPPING.get(tag.lower().strip())
        if canonical and canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

def create_splits(all_problems: List[dict],
                  contest_dates: Dict[int, dict],
                  seed_before: str = "2023-07-01",
                  eval_before: str = "2024-07-01") -> Dict[str, List[str]]:
    """Split problem IDs into seed / eval / test by contest date.

    Temporal split rationale:
      seed  — older problems, likely in LLM training data; used to bootstrap memory
      eval  — middle period; main evaluation
      test  — newest problems; held-out final evaluation, least contamination risk

    Args:
        all_problems: List of assembled problem dicts (must have "contest_id" and "problem_id").
        contest_dates: Output of cf_api.fetch_contest_dates().
        seed_before: Problems from contests before this date go to seed.
        eval_before: Problems from contests in [seed_before, eval_before) go to eval.

    Returns:
        {"seed": [...], "eval": [...], "test": [...]}  — each a list of problem_ids
        sorted chronologically within each split.
    """
    seed, eval_set, test = [], [], []

    for p in all_problems:
        cid = p["contest_id"]
        date = contest_dates.get(cid, {}).get("date", "2020-01-01")
        entry = (date, p["problem_id"])
        if date < seed_before:
            seed.append(entry)
        elif date < eval_before:
            eval_set.append(entry)
        else:
            test.append(entry)

    # Sort by date within each split (chronological order)
    seed.sort()
    eval_set.sort()
    test.sort()

    return {
        "seed":  [pid for _, pid in seed],
        "eval":  [pid for _, pid in eval_set],
        "test":  [pid for _, pid in test],
    }


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_problem(p: dict) -> List[str]:
    """Return a list of issues with a problem dict (empty = OK)."""
    issues = []
    if not p.get("statement"):
        issues.append("missing statement")
    if not p.get("sample_tests"):
        issues.append("no sample tests")
    if not p.get("tags"):
        issues.append("no canonical tags")
    if not p.get("reference_solutions"):
        issues.append("no reference solutions")
    return issues


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def save_problem(problem: dict, output_dir: str) -> None:
    """Save a problem dict to <output_dir>/problems/<problem_id>.json."""
    path = Path(output_dir) / "problems" / f"{problem['problem_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(problem, f, indent=2, ensure_ascii=False)


def load_problem(problem_id: str, dataset_dir: str) -> Optional[dict]:
    """Load a single problem dict from disk."""
    path = Path(dataset_dir) / "problems" / f"{problem_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_problems(dataset_dir: str) -> List[dict]:
    """Load all problem dicts from <dataset_dir>/problems/*.json."""
    problems_dir = Path(dataset_dir) / "problems"
    if not problems_dir.exists():
        return []
    problems = []
    for fpath in sorted(problems_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            problems.append(json.load(f))
    logging.info(f"Loaded {len(problems)} problems from {dataset_dir}")
    return problems


def load_split(split_name: str, dataset_dir: str) -> List[str]:
    """Load a list of problem IDs for a named split."""
    path = Path(dataset_dir) / "splits" / f"{split_name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_index(problems: List[dict], dataset_dir: str) -> None:
    """Write a lightweight index.json for quick browsing."""
    index = [
        {
            "problem_id":     p["problem_id"],
            "title":          p.get("title", ""),
            "rating":         p.get("rating", 0),
            "tags":           p.get("tags", []),
            "contest_date":   p.get("contest_date", ""),
            "has_solutions":  len(p.get("reference_solutions", [])) > 0,
            "num_samples":    len(p.get("sample_tests", [])),
        }
        for p in problems
    ]
    out = Path(dataset_dir) / "index.json"
    with open(out, "w") as f:
        json.dump(index, f, indent=2)
    logging.info(f"Wrote index with {len(index)} entries to {out}")


# ---------------------------------------------------------------------------
# Problem → data_structures.Problem conversion
# ---------------------------------------------------------------------------

def dict_to_problem(d: dict):
    """Convert a raw dataset dict to a data_structures.Problem object.

    Adds the parent directory to sys.path if needed so we can import
    data_structures from a subdirectory context.
    """
    # Allow importing from parent package when called from data_collection/
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from data_structures import Problem

    # Build test_cases from sample_tests
    test_cases = [
        {"input": t["input"], "expected_output": t["output"]}
        for t in d.get("sample_tests", [])
    ]

    return Problem(
        problem_id=d["problem_id"],
        contest_id=d.get("contest_id", 0),
        index=d.get("index", ""),
        title=d.get("title", ""),
        statement=_build_full_statement(d),
        difficulty_rating=d.get("rating", 0),
        algorithm_tags=d.get("tags", []),
        test_cases=test_cases,
        editorial=d.get("editorial_text"),
        reference_solutions=[s["code"] for s in d.get("reference_solutions", [])],
        contest_date=d.get("contest_date", ""),
    )


def dict_to_seed_tuples(d: dict) -> List[Tuple]:
    """Convert a problem dict to a list of (Problem, code, language) seed triples.

    Returns one tuple per reference solution.  The agent's seed_memory()
    accepts these directly, and strategy_extraction will skip AST analysis
    for non-Python languages automatically.
    """
    problem = dict_to_problem(d)
    tuples = []
    for sol in d.get("reference_solutions", []):
        code = sol.get("code", "")
        lang = sol.get("language", "")
        if code:
            tuples.append((problem, code, lang))
    return tuples


def _build_full_statement(d: dict) -> str:
    """Concatenate statement sections into one string for the LLM."""
    parts = []
    if d.get("statement"):
        parts.append(d["statement"])
    if d.get("input_spec"):
        parts.append(f"Input\n{d['input_spec']}")
    if d.get("output_spec"):
        parts.append(f"Output\n{d['output_spec']}")
    if d.get("sample_tests"):
        examples = []
        for i, t in enumerate(d["sample_tests"], 1):
            examples.append(f"Example {i}:\nInput:\n{t['input']}\nOutput:\n{t['output']}")
        parts.append("\n".join(examples))
    if d.get("note"):
        parts.append(f"Note\n{d['note']}")
    return "\n\n".join(parts)
