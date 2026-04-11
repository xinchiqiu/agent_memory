#!/usr/bin/env python3
"""Load the CodeContests dataset from HuggingFace and save in our format.

This replaces the Codeforces scraping pipeline with a clean, reliable data source.
CodeContests provides: problem statements, reference solutions, public tests,
private tests, AND generated tests (thousands per problem).

Usage
-----
# Full dataset (CF problems only, with ratings):
    python data_collection/load_codecontests.py --output_dir dataset_cc/

# Filter by rating range:
    python data_collection/load_codecontests.py --output_dir dataset_cc/ --min_rating 800 --max_rating 2500

# Quick test:
    python data_collection/load_codecontests.py --output_dir dataset_cc/ --max_problems 50

Output
------
    dataset_cc/
      index.json              # lightweight metadata index
      problems/
        {contest_id}{index}.json   # one file per problem
      splits/
        seed.json             # older problems (contest_id < seed_cutoff)
        eval.json             # middle problems
        test.json             # CC test+valid splits (newest, 282 problems)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.dataset_utils import normalize_tags, save_index


# CodeContests language ID → readable name
_LANG_MAP = {
    1: "Python2", 2: "C++", 3: "Python3", 4: "Java", 9: "C",
    12: "PyPy3", 14: "Go", 19: "Rust", 54: "TypeScript", 55: "JavaScript",
}


def load_and_save(
    output_dir: str,
    min_rating: int = 800,
    max_rating: int = 3500,
    max_problems: Optional[int] = None,
    max_solutions: int = 5,
    max_generated_tests: int = 100,
    seed_cutoff: int = 1200,
    eval_cutoff: int = 1400,
):
    """Load CodeContests from HuggingFace and save to our format.

    Args:
        output_dir: Directory to save processed problems.
        min_rating: Minimum CF rating to include.
        max_rating: Maximum CF rating to include.
        max_problems: Limit total problems (for testing).
        max_solutions: Max reference solutions per problem.
        max_generated_tests: Max generated test cases to store per problem.
        seed_cutoff: Contest ID below this goes to seed split.
        eval_cutoff: Contest ID in [seed_cutoff, eval_cutoff) goes to eval.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    (out / "problems").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    all_problems = []
    seed_ids, eval_ids, test_ids = [], [], []

    # Process train split (bulk of the data)
    for split_name in ("train", "test", "valid"):
        logging.info(f"Loading CodeContests '{split_name}' split (streaming)...")
        ds = load_dataset("deepmind/code_contests", split=split_name, streaming=True)

        for row in ds:
            cf_cid = row.get("cf_contest_id", 0)
            cf_idx = row.get("cf_index", "")
            cf_rating = row.get("cf_rating", 0)

            # Only keep Codeforces problems with ratings
            if cf_cid == 0 or not cf_idx:
                continue
            if cf_rating < min_rating or cf_rating > max_rating:
                continue

            problem_id = f"{cf_cid}{cf_idx}"

            # Extract solutions
            raw_sols = row.get("solutions", {})
            sol_codes = raw_sols.get("solution", [])
            sol_langs = raw_sols.get("language", [])
            solutions = []
            for j, code in enumerate(sol_codes):
                if not isinstance(code, str) or len(code) < 50:
                    continue
                lang_id = sol_langs[j] if j < len(sol_langs) else -1
                lang_name = _LANG_MAP.get(lang_id, f"lang_{lang_id}")
                solutions.append({"code": code, "language": lang_name})
                if len(solutions) >= max_solutions:
                    break

            # Extract test cases
            public_tests = _extract_tests(row.get("public_tests", {}))
            private_tests = _extract_tests(row.get("private_tests", {}))
            generated_tests = _extract_tests(
                row.get("generated_tests", {}), max_count=max_generated_tests
            )

            # All tests combined for verification (public + private + generated)
            all_tests = public_tests + private_tests + generated_tests

            if not public_tests:
                continue  # Need at least public tests

            # Normalize tags
            raw_tags = row.get("cf_tags", [])
            tags = normalize_tags(raw_tags)

            problem = {
                "problem_id": problem_id,
                "contest_id": cf_cid,
                "index": cf_idx,
                "title": row.get("name", ""),
                "rating": cf_rating,
                "raw_tags": raw_tags,
                "tags": tags,
                "statement": row.get("description", ""),
                "sample_tests": public_tests,
                "private_tests": private_tests,
                "generated_tests": generated_tests,
                "all_tests": all_tests,
                "reference_solutions": solutions,
                "time_limit": row.get("time_limit"),
                "memory_limit_bytes": row.get("memory_limit_bytes", 0),
                "source_split": split_name,
                "difficulty": row.get("difficulty", 0),
            }

            # Save problem file
            prob_path = out / "problems" / f"{problem_id}.json"
            with open(prob_path, "w", encoding="utf-8") as f:
                json.dump(problem, f, indent=2, ensure_ascii=False)

            all_problems.append(problem)

            # Assign to split
            if split_name in ("test", "valid"):
                # CC test/valid splits are the newest → our test split
                test_ids.append(problem_id)
            elif cf_cid < seed_cutoff:
                seed_ids.append(problem_id)
            elif cf_cid < eval_cutoff:
                eval_ids.append(problem_id)
            else:
                test_ids.append(problem_id)

            if max_problems and len(all_problems) >= max_problems:
                break

        if max_problems and len(all_problems) >= max_problems:
            break

    # Sort within splits by contest ID (chronological proxy)
    seed_ids.sort()
    eval_ids.sort()
    test_ids.sort()

    # Save splits
    for name, ids in [("seed", seed_ids), ("eval", eval_ids), ("test", test_ids)]:
        with open(out / "splits" / f"{name}.json", "w") as f:
            json.dump(ids, f, indent=2)

    # Save index
    save_index(all_problems, output_dir)

    # Summary
    with_sols = sum(1 for p in all_problems if p["reference_solutions"])
    with_gen = sum(1 for p in all_problems if p["generated_tests"])
    avg_tests = (
        sum(len(p["all_tests"]) for p in all_problems) / len(all_problems)
        if all_problems else 0
    )

    logging.info(
        f"\n{'='*60}\n"
        f"CodeContests dataset loaded!\n"
        f"  Total CF problems     : {len(all_problems)}\n"
        f"  With solutions        : {with_sols}\n"
        f"  With generated tests  : {with_gen}\n"
        f"  Avg tests per problem : {avg_tests:.0f}\n"
        f"  Rating range          : {min_rating}-{max_rating}\n"
        f"  Seed / Eval / Test    : {len(seed_ids)} / {len(eval_ids)} / {len(test_ids)}\n"
        f"  Saved to              : {output_dir}\n"
        f"{'='*60}"
    )


def _extract_tests(tests_dict: dict, max_count: Optional[int] = None) -> list:
    """Convert CodeContests test format to our format."""
    inputs = tests_dict.get("input", [])
    outputs = tests_dict.get("output", [])
    tests = []
    for inp, out in zip(inputs, outputs):
        tests.append({"input": inp, "output": out})
        if max_count and len(tests) >= max_count:
            break
    return tests


def main():
    parser = argparse.ArgumentParser(description="Load CodeContests dataset")
    parser.add_argument("--output_dir", default="dataset_cc/")
    parser.add_argument("--min_rating", type=int, default=800)
    parser.add_argument("--max_rating", type=int, default=2500)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--max_solutions", type=int, default=5)
    parser.add_argument("--max_generated_tests", type=int, default=100)
    parser.add_argument("--seed_cutoff", type=int, default=1200,
                        help="Contest ID below this → seed split")
    parser.add_argument("--eval_cutoff", type=int, default=1400,
                        help="Contest ID in [seed_cutoff, eval_cutoff) → eval split")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    load_and_save(
        output_dir=args.output_dir,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_problems=args.max_problems,
        max_solutions=args.max_solutions,
        max_generated_tests=args.max_generated_tests,
        seed_cutoff=args.seed_cutoff,
        eval_cutoff=args.eval_cutoff,
    )


if __name__ == "__main__":
    main()
