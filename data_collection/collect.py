"""Main Codeforces data collection script.

Usage
-----
# Full collection (takes ~4 hours):
python data_collection/collect.py --output_dir dataset/

# Quick test (50 problems, skip solutions):
python data_collection/collect.py --output_dir dataset/ --max_problems 50 --skip_solutions

# Resume a previous run:
python data_collection/collect.py --output_dir dataset/ --resume

# Skip already-scraped problems, only collect solutions:
python data_collection/collect.py --output_dir dataset/ --resume --solutions_only

Expected runtime
----------------
  API calls             : ~1 min
  Statement scraping    : ~75 min for 3000 problems (1.5 s/request)
  Solution scraping     : ~150 min for 3000 problems (3 solutions × 1 s/request)
  Total                 : ~4 hours

Directory output
----------------
  dataset/
    index.json
    checkpoint.json
    problems/
      1850E.json
      ...
    splits/
      seed.json
      eval.json
      test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.cf_api import (
    fetch_problem_list,
    fetch_contest_dates,
    fetch_contest_submissions,
)
from data_collection.cf_scraper import (
    scrape_problem,
    scrape_accepted_solutions,
    scrape_editorial,
)
from data_collection.dataset_utils import (
    normalize_tags,
    create_splits,
    save_problem,
    load_all_problems,
    save_index,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Codeforces problems")
    p.add_argument("--output_dir",      default="dataset",   help="Root output directory")
    p.add_argument("--max_problems",    type=int, default=3000)
    p.add_argument("--min_rating",      type=int, default=800)
    p.add_argument("--max_rating",      type=int, default=2500)
    p.add_argument("--max_solutions",   type=int, default=3,  help="Accepted solutions per problem")
    p.add_argument("--delay",           type=float, default=1.5, help="Seconds between web requests")
    p.add_argument("--resume",          action="store_true",  help="Skip already-collected problems")
    p.add_argument("--skip_solutions",  action="store_true",  help="Do not scrape solution code")
    p.add_argument("--skip_editorials", action="store_true",  help="Do not scrape editorials")
    p.add_argument("--solutions_only",  action="store_true",  help="Only fill in missing solutions")
    p.add_argument("--seed_before",     default="2023-07-01")
    p.add_argument("--eval_before",     default="2024-07-01")
    return p.parse_args()


def setup_dirs(output_dir: str) -> None:
    for sub in ("problems", "splits"):
        Path(output_dir, sub).mkdir(parents=True, exist_ok=True)


def load_checkpoint(output_dir: str) -> set:
    ckpt = Path(output_dir) / "checkpoint.json"
    if ckpt.exists():
        with open(ckpt) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(output_dir: str, done: set) -> None:
    ckpt = Path(output_dir) / "checkpoint.json"
    with open(ckpt, "w") as f:
        json.dump(sorted(done), f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(Path(args.output_dir) / "collect.log"),
            logging.StreamHandler(),
        ],
    )
    setup_dirs(args.output_dir)

    # ------------------------------------------------------------------
    # Step 1: Problem list
    # ------------------------------------------------------------------
    logging.info("Step 1: Fetching problem list from API…")
    raw_problems, _ = fetch_problem_list(args.min_rating, args.max_rating)

    # Step 2: Contest dates
    logging.info("Step 2: Fetching contest dates…")
    contest_dates = fetch_contest_dates()

    # Attach dates, sort newest-first (we want newer = less contaminated)
    for p in raw_problems:
        p["contest_date"] = contest_dates.get(p["contestId"], {}).get("date", "2020-01-01")
    raw_problems.sort(key=lambda x: x["contest_date"], reverse=True)
    selected = raw_problems[: args.max_problems]
    logging.info(f"Selected {len(selected)} problems to collect")

    # ------------------------------------------------------------------
    # Checkpoint bookkeeping
    # ------------------------------------------------------------------
    done: set = set()
    if args.resume:
        done = load_checkpoint(args.output_dir)
        logging.info(f"Resuming: {len(done)} problems already done")

    # ------------------------------------------------------------------
    # Step 3: Scrape each problem
    # ------------------------------------------------------------------
    logging.info("Step 3: Scraping problems…")
    # Track which contest editorials we have already fetched
    editorial_cache: dict = {}

    for i, raw_p in enumerate(selected):
        pid = f"{raw_p['contestId']}{raw_p['index']}"
        cid = raw_p["contestId"]
        idx = raw_p["index"]

        if pid in done and not args.solutions_only:
            continue

        logging.info(f"[{i+1}/{len(selected)}] {pid} (rating={raw_p.get('rating','?')})")

        # --- Statement ---
        problem_file = Path(args.output_dir) / "problems" / f"{pid}.json"
        if problem_file.exists() and args.resume:
            with open(problem_file) as f:
                problem_data = json.load(f)
        else:
            stmt = scrape_problem(cid, idx, delay=args.delay)
            if not stmt:
                logging.warning(f"  Could not scrape {pid}, skipping")
                continue

            problem_data = {
                "problem_id":   pid,
                "contest_id":   cid,
                "index":        idx,
                "title":        raw_p.get("name", ""),
                "rating":       raw_p.get("rating", 0),
                "raw_tags":     raw_p.get("tags", []),
                "tags":         normalize_tags(raw_p.get("tags", [])),
                "contest_date": raw_p["contest_date"],
                "solved_count": raw_p.get("solvedCount", 0),
                **stmt,
                "editorial_text":      None,
                "reference_solutions": [],
            }

        # --- Editorial ---
        if not args.skip_editorials and not problem_data.get("editorial_text"):
            if cid not in editorial_cache:
                editorial_cache[cid] = scrape_editorial(cid, delay=args.delay)
            editorial_map = editorial_cache.get(cid, {})
            problem_data["editorial_text"] = editorial_map.get(idx)

        # --- Solutions ---
        if not args.skip_solutions:
            existing = problem_data.get("reference_solutions", [])
            if len(existing) < args.max_solutions:
                logging.info(f"  Fetching submissions for {pid}…")
                candidates = fetch_contest_submissions(cid, idx)
                time.sleep(args.delay)
                new_solutions = scrape_accepted_solutions(
                    cid, idx, candidates,
                    max_solutions=args.max_solutions - len(existing),
                    delay=args.delay,
                )
                problem_data["reference_solutions"] = existing + new_solutions
                logging.info(f"  Got {len(new_solutions)} new solutions ({len(problem_data['reference_solutions'])} total)")

        # --- Validate & save ---
        if not problem_data.get("sample_tests"):
            logging.warning(f"  {pid} has no sample tests — skipping")
            continue

        save_problem(problem_data, args.output_dir)
        done.add(pid)
        save_checkpoint(args.output_dir, done)

    # ------------------------------------------------------------------
    # Step 4: Create splits
    # ------------------------------------------------------------------
    logging.info("Step 4: Creating dataset splits…")
    all_problems = load_all_problems(args.output_dir)
    splits = create_splits(all_problems, contest_dates, args.seed_before, args.eval_before)
    for name, pids in splits.items():
        out = Path(args.output_dir) / "splits" / f"{name}.json"
        with open(out, "w") as f:
            json.dump(pids, f, indent=2)
        logging.info(f"  {name}: {len(pids)} problems")

    # ------------------------------------------------------------------
    # Step 5: Write index
    # ------------------------------------------------------------------
    save_index(all_problems, args.output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    with_solutions = sum(1 for p in all_problems if p.get("reference_solutions"))
    with_editorial = sum(1 for p in all_problems if p.get("editorial_text"))
    logging.info(
        f"\n{'='*50}\n"
        f"Collection complete!\n"
        f"  Total problems      : {len(all_problems)}\n"
        f"  With solutions      : {with_solutions}\n"
        f"  With editorial      : {with_editorial}\n"
        f"  Seed / Eval / Test  : {len(splits['seed'])} / "
        f"{len(splits['eval'])} / {len(splits['test'])}\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    main()
