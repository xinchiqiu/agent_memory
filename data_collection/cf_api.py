"""Codeforces API client.

Wraps the three API endpoints we need:
  - problemset.problems  — all problems with tags and ratings
  - contest.list         — all contests with dates
  - contest.status       — submissions for a specific contest
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

CF_API_BASE = "https://codeforces.com/api"

# Codeforces allows ~1 request per second from a single IP.
_DEFAULT_DELAY = 1.0


def _get(endpoint: str,
         params: Optional[dict] = None,
         retries: int = 5,
         delay: float = _DEFAULT_DELAY) -> dict:
    """GET a Codeforces API endpoint with retry/backoff on failure.

    Returns the parsed JSON dict.  Raises on persistent failure.
    """
    url = f"{CF_API_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "OK":
                return data
            # CF-level error (rate limit returns status=FAILED)
            comment = data.get("comment", "unknown CF error")
            logging.warning(f"CF API {endpoint} returned FAILED: {comment} (attempt {attempt+1})")
        except requests.RequestException as e:
            logging.warning(f"HTTP error for {endpoint}: {e} (attempt {attempt+1})")

        backoff = delay * (2 ** attempt)
        logging.info(f"Retrying in {backoff:.1f}s…")
        time.sleep(backoff)

    raise RuntimeError(f"CF API {endpoint} failed after {retries} attempts")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def fetch_problem_list(min_rating: int = 800,
                       max_rating: int = 2500) -> Tuple[List[dict], Dict[Tuple, int]]:
    """Fetch all problems with metadata from CF API.

    Args:
        min_rating: Minimum difficulty rating to include.
        max_rating: Maximum difficulty rating to include.

    Returns:
        (problems, solved_counts)
        problems: list of raw CF problem dicts with an extra "solvedCount" key
        solved_counts: (contestId, index) -> solvedCount mapping
    """
    data = _get("problemset.problems")
    raw_problems = data["result"]["problems"]
    stats_raw = data["result"]["problemStatistics"]
    stats: Dict[Tuple, int] = {
        (s["contestId"], s["index"]): s["solvedCount"] for s in stats_raw
    }

    problems = []
    for p in raw_problems:
        if "rating" not in p:
            continue
        if not (min_rating <= p["rating"] <= max_rating):
            continue
        if not p.get("tags"):
            continue
        p["solvedCount"] = stats.get((p["contestId"], p["index"]), 0)
        problems.append(p)

    logging.info(f"fetch_problem_list: {len(problems)} problems after filtering")
    return problems, stats


def fetch_contest_dates() -> Dict[int, dict]:
    """Fetch all finished contests with their start dates.

    Returns:
        contest_id -> {"date": "YYYY-MM-DD", "name": str, "type": str}
    """
    data = _get("contest.list")
    contests: Dict[int, dict] = {}
    for c in data["result"]:
        if c.get("phase") != "FINISHED":
            continue
        if "startTimeSeconds" not in c:
            continue
        date = datetime.utcfromtimestamp(c["startTimeSeconds"]).strftime("%Y-%m-%d")
        contests[c["id"]] = {
            "date": date,
            "name": c.get("name", ""),
            "type": c.get("type", "UNKNOWN"),
        }
    logging.info(f"fetch_contest_dates: {len(contests)} finished contests")
    return contests


def fetch_contest_submissions(contest_id: int,
                              problem_index: str,
                              max_fetch: int = 1000) -> List[dict]:
    """Fetch accepted submissions for a problem in any language.

    Reference solutions are used only for strategy extraction (the LLM reads
    them), so any language works. Python/PyPy solutions are preferred because
    the AST analyser can provide extra structural hints, but C++ / Java / etc.
    are fully supported via the LLM-only extraction path.

    Preference order: Python 3 > PyPy 3 > C++ > Java > other.
    Excludes Python 2 and PyPy 2 only.

    Args:
        contest_id: Codeforces contest ID.
        problem_index: Problem index letter, e.g. "E".
        max_fetch: How many submissions to pull from the API (default 1000).

    Returns:
        List of submission dicts sorted by preference then execution time:
        [{"submission_id": int, "author": str, "execution_time_ms": int,
          "memory_kb": int, "language": str}]
    """
    try:
        data = _get(
            "contest.status",
            params={"contestId": contest_id, "from": 1, "count": max_fetch},
        )
    except RuntimeError as e:
        logging.warning(f"contest.status failed for {contest_id}: {e}")
        return []

    all_subs = data["result"]
    for_problem = [s for s in all_subs if s.get("problem", {}).get("index") == problem_index]
    accepted    = [s for s in for_problem if s.get("verdict") == "OK"]
    langs_seen  = {s.get("programmingLanguage", "") for s in accepted}
    logging.debug(
        f"  contest.status {contest_id}{problem_index}: "
        f"{len(all_subs)} total fetched, {len(for_problem)} this problem, "
        f"{len(accepted)} accepted, langs={langs_seen}"
    )

    candidates = []
    for sub in accepted:
        lang = sub.get("programmingLanguage", "").lower()
        # Exclude Python 2 and PyPy 2 (outdated, incompatible syntax)
        if "python 2" in lang or "pypy 2" in lang:
            continue

        members = sub.get("author", {}).get("members", [])
        author = members[0].get("handle", "unknown") if members else "unknown"
        candidates.append({
            "submission_id":    sub["id"],
            "author":           author,
            "execution_time_ms": sub.get("timeConsumedMillis", 0),
            "memory_kb":        sub.get("memoryConsumedBytes", 0) // 1024,
            "language":         sub.get("programmingLanguage", ""),
        })

    def _lang_priority(lang: str) -> int:
        """Lower = preferred. Python first (AST support), then fast compiled langs."""
        l = lang.lower()
        if "python" in l and "pypy" not in l:  return 0  # CPython 3
        if "pypy" in l:                         return 1  # PyPy 3
        if "c++" in l:                          return 2  # C++
        if "java" in l:                         return 3  # Java
        return 4                                          # other

    candidates.sort(key=lambda x: (_lang_priority(x["language"]), x["execution_time_ms"]))

    logging.info(
        f"  Submissions for {contest_id}{problem_index}: "
        f"{len(candidates)} accepted (any lang) from {len(all_subs)} fetched | "
        f"langs: {langs_seen}"
    )
    return candidates
