"""Codeforces web scraper.

Scrapes content that is not available via the API:
  - Problem statements (HTML → cleaned plaintext)
  - Sample test cases
  - Accepted Python submission source code
  - Contest editorials
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag

CF_BASE = "https://codeforces.com"
_DEFAULT_DELAY = 1.5   # seconds between requests


def _make_session() -> requests.Session:
    """Return a requests Session with browser-like headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
            "(research project — competitive programming NLP)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


_SESSION = _make_session()


def _get_html(url: str, retries: int = 4, delay: float = _DEFAULT_DELAY) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, with retry/backoff."""
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            wait = delay * (2 ** attempt)
            logging.warning(f"GET {url} failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    logging.error(f"Giving up on {url} after {retries} attempts")
    return None


# ---------------------------------------------------------------------------
# MathJax / LaTeX cleanup
# ---------------------------------------------------------------------------

def _math_to_text(text: str) -> str:
    """Convert common LaTeX math constructs to readable ASCII approximations.

    Handles the most frequent patterns in Codeforces problem statements.
    Not a full LaTeX parser — just heuristic cleanup for readability.
    """
    replacements = [
        # Greek letters
        (r"\\alpha", "alpha"), (r"\\beta", "beta"), (r"\\gamma", "gamma"),
        (r"\\delta", "delta"), (r"\\epsilon", "epsilon"), (r"\\lambda", "lambda"),
        (r"\\mu", "mu"), (r"\\pi", "pi"), (r"\\sigma", "sigma"),
        # Operators and relations
        (r"\\leq", "<="), (r"\\geq", ">="), (r"\\neq", "!="),
        (r"\\cdot", "*"), (r"\\times", "*"), (r"\\div", "/"),
        (r"\\infty", "inf"), (r"\\ldots", "..."), (r"\\dots", "..."),
        # Common constructs
        (r"\\lfloor", "floor("), (r"\\rfloor", ")"),
        (r"\\lceil", "ceil("),  (r"\\rceil", ")"),
        (r"\\left\|", "|"), (r"\\right\|", "|"),
        (r"\\left", ""), (r"\\right", ""),
        # Superscript/subscript: a^{b} -> a^b, a_{b} -> a_b
        (r"\^\{([^}]+)\}", r"^\1"),
        (r"_\{([^}]+)\}", r"_\1"),
        # sum, prod
        (r"\\sum_?\{?[^}]*\}?\^?\{?[^}]*\}?", "sum"),
        (r"\\prod_?\{?[^}]*\}?\^?\{?[^}]*\}?", "prod"),
        # Remove remaining braces
        (r"\{", ""), (r"\}", ""),
        # Collapse whitespace
        (r"\s+", " "),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text.strip()


def _element_to_text(element: Tag) -> str:
    """Recursively convert a BeautifulSoup element to clean plaintext.

    - Strips MathJax/LaTeX spans
    - Converts <br> and <p> to newlines
    - Preserves list structure with dashes
    """
    if element is None:
        return ""

    parts = []
    for child in element.descendants:
        if not hasattr(child, "name"):
            # NavigableString (raw text)
            text = str(child)
            # Check if parent is a MathJax span — apply math cleanup
            parent = child.parent
            if parent and parent.name == "span" and "MathJax" in parent.get("class", [""]):
                continue  # the rendered MathJax text is redundant; we use the source
            parts.append(text)
        elif child.name == "br":
            parts.append("\n")
        elif child.name in ("p", "div"):
            parts.append("\n")

    raw = "".join(parts)

    # Extract inline LaTeX from $...$ patterns that appear in the raw HTML
    # (Codeforces embeds LaTeX in the page source)
    def replace_dollar_math(m: re.Match) -> str:
        return _math_to_text(m.group(1))

    raw = re.sub(r"\$\$(.+?)\$\$", replace_dollar_math, raw, flags=re.DOTALL)
    raw = re.sub(r"\$(.+?)\$", replace_dollar_math, raw)

    # Collapse multiple blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Problem statement scraping
# ---------------------------------------------------------------------------

def scrape_problem(contest_id: int, index: str,
                   delay: float = _DEFAULT_DELAY) -> Optional[dict]:
    """Scrape a single problem's full statement and sample tests.

    Args:
        contest_id: Codeforces contest ID.
        index: Problem index (e.g. "E").
        delay: Seconds to sleep after the request.

    Returns:
        dict with keys: statement, input_spec, output_spec, note,
                        sample_tests, time_limit, memory_limit
        Returns None if the page cannot be parsed.
    """
    url = f"{CF_BASE}/problemset/problem/{contest_id}/{index}"
    soup = _get_html(url)
    time.sleep(delay)

    if soup is None:
        return None

    problem_div = soup.find("div", class_="problem-statement")
    if not problem_div:
        logging.warning(f"No problem-statement div for {contest_id}{index}")
        return None

    result: dict = {
        "statement": "",
        "input_spec": "",
        "output_spec": "",
        "note": "",
        "sample_tests": [],
        "time_limit": "",
        "memory_limit": "",
    }

    # Limits (inside .header div)
    header = problem_div.find("div", class_="header")
    if header:
        tl = header.find("div", class_="time-limit")
        ml = header.find("div", class_="memory-limit")
        if tl:
            result["time_limit"] = tl.get_text(strip=True).replace("time limit per test", "").strip()
        if ml:
            result["memory_limit"] = ml.get_text(strip=True).replace("memory limit per test", "").strip()

    # Walk top-level divs inside problem-statement
    statement_parts = []
    for div in problem_div.find_all("div", recursive=False):
        classes = div.get("class", [])
        if "header" in classes:
            continue
        elif "input-specification" in classes:
            result["input_spec"] = _element_to_text(div)
        elif "output-specification" in classes:
            result["output_spec"] = _element_to_text(div)
        elif "sample-tests" in classes:
            result["sample_tests"] = _extract_sample_tests(div)
        elif "note" in classes:
            result["note"] = _element_to_text(div)
        else:
            part = _element_to_text(div)
            if part:
                statement_parts.append(part)

    result["statement"] = "\n\n".join(statement_parts)
    return result


def _extract_sample_tests(sample_div: Tag) -> List[dict]:
    """Extract (input, output) pairs from the sample-tests div."""
    tests = []
    inputs  = sample_div.find_all("div", class_="input")
    outputs = sample_div.find_all("div", class_="output")
    for inp, out in zip(inputs, outputs):
        inp_pre = inp.find("pre")
        out_pre = out.find("pre")
        if inp_pre and out_pre:
            # Preserve newlines inside <pre> by using get_text with separator
            tests.append({
                "input":  inp_pre.get_text("\n"),
                "output": out_pre.get_text("\n"),
            })
    return tests


# ---------------------------------------------------------------------------
# Submission source code scraping
# ---------------------------------------------------------------------------

def scrape_submission_code(contest_id: int, submission_id: int,
                           delay: float = _DEFAULT_DELAY) -> Optional[str]:
    """Scrape the source code of a specific submission.

    Returns the source code string, or None on failure.
    """
    url = f"{CF_BASE}/contest/{contest_id}/submission/{submission_id}"
    soup = _get_html(url)
    time.sleep(delay)

    if soup is None:
        return None

    code_pre = soup.find("pre", id="program-source-text")
    if code_pre:
        return code_pre.get_text()

    # Fallback: look inside the source viewer div
    source_div = soup.find("div", id="program-source")
    if source_div:
        pre = source_div.find("pre")
        if pre:
            return pre.get_text()

    logging.warning(f"Could not find source code for submission {submission_id}")
    return None


def scrape_accepted_solutions(contest_id: int,
                               index: str,
                               submission_candidates: List[dict],
                               max_solutions: int = 3,
                               min_code_len: int = 100,
                               max_code_len: int = 5000,
                               delay: float = _DEFAULT_DELAY) -> List[dict]:
    """Scrape source code for a list of candidate submissions.

    Args:
        contest_id: Contest ID (used to build the submission URL).
        index: Problem index (for logging only).
        submission_candidates: Output of cf_api.fetch_contest_submissions().
        max_solutions: Stop after collecting this many solutions.
        min_code_len / max_code_len: Skip solutions outside this length range.
        delay: Seconds to sleep between requests.

    Returns:
        List of dicts: {submission_id, author, execution_time_ms, memory_kb,
                        language, code}
    """
    solutions = []
    for cand in submission_candidates:
        if len(solutions) >= max_solutions:
            break
        code = scrape_submission_code(contest_id, cand["submission_id"], delay=delay)
        if code is None:
            continue
        code = code.strip()
        if not (min_code_len <= len(code) <= max_code_len):
            continue
        solutions.append({**cand, "code": code})
    return solutions


# ---------------------------------------------------------------------------
# Editorial scraping
# ---------------------------------------------------------------------------

def scrape_editorial(contest_id: int,
                     delay: float = _DEFAULT_DELAY) -> Dict[str, str]:
    """Try to find and scrape the editorial for a contest.

    Strategy:
      1. Load the contest page sidebar for announcement/blog links.
      2. Find a link whose text contains "editorial" or "tutorial".
      3. Scrape that blog post and split text by problem letter headings.

    Returns:
        Dict: problem_index (e.g. "E") -> editorial_text.
        Empty dict if no editorial found.
    """
    url = f"{CF_BASE}/contest/{contest_id}"
    soup = _get_html(url)
    time.sleep(delay)

    if soup is None:
        return {}

    # Look for editorial link in sidebar / announcements
    editorial_url = None
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        if "editorial" in text or "tutorial" in text:
            href = a["href"]
            if href.startswith("/blog/entry/"):
                editorial_url = CF_BASE + href
                break

    if not editorial_url:
        return {}

    blog_soup = _get_html(editorial_url)
    time.sleep(delay)
    if blog_soup is None:
        return {}

    blog_content = blog_soup.find("div", class_="content")
    if not blog_content:
        return {}

    full_text = _element_to_text(blog_content)

    # Split by "Problem X" / "### X" headings
    result: Dict[str, str] = {}
    current_index = None
    current_lines: List[str] = []

    for line in full_text.split("\n"):
        # Match headings like "Problem E", "E.", "### E"
        m = re.match(r"^(?:problem\s+)?([A-Z])\b\.?", line.strip(), re.IGNORECASE)
        if m:
            if current_index and current_lines:
                result[current_index.upper()] = "\n".join(current_lines).strip()
            current_index = m.group(1).upper()
            current_lines = [line]
        elif current_index:
            current_lines.append(line)

    if current_index and current_lines:
        result[current_index.upper()] = "\n".join(current_lines).strip()

    return result
