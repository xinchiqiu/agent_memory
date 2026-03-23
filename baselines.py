"""Module 8: Baseline agents for comparison.

All baselines share the same Verifier and LLM client as the main agent.
They differ only in retrieval strategy or memory representation.

Baselines:
  1. NoMemoryBaseline       — free generation for every problem, no memory
  2. RandomRetrievalBaseline — random k entries from memory (same adaptation)
  3. FullHistoryBaseline     — dump raw text of k recent solved problems + code
  4. TagOracleBaseline       — oracle retrieval by ground-truth algorithm tags
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from data_structures import Memory, EpisodicEntry, Problem
from encoder import ProblemEncoder
from retriever import Retriever, RandomRetriever, TagOracleRetriever
from verifier import Verifier
from adaptation import free_generation, adapt_and_solve, solve_with_fallbacks
from memory_update import update_memory
from strategy_extraction import extract_strategy
from config import CONFIG


# ---------------------------------------------------------------------------
# Baseline 1: No memory
# ---------------------------------------------------------------------------

class NoMemoryBaseline:
    """Uses free generation for every problem. No retrieval, no memory."""

    def __init__(self, llm_client, verifier: Optional[Verifier] = None,
                 log_dir: str = "logs/baseline_no_memory"):
        self.llm_client = llm_client
        self.verifier = verifier or Verifier()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_log: List[Dict] = []

    def run(self, problems: List[Problem]) -> Dict:
        total = len(problems)
        successes = 0
        for i, problem in enumerate(problems):
            code, analysis, plan = free_generation(problem, self.llm_client)
            passed, error_info = self.verifier.verify(code, problem.test_cases)
            if passed:
                successes += 1
            self.results_log.append({
                "step": i + 1,
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty_rating,
                "ground_truth_tags": problem.algorithm_tags,
                "success": passed,
                "method": "free",
                "running_accuracy": successes / (i + 1),
            })
            logging.info(
                f"[NoMemory] {problem.problem_id}  "
                f"{'PASS' if passed else 'FAIL'}  "
                f"acc={successes/(i+1):.3f}"
            )

        final = {
            "baseline": "no_memory",
            "total_problems": total,
            "total_successes": successes,
            "overall_accuracy": successes / total if total else 0.0,
            "per_problem_log": self.results_log,
        }
        with open(self.log_dir / "final_results.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


# ---------------------------------------------------------------------------
# Baseline 2: Random retrieval
# ---------------------------------------------------------------------------

class RandomRetrievalBaseline:
    """Random k entries from memory; same two-step adaptation pipeline."""

    def __init__(self, llm_client, encoder: Optional[ProblemEncoder] = None,
                 verifier: Optional[Verifier] = None,
                 log_dir: str = "logs/baseline_random"):
        self.llm_client = llm_client
        self.encoder = encoder or ProblemEncoder()
        self.retriever = RandomRetriever(self.encoder)
        self.verifier = verifier or Verifier()
        self.memory = Memory()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_log: List[Dict] = []

    def seed_memory(self, seed_problems: List[Tuple[Problem, str]]) -> None:
        _seed_memory(self.memory, seed_problems, self.encoder, self.llm_client)

    def run(self, problems: List[Problem]) -> Dict:
        return _run_with_retriever(
            self, problems, baseline_name="random_retrieval"
        )


# ---------------------------------------------------------------------------
# Baseline 3: Full history (raw text + code in prompt)
# ---------------------------------------------------------------------------

FULL_HISTORY_PROMPT = """\
You are an expert competitive programmer.
Below are examples of problems you have solved before, including the full solution code.
Use these examples to help you solve the new problem.

## Previously Solved Problems

{history_section}

## New Problem to Solve
{new_problem_statement}

## Instructions
Write a complete Python program that reads from stdin and writes to stdout.
Think step by step, then output ONLY the code (no markdown fences).
"""


class FullHistoryBaseline:
    """Dumps raw text + code of k most recently solved problems into the prompt."""

    def __init__(self, llm_client, encoder: Optional[ProblemEncoder] = None,
                 verifier: Optional[Verifier] = None, top_k: int = None,
                 log_dir: str = "logs/baseline_full_history"):
        self.llm_client = llm_client
        self.encoder = encoder or ProblemEncoder()
        self.verifier = verifier or Verifier()
        self.top_k = top_k or CONFIG["top_k"]
        self.memory = Memory()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_log: List[Dict] = []

    def seed_memory(self, seed_problems: List[Tuple[Problem, str]]) -> None:
        _seed_memory(self.memory, seed_problems, self.encoder, self.llm_client)

    def run(self, problems: List[Problem]) -> Dict:
        total = len(problems)
        successes = 0
        for i, problem in enumerate(problems):
            # Use most recent k entries
            recent = sorted(
                self.memory.entries.values(),
                key=lambda e: e.created_at,
                reverse=True,
            )[:self.top_k]

            history_parts = []
            for j, entry in enumerate(recent):
                history_parts.append(
                    f"### Example {j+1}: {entry.problem_id} "
                    f"(rating {entry.difficulty_rating})\n"
                    f"**Problem:** {entry.problem_statement[:600]}...\n"
                    f"**Solution:**\n```python\n{entry.solution_code[:800]}\n```"
                )
            history_section = "\n\n".join(history_parts) if history_parts else "None yet."

            prompt = FULL_HISTORY_PROMPT.format(
                history_section=history_section,
                new_problem_statement=problem.statement,
            )
            response = self.llm_client.generate(prompt, temperature=0.0, max_tokens=2048)

            from adaptation import extract_python_code
            code = extract_python_code(response)
            passed, error_info = self.verifier.verify(code, problem.test_cases)

            if passed:
                successes += 1
                # Add to memory (simple embedding-only entry, no strategy extraction needed)
                embedding = self.encoder.encode(problem.statement)
                self.memory.current_timestep += 1
                entry = EpisodicEntry(
                    entry_id=f"fh_{self.memory.current_timestep}",
                    problem_id=problem.problem_id,
                    problem_statement=problem.statement,
                    strategy=None,  # type: ignore  # Not used in this baseline
                    solution_code=code,
                    difficulty_rating=problem.difficulty_rating,
                    verification_passed=True,
                    embedding=embedding,
                    created_at=self.memory.current_timestep,
                    source="solved",
                )
                self.memory.entries[entry.entry_id] = entry

            self.results_log.append({
                "step": i + 1,
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty_rating,
                "ground_truth_tags": problem.algorithm_tags,
                "success": passed,
                "method": "full_history",
                "running_accuracy": successes / (i + 1),
            })
            logging.info(
                f"[FullHistory] {problem.problem_id}  "
                f"{'PASS' if passed else 'FAIL'}  "
                f"acc={successes/(i+1):.3f}"
            )

        final = {
            "baseline": "full_history",
            "total_problems": total,
            "total_successes": successes,
            "overall_accuracy": successes / total if total else 0.0,
            "per_problem_log": self.results_log,
        }
        with open(self.log_dir / "final_results.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


# ---------------------------------------------------------------------------
# Baseline 4: Tag oracle
# ---------------------------------------------------------------------------

class TagOracleBaseline:
    """Uses ground-truth algorithm tags for retrieval (oracle; not available to real agent)."""

    def __init__(self, llm_client, encoder: Optional[ProblemEncoder] = None,
                 verifier: Optional[Verifier] = None,
                 log_dir: str = "logs/baseline_tag_oracle"):
        self.llm_client = llm_client
        self.encoder = encoder or ProblemEncoder()
        self.retriever = TagOracleRetriever(self.encoder)
        self.verifier = verifier or Verifier()
        self.memory = Memory()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_log: List[Dict] = []

    def seed_memory(self, seed_problems: List[Tuple[Problem, str]]) -> None:
        _seed_memory(self.memory, seed_problems, self.encoder, self.llm_client)

    def run(self, problems: List[Problem]) -> Dict:
        return _run_with_retriever(self, problems, baseline_name="tag_oracle")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _seed_memory(
    memory: Memory,
    seed_problems: List[Tuple[Problem, str]],
    encoder: ProblemEncoder,
    llm_client,
) -> None:
    logging.info(f"Seeding memory with {len(seed_problems)} problems…")
    for problem, code in seed_problems:
        strategy = extract_strategy(problem, code, llm_client)
        embedding = encoder.encode(problem.statement)
        memory.current_timestep += 1
        entry = EpisodicEntry(
            entry_id=f"seed_{memory.current_timestep}",
            problem_id=problem.problem_id,
            problem_statement=problem.statement,
            strategy=strategy,
            solution_code=code,
            difficulty_rating=problem.difficulty_rating,
            verification_passed=True,
            embedding=embedding,
            created_at=memory.current_timestep,
            source="seed",
        )
        memory.entries[entry.entry_id] = entry
    logging.info(f"Memory seeded with {len(memory.entries)} entries.")


def _run_with_retriever(baseline, problems: List[Problem], baseline_name: str) -> Dict:
    """Generic run loop for baselines that use a retriever + adaptation pipeline."""
    total = len(problems)
    successes = 0
    for i, problem in enumerate(problems):
        retrieved = baseline.retriever.retrieve(problem, baseline.memory)

        if retrieved:
            result = solve_with_fallbacks(
                problem, retrieved, baseline.llm_client, baseline.verifier
            )
        else:
            code, analysis, plan = free_generation(problem, baseline.llm_client)
            passed, error_info = baseline.verifier.verify(code, problem.test_cases)
            result = {
                "success": passed,
                "code": code,
                "method": "free" if passed else "failed",
                "alignment_analysis": analysis,
                "adapted_plan": plan,
                "attempts": [{
                    "method": "free",
                    "retrieved_entry_id": None,
                    "similarity_score": None,
                    "code": code,
                    "passed": passed,
                    "error_info": error_info,
                    "alignment_analysis": analysis,
                    "adapted_plan": plan,
                }],
            }

        baseline.memory = update_memory(
            baseline.memory, problem, result, baseline.encoder, baseline.llm_client
        )

        if result["success"]:
            successes += 1

        baseline.results_log.append({
            "step": i + 1,
            "problem_id": problem.problem_id,
            "difficulty": problem.difficulty_rating,
            "ground_truth_tags": problem.algorithm_tags,
            "success": result["success"],
            "method": result["method"],
            "memory_size": len(baseline.memory.entries),
            "running_accuracy": successes / (i + 1),
        })
        logging.info(
            f"[{baseline_name}] {problem.problem_id}  "
            f"{'PASS' if result['success'] else 'FAIL'}  "
            f"acc={successes/(i+1):.3f}"
        )

    final = {
        "baseline": baseline_name,
        "total_problems": total,
        "total_successes": successes,
        "overall_accuracy": successes / total if total else 0.0,
        "per_problem_log": baseline.results_log,
        "memory_size_final": len(baseline.memory.entries),
    }
    with open(baseline.log_dir / "final_results.json", "w") as f:
        json.dump(final, f, indent=2)
    return final
