"""Module 7: Main agent loop — orchestrates the full pipeline."""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from data_structures import Memory, EpisodicEntry, Problem
from encoder import ProblemEncoder
from retriever import Retriever
from verifier import Verifier
from adaptation import (
    adapt_and_solve, free_generation, solve_with_fallbacks,
    adapt_and_solve_granularity,
)
from memory_update import update_memory
from strategy_extraction import extract_strategy
from config import CONFIG


class StrategyAdaptationAgent:
    """Processes competitive programming problems sequentially with memory."""

    def __init__(self,
                 llm_client,
                 encoder: Optional[ProblemEncoder] = None,
                 retriever: Optional[Retriever] = None,
                 verifier: Optional[Verifier] = None,
                 log_dir: str = None,
                 granularity_mode: Optional[str] = None):
        self.llm_client = llm_client
        self.encoder = encoder or ProblemEncoder()
        self.retriever = retriever or Retriever(self.encoder)
        self.verifier = verifier or Verifier()
        self.memory = Memory()
        self.granularity_mode = granularity_mode or CONFIG.get("granularity_mode", "G3")
        self.log_dir = Path(log_dir or CONFIG["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_log: List[Dict] = []

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "run.log"),
                logging.StreamHandler(),
            ],
        )

    # ------------------------------------------------------------------
    # Memory seeding
    # ------------------------------------------------------------------

    def seed_memory(self, seed_problems: List[Tuple[Problem, str]]) -> None:
        """Initialize memory with pre-solved problems.

        Args:
            seed_problems: List of (problem, solution_code) pairs.
        """
        logging.info(f"Seeding memory with {len(seed_problems)} problems…")
        for item in seed_problems:
            # Accept (problem, code) or (problem, code, language)
            if len(item) == 3:
                problem, code, language = item
            else:
                problem, code = item
                language = ""
            strategy = extract_strategy(problem, code, self.llm_client,
                                        solution_language=language)
            embedding = self.encoder.encode(problem.statement)
            self.memory.current_timestep += 1
            entry = EpisodicEntry(
                entry_id=f"seed_{self.memory.current_timestep}",
                problem_id=problem.problem_id,
                problem_statement=problem.statement,
                strategy=strategy,
                solution_code=code,
                difficulty_rating=problem.difficulty_rating,
                verification_passed=True,
                embedding=embedding,
                created_at=self.memory.current_timestep,
                source="seed",
            )
            self.memory.entries[entry.entry_id] = entry
        logging.info(f"Memory seeded. Total entries: {len(self.memory.entries)}")

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(self, problems: List[Problem]) -> Dict:
        """Process a sequence of problems with the current memory.

        Problems should be in temporal/contest order to simulate realistic
        sequential problem-solving.

        Returns:
            Dict with aggregate results and per-problem logs.
        """
        total = len(problems)
        successes = 0
        checkpoint_interval = CONFIG.get("checkpoint_interval", 50)

        for i, problem in enumerate(problems):
            sep = "=" * 60
            logging.info(f"\n{sep}")
            logging.info(
                f"Problem {i+1}/{total}: {problem.problem_id}  "
                f"rating={problem.difficulty_rating}  "
                f"memory_size={len(self.memory.entries)}"
            )

            # --- RETRIEVE ---
            retrieved = self.retriever.retrieve(problem, self.memory)
            logging.info(f"Retrieved {len(retrieved)} entries:")
            for entry, score in retrieved:
                logging.info(
                    f"  {entry.problem_id}  sim={score:.3f}  "
                    f"tags={entry.strategy.algorithm_tags}"
                )

            # --- SOLVE ---
            if retrieved:
                result = solve_with_fallbacks(
                    problem, retrieved, self.llm_client, self.verifier,
                    granularity_mode=self.granularity_mode,
                )
            else:
                code, analysis, plan = free_generation(problem, self.llm_client)
                passed, error_info = self.verifier.verify(code, problem.test_cases)
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

            # --- UPDATE MEMORY ---
            self.memory = update_memory(
                self.memory, problem, result, self.encoder, self.llm_client
            )

            # --- LOG ---
            if result["success"]:
                successes += 1

            log_entry = {
                "step": i + 1,
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty_rating,
                "ground_truth_tags": problem.algorithm_tags,
                "success": result["success"],
                "method": result["method"],
                "memory_size": len(self.memory.entries),
                "num_retrieved": len(retrieved),
                "retrieved_ids": [e.entry_id for e, _ in retrieved],
                "retrieved_scores": [float(s) for _, s in retrieved],
                "num_attempts": len(result["attempts"]),
                "running_accuracy": successes / (i + 1),
            }
            self.results_log.append(log_entry)

            logging.info(
                f"Result: {'PASS' if result['success'] else 'FAIL'}  "
                f"method={result['method']}  "
                f"running_acc={successes/(i+1):.3f}"
            )

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(i + 1)

        final = {
            "total_problems": total,
            "total_successes": successes,
            "overall_accuracy": successes / total if total else 0.0,
            "per_problem_log": self.results_log,
            "memory_size_final": len(self.memory.entries),
            "failure_annotations": len(self.memory.failures),
        }
        self._save_results(final)
        return final

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        """Save results log and memory snapshot."""
        log_path = self.log_dir / f"results_log_step_{step}.json"
        with open(log_path, "w") as f:
            json.dump(self.results_log, f, indent=2)

        mem_path = self.log_dir / f"memory_step_{step}.pkl"
        self._save_memory(mem_path)
        logging.info(f"Checkpoint saved at step {step}")

    def _save_results(self, results: Dict) -> None:
        with open(self.log_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        self._save_memory(self.log_dir / "memory_final.pkl")
        logging.info("Final results saved.")

    def _save_memory(self, path: Path) -> None:
        """Pickle the memory (embeddings included as numpy arrays)."""
        with open(path, "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path: str) -> None:
        """Load a previously saved memory state."""
        with open(path, "rb") as f:
            self.memory = pickle.load(f)
        logging.info(f"Loaded memory with {len(self.memory.entries)} entries from {path}")
