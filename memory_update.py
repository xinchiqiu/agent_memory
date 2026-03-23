"""Module 6: Memory update — add successes and annotate failures."""

import logging
from typing import Dict

from data_structures import Memory, EpisodicEntry, FailureAnnotation
from data_structures import Problem
from strategy_extraction import extract_strategy
from encoder import ProblemEncoder


FAILURE_DIAGNOSIS_PROMPT = """\
You are analyzing why a solution strategy failed on a competitive programming problem.

## Problem
{problem_statement}

## Strategy That Was Tried
Technique Chain:
{technique_chain}

Key Insight: {key_insight}

## Failed Code
{failed_code}

## Error Information
{error_info}

## Task
In 2-3 sentences, explain WHY this strategy didn't work for this problem.
Focus on the structural mismatch between the strategy's assumptions and this
problem's actual requirements. What property of this problem violates the
strategy's preconditions?

Diagnosis:"""


def update_memory(
    memory: Memory,
    problem: Problem,
    result: Dict,
    encoder: ProblemEncoder,
    llm_client,
) -> Memory:
    """Update memory after a problem attempt.

    On success: extract strategy from the working code and add a new entry.
    On failure: generate a diagnosis for each failed adapted attempt and store
                a FailureAnnotation.

    Args:
        memory: Current memory state (mutated in-place and returned).
        problem: The problem that was attempted.
        result: Output dict from solve_with_fallbacks / the agent loop.
        encoder: For computing the new entry's embedding.
        llm_client: For strategy extraction and failure diagnosis.

    Returns:
        Updated memory.
    """
    memory.current_timestep += 1

    if result["success"]:
        _handle_success(memory, problem, result, encoder, llm_client)
    else:
        _handle_failure(memory, problem, result, llm_client)

    return memory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _handle_success(
    memory: Memory,
    problem: Problem,
    result: Dict,
    encoder: ProblemEncoder,
    llm_client,
) -> None:
    strategy = extract_strategy(problem, result["code"], llm_client)
    embedding = encoder.encode(problem.statement)

    entry = EpisodicEntry(
        entry_id=f"e_{memory.current_timestep}",
        problem_id=problem.problem_id,
        problem_statement=problem.statement,
        strategy=strategy,
        solution_code=result["code"],
        difficulty_rating=problem.difficulty_rating,
        verification_passed=True,
        embedding=embedding,
        created_at=memory.current_timestep,
        source="solved",
    )
    memory.entries[entry.entry_id] = entry
    logging.debug(f"  Memory updated: added entry {entry.entry_id} for {problem.problem_id}")

    # Update usage stats for retrieved entries that contributed
    for attempt in result.get("attempts", []):
        rid = attempt.get("retrieved_entry_id")
        if rid and rid in memory.entries:
            memory.entries[rid].times_retrieved += 1
            if attempt.get("passed"):
                memory.entries[rid].times_led_to_success += 1


def _handle_failure(
    memory: Memory,
    problem: Problem,
    result: Dict,
    llm_client,
) -> None:
    for attempt in result.get("attempts", []):
        if not attempt["method"].startswith("adapted_"):
            continue
        if attempt.get("passed"):
            continue

        rid = attempt.get("retrieved_entry_id")
        if not rid or rid not in memory.entries:
            continue

        retrieved_entry = memory.entries[rid]
        retrieved_entry.times_retrieved += 1
        retrieved_entry.times_led_to_failure += 1

        # Generate failure diagnosis
        tc_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(retrieved_entry.strategy.technique_chain))
        diagnosis_prompt = FAILURE_DIAGNOSIS_PROMPT.format(
            problem_statement=problem.statement,
            technique_chain=tc_str,
            key_insight=retrieved_entry.strategy.key_insight,
            failed_code=attempt["code"][:1000],
            error_info=attempt.get("error_info", "")[:500],
        )
        try:
            diagnosis = llm_client.generate(
                diagnosis_prompt,
                model=None,  # uses default
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as e:
            diagnosis = f"Diagnosis unavailable: {e}"

        annotation = FailureAnnotation(
            problem_id=problem.problem_id,
            problem_statement=problem.statement,
            retrieved_entry_id=rid,
            attempted_strategy=retrieved_entry.strategy,
            failure_code=attempt["code"],
            error_info=attempt.get("error_info", ""),
            diagnosis=diagnosis.strip(),
        )
        memory.failures.append(annotation)
        logging.debug(f"  FailureAnnotation stored for entry {rid} on problem {problem.problem_id}")
