"""Module 4: Adaptation module — two-step strategy alignment + code generation.

Also contains free-generation fallback and solve_with_fallbacks orchestrator.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

from data_structures import Problem, EpisodicEntry
from config import CONFIG


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ALIGNMENT_PROMPT = """\
You are an expert competitive programmer. You are given a new problem to solve, \
along with strategies from similar problems you have solved before.

## New Problem to Solve
{new_problem_statement}

## Retrieved Strategies from Similar Solved Problems

{retrieved_strategies_formatted}

## Your Task

Analyze which retrieved strategy (or combination) best applies to the new problem.

### MATCH ANALYSIS
For each retrieved strategy, assess how well its preconditions match the new problem.
Quote specific elements that match or don't match.

### KEY DIFFERENCES
Identify key structural differences between the new problem and the retrieved problems.

### ADAPTATIONS NEEDED
Describe how to modify the best-matching strategy. Specify which steps carry over,
which need modification, and any new steps needed.

### ADAPTED PLAN
Write the adapted technique chain as a numbered list of concrete algorithmic steps.
Each step should be specific enough to implement directly.
1. [First step]
2. [Second step]
...
"""

CODE_GENERATION_PROMPT = """\
You are an expert competitive programmer.
Solve the following problem by implementing the plan below.

## Problem
{new_problem_statement}

## Solution Plan (follow this plan)
{adapted_plan}

## Key Insight to Remember
{key_insight}

## Instructions
- Implement the solution in Python.
- Read input from stdin, write output to stdout.
- Follow the plan above step by step.
- Handle edge cases (empty input, n=1, large values).
- Complexity target: {complexity_target}
- Write a complete, runnable Python program.
- Output ONLY the code — no explanation, no markdown fences.
"""

FREE_GENERATION_PROMPT = """\
You are an expert competitive programmer.
Solve the following problem.

## Problem
{problem_statement}

## Instructions
Think through the problem step by step:
1. Understand what is being asked.
2. Identify the key algorithmic technique needed.
3. Design your approach.
4. Consider edge cases.
5. Implement the solution.

Write a complete Python program that reads from stdin and writes to stdout.

### Thinking
[Think step by step here]

### Solution
[Python code only, no fences]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_retrieved_strategies(retrieved: List[Tuple[EpisodicEntry, float]]) -> str:
    """Format retrieved entries for the alignment prompt.

    IMPORTANT: includes strategy only — NOT the solution code.
    """
    parts = []
    for i, (entry, score) in enumerate(retrieved):
        s = entry.strategy
        chain = "\n".join(f"  {j+1}. {step}" for j, step in enumerate(s.technique_chain))
        preconds = "\n".join(f"  - {p}" for p in s.preconditions)
        part = (
            f"### Strategy {i+1} (similarity: {score:.3f})\n"
            f"**Problem:** {entry.problem_statement[:500]}...\n"
            f"**Difficulty:** {entry.difficulty_rating}\n"
            f"**Technique Chain:**\n{chain}\n"
            f"**Key Insight:** {s.key_insight}\n"
            f"**Preconditions:**\n{preconds}\n"
            f"**Algorithm Tags:** {', '.join(s.algorithm_tags)}\n"
            f"**Complexity:** {s.complexity or 'Not specified'}"
        )
        parts.append(part)
    return "\n\n".join(parts)


def extract_adapted_plan(alignment_response: str) -> str:
    """Extract the ADAPTED PLAN section from the alignment response."""
    match = re.search(
        r'###\s*ADAPTED PLAN\s*\n(.*?)(?:###|\Z)',
        alignment_response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    # Fallback: return everything after "ADAPTED PLAN"
    idx = alignment_response.lower().find("adapted plan")
    if idx != -1:
        return alignment_response[idx + len("adapted plan"):].strip()
    return alignment_response.strip()


def extract_python_code(response: str) -> str:
    """Strip markdown fences and extract Python code from a response."""
    # Try ```python ... ``` first
    match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ### Solution section
    match = re.search(r'###\s*Solution\s*\n(.*?)(?:###|\Z)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return the full response stripped
    return response.strip()


def extract_thinking(response: str) -> str:
    """Extract the ### Thinking section from a free-generation response."""
    match = re.search(
        r'###\s*Thinking\s*\n(.*?)(?:###|\Z)',
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Core adaptation function
# ---------------------------------------------------------------------------

def adapt_and_solve(
    new_problem: Problem,
    retrieved: List[Tuple[EpisodicEntry, float]],
    llm_client,
    alignment_model: Optional[str] = None,
    generation_model: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Two-step adaptation: alignment then code generation.

    Args:
        new_problem: Problem to solve.
        retrieved: List of (entry, score) from the retriever.
        llm_client: LLM client.
        alignment_model: Model for alignment step (defaults to config).
        generation_model: Model for code generation step (defaults to config).

    Returns:
        (generated_code, alignment_analysis, adapted_plan)
    """
    # --- Step 1: Strategy Alignment ---
    retrieved_fmt = format_retrieved_strategies(retrieved)
    alignment_prompt = ALIGNMENT_PROMPT.format(
        new_problem_statement=new_problem.statement,
        retrieved_strategies_formatted=retrieved_fmt,
    )
    # Append ":align" role suffix so RoleAwareLLMClient can enable thinking mode
    _align_model = alignment_model or CONFIG.get("alignment_model")
    if _align_model and ":align" not in _align_model:
        _align_model = f"{_align_model}:align"
    alignment_response = llm_client.generate(
        alignment_prompt,
        model=_align_model,
        temperature=0.3,
        max_tokens=2048,
    )
    adapted_plan = extract_adapted_plan(alignment_response)

    best_strategy = retrieved[0][0].strategy if retrieved else None
    best_insight = best_strategy.key_insight if best_strategy else "No prior strategy available."
    complexity_target = best_strategy.complexity if best_strategy else "As efficient as possible"

    # --- Step 2: Guided Code Generation ---
    code_prompt = CODE_GENERATION_PROMPT.format(
        new_problem_statement=new_problem.statement,
        adapted_plan=adapted_plan,
        key_insight=best_insight,
        complexity_target=complexity_target,
    )
    # Append ":generate" role suffix so RoleAwareLLMClient disables thinking mode
    _gen_model = generation_model or CONFIG.get("generation_model")
    if _gen_model and ":generate" not in _gen_model:
        _gen_model = f"{_gen_model}:generate"
    code_response = llm_client.generate(
        code_prompt,
        model=_gen_model,
        temperature=0.0,
        max_tokens=2048,
    )
    generated_code = extract_python_code(code_response)
    return generated_code, alignment_response, adapted_plan


# ---------------------------------------------------------------------------
# Free generation (no strategy)
# ---------------------------------------------------------------------------

def free_generation(problem: Problem, llm_client) -> Tuple[str, str, str]:
    """Generate a solution without any retrieved strategy.

    Used as a fallback or as the no-memory baseline.

    Returns:
        (generated_code, full_response, thinking_section)
    """
    prompt = FREE_GENERATION_PROMPT.format(problem_statement=problem.statement)
    response = llm_client.generate(
        prompt,
        model=CONFIG.get("generation_model"),
        temperature=0.0,
        max_tokens=2048,
    )
    code = extract_python_code(response)
    thinking = extract_thinking(response)
    return code, response, thinking


# ---------------------------------------------------------------------------
# Solve with fallbacks
# ---------------------------------------------------------------------------

def solve_with_fallbacks(
    new_problem: Problem,
    retrieved: List[Tuple[EpisodicEntry, float]],
    llm_client,
    verifier,
) -> Dict:
    """Attempt to solve the problem, trying each retrieved strategy then free generation.

    Fallback order:
      1. Adapt strategy from best retrieved entry
      2. Adapt strategy from second-best retrieved entry
      3. Adapt strategy from third-best retrieved entry
      4. Free generation (no strategy)
      5. Give up

    Returns a dict:
      success, code, method, alignment_analysis, adapted_plan, attempts
    """
    attempts = []

    # Try each retrieved strategy individually
    for i, (entry, score) in enumerate(retrieved):
        logging.debug(f"  Trying adapted strategy {i+1} (entry {entry.entry_id}, sim={score:.3f})")
        code, analysis, plan = adapt_and_solve(
            new_problem, [(entry, score)], llm_client
        )
        passed, error_info = verifier.verify(code, new_problem.test_cases)
        attempt = {
            "method": f"adapted_{i+1}",
            "retrieved_entry_id": entry.entry_id,
            "similarity_score": float(score),
            "code": code,
            "passed": passed,
            "error_info": error_info,
            "alignment_analysis": analysis,
            "adapted_plan": plan,
        }
        attempts.append(attempt)
        if passed:
            return {
                "success": True,
                "code": code,
                "method": f"adapted_{i+1}",
                "alignment_analysis": analysis,
                "adapted_plan": plan,
                "attempts": attempts,
            }

    # Fallback: free generation
    logging.debug("  All adapted strategies failed. Trying free generation.")
    code, analysis, plan = free_generation(new_problem, llm_client)
    passed, error_info = verifier.verify(code, new_problem.test_cases)
    attempts.append({
        "method": "free",
        "retrieved_entry_id": None,
        "similarity_score": None,
        "code": code,
        "passed": passed,
        "error_info": error_info,
        "alignment_analysis": analysis,
        "adapted_plan": plan,
    })

    return {
        "success": passed,
        "code": code,
        "method": "free" if passed else "failed",
        "alignment_analysis": analysis,
        "adapted_plan": plan,
        "attempts": attempts,
    }
