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

# ---------------------------------------------------------------------------
# Granularity-specific prompt templates (Experiment 2: G1–G6)
# ---------------------------------------------------------------------------

# G2: Tag hints only — minimal information (~20 tokens per entry)
TAG_HINTS_ALIGNMENT_PROMPT = """\
You are an expert competitive programmer. You are given a new problem to solve, \
along with algorithm tag hints from similar problems you have solved before.

## New Problem to Solve
{new_problem_statement}

## Algorithm Tag Hints from Similar Solved Problems

{tag_hints_formatted}

## Your Task

Based on these tag hints, design a solution plan for the new problem.

### ANALYSIS
Consider which algorithm families from the hints might apply to this problem.

### ADAPTED PLAN
Write a technique chain as a numbered list of concrete algorithmic steps.
1. [First step]
2. [Second step]
...
"""

# G4: Strategy + code snippet (~500 tokens per entry)
STRATEGY_PLUS_SNIPPET_ALIGNMENT_PROMPT = """\
You are an expert competitive programmer. You are given a new problem to solve, \
along with strategies and partial code from similar problems you have solved before.

## New Problem to Solve
{new_problem_statement}

## Retrieved Strategies and Code Snippets from Similar Solved Problems

{strategies_with_snippets_formatted}

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

# G5: Full solution (~1000 tokens per entry)
FULL_SOLUTION_ALIGNMENT_PROMPT = """\
You are an expert competitive programmer. You are given a new problem to solve, \
along with full solutions from similar problems you have solved before.

## New Problem to Solve
{new_problem_statement}

## Full Solutions from Similar Solved Problems

{full_solutions_formatted}

## Your Task

Analyze the provided solutions and adapt the approach for the new problem.

### MATCH ANALYSIS
For each solution, assess how well its approach applies to the new problem.

### KEY DIFFERENCES
Identify key structural differences between the new problem and the solved ones.

### ADAPTATIONS NEEDED
Describe how to modify the best-matching approach.

### ADAPTED PLAN
Write the adapted technique chain as a numbered list of concrete algorithmic steps.
1. [First step]
2. [Second step]
...
"""

# G6: 3 full solutions (all retrieved, ~3000 tokens total)
MULTI_SOLUTION_CODE_PROMPT = """\
You are an expert competitive programmer.
Solve the following problem. Below are full solutions from similar problems for reference.

## Problem
{new_problem_statement}

## Reference Solutions from Similar Problems

{multi_solutions_formatted}

## Instructions
Study the reference solutions above to understand common patterns.
Then write a complete Python program for the new problem that reads from stdin
and writes to stdout.

### Thinking
[Think step by step here]

### Solution
[Python code only, no fences]
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

def format_tag_hints(retrieved: List[Tuple[EpisodicEntry, float]]) -> str:
    """G2: Format retrieved entries as tag hints only."""
    parts = []
    for i, (entry, score) in enumerate(retrieved):
        s = entry.strategy
        tags = ", ".join(s.algorithm_tags) if s.algorithm_tags else "unknown"
        parts.append(
            f"- Similar Problem {i+1} (similarity: {score:.3f}): "
            f"Tags: [{tags}]"
        )
    return "\n".join(parts)


def format_strategies_with_snippets(retrieved: List[Tuple[EpisodicEntry, float]]) -> str:
    """G4: Format retrieved entries with strategy + code snippet."""
    parts = []
    for i, (entry, score) in enumerate(retrieved):
        s = entry.strategy
        chain = "\n".join(f"  {j+1}. {step}" for j, step in enumerate(s.technique_chain))
        preconds = "\n".join(f"  - {p}" for p in s.preconditions)
        snippet = entry.solution_code[:400] + ("..." if len(entry.solution_code) > 400 else "")
        part = (
            f"### Strategy {i+1} (similarity: {score:.3f})\n"
            f"**Problem:** {entry.problem_statement[:500]}...\n"
            f"**Technique Chain:**\n{chain}\n"
            f"**Key Insight:** {s.key_insight}\n"
            f"**Preconditions:**\n{preconds}\n"
            f"**Algorithm Tags:** {', '.join(s.algorithm_tags)}\n"
            f"**Complexity:** {s.complexity or 'Not specified'}\n"
            f"**Code Snippet:**\n```python\n{snippet}\n```"
        )
        parts.append(part)
    return "\n\n".join(parts)


def format_full_solutions(retrieved: List[Tuple[EpisodicEntry, float]]) -> str:
    """G5: Format retrieved entries with full solution code."""
    parts = []
    for i, (entry, score) in enumerate(retrieved):
        code = entry.solution_code[:1000] + ("..." if len(entry.solution_code) > 1000 else "")
        part = (
            f"### Solution {i+1} (similarity: {score:.3f})\n"
            f"**Problem:** {entry.problem_statement[:500]}...\n"
            f"**Difficulty:** {entry.difficulty_rating}\n"
            f"**Solution Code:**\n```python\n{code}\n```"
        )
        parts.append(part)
    return "\n\n".join(parts)


def format_multi_solutions(retrieved: List[Tuple[EpisodicEntry, float]]) -> str:
    """G6: Format all retrieved entries with full solutions (no alignment step)."""
    parts = []
    for i, (entry, score) in enumerate(retrieved):
        code = entry.solution_code[:1000] + ("..." if len(entry.solution_code) > 1000 else "")
        part = (
            f"### Reference {i+1}: {entry.problem_id} "
            f"(similarity: {score:.3f}, rating: {entry.difficulty_rating})\n"
            f"**Problem:** {entry.problem_statement[:400]}...\n"
            f"```python\n{code}\n```"
        )
        parts.append(part)
    return "\n\n".join(parts)


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

def adapt_and_solve_granularity(
    new_problem: Problem,
    retrieved: List[Tuple[EpisodicEntry, float]],
    llm_client,
    granularity_mode: str = "G3",
    alignment_model: Optional[str] = None,
    generation_model: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Granularity-aware adaptation: dispatches to the right prompt based on mode.

    G1: free generation (no retrieval info used)
    G2: tag hints → alignment → code generation
    G3: strategy only → alignment → code generation (DEFAULT, same as adapt_and_solve)
    G4: strategy + code snippet → alignment → code generation
    G5: full solution → alignment → code generation
    G6: 3 full solutions → single-step code generation (no separate alignment)

    Returns:
        (generated_code, alignment_analysis, adapted_plan)
    """
    if granularity_mode == "G1":
        return free_generation(new_problem, llm_client)

    if granularity_mode == "G6":
        # Single-step: inject all solutions directly into a code-gen prompt
        multi_fmt = format_multi_solutions(retrieved)
        prompt = MULTI_SOLUTION_CODE_PROMPT.format(
            new_problem_statement=new_problem.statement,
            multi_solutions_formatted=multi_fmt,
        )
        _gen_model = generation_model or CONFIG.get("generation_model")
        if _gen_model and ":generate" not in _gen_model:
            _gen_model = f"{_gen_model}:generate"
        response = llm_client.generate(prompt, model=_gen_model, temperature=0.0, max_tokens=2048)
        code = extract_python_code(response)
        return code, response, ""

    # G2–G5: two-step (alignment → code generation) with different info levels
    if granularity_mode == "G2":
        formatted = format_tag_hints(retrieved)
        alignment_prompt = TAG_HINTS_ALIGNMENT_PROMPT.format(
            new_problem_statement=new_problem.statement,
            tag_hints_formatted=formatted,
        )
    elif granularity_mode == "G4":
        formatted = format_strategies_with_snippets(retrieved)
        alignment_prompt = STRATEGY_PLUS_SNIPPET_ALIGNMENT_PROMPT.format(
            new_problem_statement=new_problem.statement,
            strategies_with_snippets_formatted=formatted,
        )
    elif granularity_mode == "G5":
        formatted = format_full_solutions(retrieved)
        alignment_prompt = FULL_SOLUTION_ALIGNMENT_PROMPT.format(
            new_problem_statement=new_problem.statement,
            full_solutions_formatted=formatted,
        )
    else:  # G3 (default)
        formatted = format_retrieved_strategies(retrieved)
        alignment_prompt = ALIGNMENT_PROMPT.format(
            new_problem_statement=new_problem.statement,
            retrieved_strategies_formatted=formatted,
        )

    # Step 1: Alignment
    _align_model = alignment_model or CONFIG.get("alignment_model")
    if _align_model and ":align" not in _align_model:
        _align_model = f"{_align_model}:align"
    alignment_response = llm_client.generate(
        alignment_prompt, model=_align_model, temperature=0.3, max_tokens=2048,
    )
    adapted_plan = extract_adapted_plan(alignment_response)

    best_strategy = retrieved[0][0].strategy if retrieved else None
    best_insight = best_strategy.key_insight if best_strategy else "No prior strategy available."
    complexity_target = best_strategy.complexity if best_strategy else "As efficient as possible"

    # Step 2: Guided Code Generation
    code_prompt = CODE_GENERATION_PROMPT.format(
        new_problem_statement=new_problem.statement,
        adapted_plan=adapted_plan,
        key_insight=best_insight,
        complexity_target=complexity_target,
    )
    _gen_model = generation_model or CONFIG.get("generation_model")
    if _gen_model and ":generate" not in _gen_model:
        _gen_model = f"{_gen_model}:generate"
    code_response = llm_client.generate(
        code_prompt, model=_gen_model, temperature=0.0, max_tokens=2048,
    )
    generated_code = extract_python_code(code_response)
    return generated_code, alignment_response, adapted_plan


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
    granularity_mode: Optional[str] = None,
) -> Dict:
    """Attempt to solve the problem, trying each retrieved strategy then free generation.

    Fallback order:
      1. Adapt strategy from best retrieved entry
      2. Adapt strategy from second-best retrieved entry
      3. Adapt strategy from third-best retrieved entry
      4. Free generation (no strategy)
      5. Give up

    Args:
        granularity_mode: If set, uses granularity-aware adaptation (G1-G6).
            If None, uses the original adapt_and_solve (same as G3).

    Returns a dict:
      success, code, method, alignment_analysis, adapted_plan, attempts
    """
    attempts = []
    _mode = granularity_mode or CONFIG.get("granularity_mode")

    # G1 and G6 don't do per-entry fallback
    if _mode == "G1":
        code, analysis, plan = free_generation(new_problem, llm_client)
        passed, error_info = verifier.verify(code, new_problem.test_cases)
        attempts.append({
            "method": "free", "retrieved_entry_id": None,
            "similarity_score": None, "code": code, "passed": passed,
            "error_info": error_info, "alignment_analysis": analysis,
            "adapted_plan": plan,
        })
        return {
            "success": passed, "code": code,
            "method": "free" if passed else "failed",
            "alignment_analysis": analysis, "adapted_plan": plan,
            "attempts": attempts,
        }

    if _mode == "G6":
        # G6 uses all retrieved entries at once, no per-entry fallback
        code, analysis, plan = adapt_and_solve_granularity(
            new_problem, retrieved, llm_client, granularity_mode="G6",
        )
        passed, error_info = verifier.verify(code, new_problem.test_cases)
        attempts.append({
            "method": "adapted_all", "retrieved_entry_id": None,
            "similarity_score": None, "code": code, "passed": passed,
            "error_info": error_info, "alignment_analysis": analysis,
            "adapted_plan": plan,
        })
        if passed:
            return {
                "success": True, "code": code, "method": "adapted_all",
                "alignment_analysis": analysis, "adapted_plan": plan,
                "attempts": attempts,
            }
        # Fallback to free generation
        code, analysis, plan = free_generation(new_problem, llm_client)
        passed, error_info = verifier.verify(code, new_problem.test_cases)
        attempts.append({
            "method": "free", "retrieved_entry_id": None,
            "similarity_score": None, "code": code, "passed": passed,
            "error_info": error_info, "alignment_analysis": analysis,
            "adapted_plan": plan,
        })
        return {
            "success": passed, "code": code,
            "method": "free" if passed else "failed",
            "alignment_analysis": analysis, "adapted_plan": plan,
            "attempts": attempts,
        }

    # G2–G5: per-entry fallback chain (same structure as original)
    for i, (entry, score) in enumerate(retrieved):
        logging.debug(f"  Trying adapted strategy {i+1} (entry {entry.entry_id}, sim={score:.3f})")
        if _mode and _mode != "G3":
            code, analysis, plan = adapt_and_solve_granularity(
                new_problem, [(entry, score)], llm_client, granularity_mode=_mode,
            )
        else:
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
