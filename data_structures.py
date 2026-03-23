"""Module 1: Core data structures for the strategy-only adaptation agent."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Strategy:
    """The strategy extracted from a solved problem.

    Stored in memory and retrieved for new problems.
    Does NOT contain solution code — only transferable reasoning.
    """

    technique_chain: List[str]
    # Ordered list of algorithmic steps at medium abstraction level.
    # GOOD: ["sort intervals by end time",
    #         "greedy selection: pick earliest-ending non-overlapping interval",
    #         "track last selected endpoint"]

    key_insight: str
    # The single most important conceptual leap that makes the problem solvable.
    # 1-3 sentences explaining WHY the approach works.

    preconditions: List[str]
    # Structural properties of problems where this strategy applies.
    # Describes WHEN to use the technique. May include negative preconditions.

    complexity: Optional[str] = None
    # Time/space complexity, e.g. "O(n log n) time, O(1) space"

    algorithm_tags: List[str] = field(default_factory=list)
    # Canonical algorithm family tags. Fixed taxonomy:
    # greedy, dp, binary_search, graph_bfs, graph_dfs, graph_dijkstra,
    # graph_mst, segment_tree, binary_indexed_tree, two_pointers,
    # sliding_window, divide_and_conquer, math_number_theory,
    # math_combinatorics, string_hashing, string_kmp, union_find,
    # topological_sort, network_flow, geometry, brute_force,
    # constructive, implementation, sorting, stack, priority_queue


@dataclass
class EpisodicEntry:
    """A single entry in the agent's memory — one solved problem."""

    entry_id: str
    problem_id: str
    problem_statement: str
    strategy: Strategy
    solution_code: str           # Stored but NOT given to model during adaptation
    difficulty_rating: int       # Codeforces rating (800–3500)
    verification_passed: bool

    embedding: Optional[np.ndarray] = None  # Precomputed for retrieval

    # Usage tracking
    times_retrieved: int = 0
    times_led_to_success: int = 0
    times_led_to_failure: int = 0

    created_at: int = 0
    source: str = "solved"       # "solved" or "seed"


@dataclass
class FailureAnnotation:
    """Record of a strategy that was retrieved but did not solve the problem."""

    problem_id: str
    problem_statement: str
    retrieved_entry_id: str
    attempted_strategy: Strategy
    failure_code: str
    error_info: str
    diagnosis: str               # LLM-generated explanation of the mismatch


@dataclass
class Memory:
    """The agent's complete memory state."""

    entries: Dict[str, EpisodicEntry] = field(default_factory=dict)
    failures: List[FailureAnnotation] = field(default_factory=list)
    current_timestep: int = 0


@dataclass
class Problem:
    """A competitive programming problem."""

    problem_id: str
    contest_id: int
    index: str
    title: str
    statement: str
    difficulty_rating: int
    algorithm_tags: List[str]    # Ground truth (for evaluation only, not given to agent)
    test_cases: List[Dict]       # [{"input": str, "expected_output": str}]
    editorial: Optional[str] = None
    reference_solutions: List[str] = field(default_factory=list)
    contest_date: str = ""
