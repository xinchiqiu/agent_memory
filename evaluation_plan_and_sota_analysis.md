# Evaluation Plan & SOTA Landscape Analysis
## Strategy-Augmented Memory for Competitive Programming Agents

---

# Part A: Current SOTA for Coding Agents Using Memory

## The Landscape at a Glance

There is no single system that does exactly what we propose. The relevant prior work
falls into distinct categories, each solving a different sub-problem. Understanding
where each system sits — and where the gaps are — is critical for positioning our work.

## 1. Memory-Augmented Coding Agents (Software Engineering)

### MemCoder (February 2026) — Current SOTA on SWE-bench

MemCoder is the most recent and strongest result for memory-augmented coding agents.
It structures historical human commit data into long-term memory, retrieves relevant
past commit patterns when facing new issues, and crystallizes human-verified solutions
back into memory. On SWE-bench Verified, it achieves SOTA performance and delivers a 
9.4% improvement in resolved rate when applied to DeepSeek-V3.2 (68.4% → 77.8%).

**What it does well:** Dual-stage retrieval (coarse then fine), experience 
self-internalization (storing verified solutions), and co-evolution with human developers.

**What it does NOT do:** It operates on repository-level bug fixing (SWE-bench), 
not algorithmic reasoning. Memory entries are commit-level patterns (file changes, 
intent-to-code mappings), not algorithmic strategies. There is no notion of technique 
families, no structural similarity beyond text embedding, and no strategy-level 
abstraction. The problems in SWE-bench are software engineering tasks (fix this bug, 
implement this feature) — fundamentally different from competitive programming where 
the challenge is discovering the right algorithm.

**Gap our work fills:** MemCoder shows memory helps for SE tasks. We show memory 
helps for algorithmic reasoning — a different and arguably harder transfer problem, 
because two competitive programming problems can share deep algorithmic structure 
with zero surface similarity.

### SWE-Bench-CL (June 2025) — Continual Learning Benchmark for Coding Agents

This introduces the first continual learning benchmark for coding agents, organizing 
SWE-bench tasks into chronological sequences and measuring forward/backward transfer. 
It includes a FAISS-backed semantic memory module and proposes CL-specific metrics.

**What it does well:** Formalizes the continual learning evaluation protocol for 
coding agents. Proposes metrics for forgetting, forward transfer, and stability-plasticity.

**What it does NOT do:** The paper proposes the benchmark and metrics but the 
empirical evaluation using memory-enabled agents was noted as "ongoing" at time of 
publication. The memory module uses standard vector similarity retrieval without any 
technique-aware encoding or structured strategy representation.

**Gap our work fills:** We provide the actual empirical results on a learning-curve 
evaluation with a concrete memory mechanism, while SWE-Bench-CL provides the 
evaluation framework but limited experimental results.


## 2. Experiential Learning Agents (General Decision-Making)

### ExpeL (AAAI 2024)

ExpeL autonomously gathers experiences through trial and error, extracts natural 
language insights, and retrieves past successful trajectories as in-context examples 
at test time. Evaluated on HotpotQA, ALFWorld, and WebShop.

**What it does well:** Clean separation of experience gathering, insight extraction, 
and test-time retrieval. Shows cross-task transfer (HotpotQA → FEVER). Demonstrates 
that experience improves performance without parameter updates.

**What it does NOT do:** Operates on decision-making tasks (web navigation, 
household tasks, QA), not algorithmic reasoning. Insights are flat natural language 
rules, not structured strategy representations. Retrieval uses surface similarity 
of task descriptions. No notion of technique families or structural similarity.
No failure-informed boundary conditions.

**Gap our work fills:** We apply the experiential learning paradigm specifically to 
algorithmic problem-solving, with domain-specific innovations: structured strategy 
extraction (technique chain + insight + preconditions), technique-aware retrieval 
via contrastive learning, and a two-step adaptation process with explicit alignment.


### Dynamic Cheatsheet (EACL 2026)

DC endows black-box LMs with persistent, evolving memory at inference time. 
The model stores and reuses accumulated strategies, code snippets, and heuristics 
across sequential queries. Claude 3.5 Sonnet's accuracy more than doubled on AIME 
math exams; GPT-4o went from 10% to 99% on Game of 24.

**What it does well:** Strong results on math reasoning (AIME) and algorithmic 
puzzles. Demonstrates that self-curated memory outperforms full-history retention. 
Works with black-box models (no parameter access needed).

**What it does NOT do:** Memory entries are flat text snippets (code snippets, 
heuristic tips), not structured strategy representations. No explicit retrieval step 
— the entire memory is included in every prompt (works for small memories, doesn't 
scale). No technique-aware similarity matching. No explicit adaptation reasoning. 
The "cheatsheet" accumulates tricks but doesn't build a structured understanding of 
technique families and their applicability conditions.

**Gap our work fills:** We provide structured, indexed memory (enabling selective 
retrieval at scale) with technique-level representations (enabling analogical 
transfer based on deep structure, not surface similarity). DC's approach of including 
the entire memory in every prompt works for small problem sets (30 AIME problems) 
but can't scale to hundreds or thousands of problems.


## 3. Analogical Reasoning in LLMs

### Analogical Prompting (ICLR 2024)

Prompts LLMs to self-generate relevant exemplars before solving a problem. Inspired 
by human analogical reasoning. Outperforms 0-shot CoT and manual few-shot CoT on 
math, code generation, and reasoning tasks.

**What it does well:** Shows that generating relevant examples helps reasoning. 
No labeled exemplars needed. Achieves +4% average accuracy gain across tasks.

**What it does NOT do:** The model generates "imagined" examples from parametric 
memory, not from an external memory of actually solved problems. A follow-up study 
(ACL 2025 Findings) showed that self-generated random examples can achieve comparable 
performance to relevant ones — raising questions about whether the model does genuine 
analogical transfer or just activates better reasoning modes. No persistent memory 
across problems. No learning curve (performance doesn't improve over time).

**Critical finding from follow-up work:** The paper "Can LLMs Truly Perform Analogical 
Reasoning?" found that the benefit of analogical prompting may come more from the 
generation process itself (forcing the model to reason about problem structure) than 
from the relevance of the generated examples. This is an important finding for us — 
it suggests that our two-step adaptation process (which also forces structural 
reasoning) may capture a similar benefit, and that adding REAL relevant examples from 
external memory on top of this should provide additional gains.

**Gap our work fills:** We provide real external memory of actually solved problems 
(not imagined ones), persistent and growing over time, with technique-aware retrieval. 
Our two-step adaptation captures the "structural reasoning" benefit of analogical 
prompting while adding the additional benefit of genuinely relevant past solutions.


## 4. Memory Architectures for LLM Agents (General Purpose)

### A-MEM (NeurIPS 2025)

Agentic Memory system inspired by the Zettelkasten method. Creates interconnected 
knowledge networks through dynamic indexing and linking. Each memory note contains 
contextual descriptions, keywords, tags, and links to related notes.

**What it does well:** Dynamic self-organization of memory. Efficient retrieval 
(1,200-2,500 tokens vs. 16,900 for baselines). Strong results on conversation QA 
benchmarks (LoCoMo).

**What it does NOT do:** General-purpose memory for conversations, not designed for 
algorithmic problem-solving. No notion of solution strategies, technique families, 
or problem-solving trajectories. Evaluated on QA and dialogue, not reasoning tasks.

### SAGE (Neurocomputing 2025)

Self-evolving agents with reflective and memory-augmented abilities. Uses Ebbinghaus 
forgetting curve for memory optimization. Three-agent framework (User, Assistant, 
Checker) with iterative feedback and reflection.

**What it does well:** Memory optimization via forgetting curves. Self-reflection 
mechanism. Significant improvements on AgentBench (2.26X on closed-source models).

**What it does NOT do:** General-purpose agent framework, not specialized for 
algorithmic reasoning. Memory stores interaction history, not structured problem-solving 
strategies. No technique-aware retrieval or strategy-level abstraction.

### mem-agent (HuggingFace, 2025)

RL-trained memory agent using multi-turn GRPO. Trains an agent specifically for 
memory management (retrieval, update, clarification) using reinforcement learning 
with verifiable rewards.

**What it does well:** End-to-end trained memory management. Available as MCP server.
Uses RL to learn optimal memory CRUD operations.

**What it does NOT do:** General-purpose memory management, not specialized for 
problem-solving. The RL training focuses on memory operations, not on solving 
reasoning tasks.


## 5. Summary Comparison Table

| System | Domain | Memory Content | Retrieval Method | Strategy Extraction | Learning Curve | Task |
|--------|--------|---------------|------------------|--------------------| --------------|------|
| **Ours (proposed)** | Competitive Programming | Structured strategies (technique chain + insight + preconditions) | Technique-aware contrastive encoder | AST + LLM extraction | Yes (improves over time) | Algorithmic reasoning |
| MemCoder | Software Engineering | Commit patterns, intent-to-code mappings | Dual-stage text retrieval | From commit history | Partial (co-evolution) | Bug fixing |
| ExpeL | General decision-making | Full trajectories + flat NL insights | Surface text similarity | LLM comparison of success/failure | Yes | HotpotQA, ALFWorld |
| Dynamic Cheatsheet | Math & puzzles | Flat code snippets & heuristic tips | No retrieval (full memory in prompt) | Self-curated during inference | Yes | AIME, Game of 24 |
| Analogical Prompting | Math & code generation | Self-generated imagined examples (no external memory) | N/A (parametric) | N/A | No | GSM8K, MATH, Codeforces |
| A-MEM | Conversation QA | Zettelkasten-style notes with links | Embedding + linked notes | Agentic self-organization | No | LoCoMo QA |
| SAGE | General agents | Interaction history with forgetting | Ebbinghaus-based retention | Reflective summarization | Yes | AgentBench |
| SWE-Bench-CL | Software Engineering | Vectorized task summaries | FAISS vector search | From problem/solution pairs | Benchmark proposed | SWE-bench tasks |


## 6. The Clear Gap Our Work Fills

No existing system combines ALL of these:

1. **Structured strategy representation** (technique chain + key insight + preconditions)
   — rather than flat text or full code
2. **Technique-aware retrieval** trained on algorithm family labels
   — rather than surface text similarity
3. **Two-step adaptation with explicit alignment reasoning**
   — rather than single-step generation with retrieved context
4. **Applied to algorithmic reasoning** where deep structural transfer is the challenge
   — rather than SE tasks, QA, or general decision-making
5. **Evaluation with granularity ablations** showing strategy-level is the right abstraction
   — no prior work compares hint-level vs. strategy-level vs. full-solution retrieval

The closest systems are ExpeL (experiential learning paradigm, but wrong domain and
wrong representation) and Dynamic Cheatsheet (right domain of math/algorithmic, but
wrong memory structure and no retrieval mechanism). Our work sits at the intersection.


---


# Part B: Complete Evaluation Plan

## Overview

The evaluation answers six fundamental questions, each requiring specific experiments
and metrics. Every experiment should be runnable independently, and results from 
different experiments should compose into a coherent narrative.

```
Q1: Does the agent improve over time?              → Learning Curve
Q2: Is strategy-level the right abstraction?        → Granularity Ablation
Q3: Does technique-aware retrieval matter?          → Retrieval Quality
Q4: Does explicit alignment reasoning help?         → Adaptation Ablation
Q5: Where does memory help most?                    → Breakdown Analysis
Q6: Does this generalize across models?             → Multi-Model Experiment
```


## Experiment 1: Learning Curve (Q1)

### Question
Does accumulated experience improve the agent's performance on new problems?

### Setup
- Seed memory with 200 pre-solved problems (from the seed set, pre-2023)
- Process 500 evaluation problems sequentially in chronological order
- After each problem, update memory with the outcome
- Measure rolling accuracy over sliding windows of 50 problems

### Methods to Compare

| ID | Method | Description |
|----|--------|-------------|
| M1 | No Memory | Free generation every time. No retrieval. |
| M2 | Random Retrieval | Retrieve 3 random entries from memory, then use same adaptation pipeline |
| M3 | Surface Retrieval | Off-the-shelf sentence encoder (no fine-tuning), strategy adaptation |
| M4 | Full System | Fine-tuned technique-aware encoder + strategy adaptation |
| M5 | Tag Oracle | Use ground truth Codeforces tags to retrieve, strategy adaptation |

All methods use the same base LLM (Qwen2.5-Coder-7B-Instruct) and the same verifier.

### Metrics
- **Rolling accuracy**: accuracy over the last 50 problems, plotted at each step
- **Cumulative accuracy**: total successes / total problems seen so far
- **Final accuracy**: accuracy over the last 100 problems (steady-state performance)

### Expected Results
M1 (flat line) < M2 (slightly above M1) < M3 (improving curve) < M4 (steeper improving curve) ≤ M5 (ceiling)

### Key Plot
**Figure 1 of the paper.** X-axis: problems seen (0–500). Y-axis: rolling accuracy (window=50). One line per method. This is the headline result.

### Statistical Rigor
Run with 3 different random seeds (different orderings of the eval problems within the same temporal window). Report mean and standard deviation. Use a paired t-test or bootstrap confidence interval on the difference between M4 and M1.


## Experiment 2: Granularity Ablation (Q2)

### Question
What level of information from retrieved problems produces the best adaptation?

### Setup
- Fix the retrieval method (use Full System retriever for all variants)
- Fix the same top-3 retrieved problems for each test problem across all variants
- Vary only what information is shown to the model from the retrieved problems

### Variants

| ID | Variant | Information Shown | Approx Tokens |
|----|---------|-------------------|---------------|
| G1 | No Retrieval | Nothing (free generation) | 0 |
| G2 | Tag Hint | Algorithm tag names only ("binary_search, greedy") | ~20 |
| G3 | Strategy Only (proposed) | Technique chain + key insight + preconditions | ~200 |
| G4 | Strategy + Code Snippet | Strategy plus first 30 lines of solution code | ~500 |
| G5 | Full Solution | Complete problem statement + full solution code | ~1000 |
| G6 | Full Solution × 3 | All 3 retrieved problems with full solutions | ~3000 |

### Metrics
- Accuracy on the same set of 300 evaluation problems (static memory, no updates)
- Accuracy on "hard" problems (rating ≥ 1600) separately

### Key Control: Token-Matched Comparison
A reviewer might argue Strategy Only wins just because shorter prompts are less 
distracting. To control for this:
- G3-token-matched: Give the full solution but randomly truncated to ~200 tokens 
  (same token budget as strategy)
- If G3 > G3-token-matched, the advantage is from the quality of abstraction, 
  not prompt length

### Expected Results
G1 < G2 < G3 ≈ G4 > G5 > G6

Strategy Only (G3) should outperform Full Solution (G5) because full solutions
over-constrain the model and introduce irrelevant implementation details. Full 
Solution × 3 (G6) should be worse due to prompt pollution. Tag Hint (G2) should 
be better than nothing but worse than strategy because it lacks the key insight 
and structural reasoning.

### Key Table
**Table 1 of the paper.** Rows: G1–G6. Columns: Overall accuracy, accuracy on 
hard problems (≥1600), accuracy on easy problems (<1200).


## Experiment 3: Retrieval Quality Analysis (Q3)

### Question
Does the contrastive-trained encoder retrieve more technique-relevant problems?

### 3A: Intrinsic Retrieval Evaluation

**Setup:** For each of 500 eval problems, retrieve top-k from the seed memory. 
Check whether retrieved entries share algorithm tags with the query.

**Metrics:**
- Precision@k for k ∈ {1, 3, 5, 10}
- Mean Reciprocal Rank (MRR): 1/rank of first relevant retrieval
- Recall@k: fraction of the query's tags represented in top-k retrievals

**Methods to compare:**
- Random retrieval (lower bound)
- BM25 text retrieval
- Off-the-shelf sentence transformer (all-MiniLM-L6-v2)
- Fine-tuned contrastive encoder (proposed)
- Tag oracle (upper bound)

### 3B: Extrinsic Retrieval Evaluation (End-to-End Impact)

**Setup:** Use the same adaptation pipeline, swap only the retrieval method. 
Measure end-to-end accuracy.

**Key analysis:** Partition eval problems into:
- "Relevant retrieval": at least one of top-3 retrieved entries shares a tag
- "Irrelevant retrieval": none of top-3 share any tag

Show that accuracy is substantially higher for "relevant retrieval" cases. Then 
show that the fine-tuned encoder produces more "relevant retrieval" cases than the 
base encoder.

### 3C: Retrieval Quality Over Time

**Setup:** During the learning curve experiment (Exp 1), log retrieval precision 
at each step.

**Plot:** X-axis: memory size. Y-axis: precision@3. As memory grows, does 
retrieval quality improve (more potential matches) or degrade (more noise)?


## Experiment 4: Adaptation Ablation (Q4)

### Question
Does the two-step adaptation process (alignment → code generation) outperform 
simpler approaches?

### Variants

| ID | Variant | Process |
|----|---------|---------|
| A1 | Two-Step (proposed) | Alignment LLM call → adapted plan → code generation LLM call |
| A2 | Single-Step | Put retrieved strategies directly into code generation prompt (no separate alignment) |
| A3 | Plan-Only | Give the adapted plan from A1 but NOT the original retrieved strategies |
| A4 | No Adaptation | Give retrieved strategies as raw context, ask model to "solve, using these as hints" |

All variants use the same retrieved strategies (from Full System retriever).

### Metrics
- Accuracy on 300 evaluation problems
- Analysis of failure modes: when A1 succeeds but A2 fails, what went wrong? 
  (Categorize failures manually for ~50 cases)

### Expected Results
A1 > A2 > A4, demonstrating that explicit alignment reasoning helps.
A3 ≈ A1, suggesting the value is in the alignment reasoning process, not in 
the raw retrieved content.


## Experiment 5: Breakdown Analysis (Q5)

### Question
Where does memory help most, and where does it fail?

### 5A: By Difficulty Rating

**Buckets:** 800-1200 (easy), 1200-1600 (medium), 1600-2000 (hard), 2000+ (very hard)

**Plot:** Grouped bar chart. For each bucket, show accuracy for Full System vs. 
No Memory. Compute the absolute improvement in each bucket.

**Hypothesis:** Memory helps most for medium-hard problems (1200-2000). Easy 
problems the model solves anyway. Very hard problems may need novel techniques 
not yet in memory.

### 5B: By Algorithm Technique Family

**Group problems by primary tag.** For each tag with ≥ 20 eval problems, 
compute accuracy improvement from memory.

**Expected insight:** Formulaic technique families (binary search on the answer, 
standard DP patterns, BFS/DFS) should benefit a lot. Idiosyncratic families 
(constructive algorithms, ad-hoc implementation) should benefit less.

**Table:** Rows: technique families. Columns: # problems, no-memory accuracy, 
full-system accuracy, improvement, retrieval precision for this family.

### 5C: By Technique Novelty

**Classification:** For each eval problem, check whether its primary algorithm 
tag appears in any seed memory entry.
- "Technique seen": yes (the technique family is in memory)
- "Technique unseen": no (the agent has never solved a problem with this technique)

**Expected result:** Large improvement for "technique seen" (genuine analogical 
transfer). Small or no improvement for "technique unseen" (memory can't help with 
completely novel techniques — this is expected and honest to report).

**This is the strongest evidence of transfer:** The agent has never seen this 
specific problem, but it has solved problems with the same technique. It applies 
the known technique to the new problem. That's real analogical reasoning.

### 5D: By Fallback Level

**Track for each problem:** Which fallback level produced the successful solution?
- adapted_1: first retrieved strategy worked
- adapted_2: second strategy worked
- adapted_3: third strategy worked
- free: no retrieved strategy worked, fell back to free generation
- failed: nothing worked

**Plot:** Stacked area chart over time. X-axis: problems seen. Y-axis: fraction 
of problems in each category. As memory grows, adapted_1 should increase and 
free/failed should decrease.

### 5E: Error Analysis (Qualitative)

Manually examine 50 failure cases and categorize them:
- **Retrieval failure**: retrieved strategies were irrelevant (wrong technique family)
- **Adaptation failure**: right technique retrieved but model failed to adapt it correctly
- **Technique gap**: the required technique is not in memory at all
- **Verification edge case**: solution is conceptually correct but fails on edge cases
- **Inherent difficulty**: problem is beyond the model's capability regardless of memory

Report the distribution. This tells you where to invest future effort. If most 
failures are retrieval failures, improve the encoder. If most are adaptation failures, 
improve the prompts. If most are technique gaps, seed more problems.


## Experiment 6: Multi-Model Generalization (Q6)

### Question
Does the framework help different LLMs, including frontier models?

### Setup
Run the full system (Experiment 1, M4) with:
- Qwen2.5-Coder-7B-Instruct (primary, open-source)
- Qwen2.5-Coder-32B-Instruct (larger open-source)
- GPT-4o (API, frontier)
- Claude Sonnet (API, frontier) — budget permitting

For each model, also run the No Memory baseline (M1).

### Metrics
- Absolute accuracy for each model × method combination
- Relative improvement from memory for each model

### Expected Results
| Model | No Memory | With Memory | Improvement |
|-------|-----------|-------------|-------------|
| 7B | Low (~25%) | Higher (~40%) | Large (+15%) |
| 32B | Medium (~40%) | Higher (~50%) | Moderate (+10%) |
| GPT-4o | High (~55%) | Higher (~60%) | Small but meaningful (+5%) |

The ideal story: memory helps all models, with diminishing but still positive 
returns for stronger models. This proves the framework is a general capability 
enhancement, not a crutch for weak models.


## Experiment 7: Memory Quality Analysis

### Question
Does the memory evolve in useful ways over time?

### 7A: Memory Growth and Coverage

**Plot:** Memory size (# entries) over time. Should grow roughly linearly with 
success rate (only successful solutions add entries).

**Coverage analysis:** After the full run, what fraction of the algorithm tag 
taxonomy has at least one entry in memory? Plot this over time.

### 7B: Entry Utility Distribution

After the full run, for every memory entry compute:
- Times retrieved
- Times led to successful adaptation
- Success rate when retrieved

**Plot:** Histogram of entry utility scores. Identify "hero entries" (high utility, 
frequently retrieved and helpful) and "dead weight" (never retrieved or always failed).

Expected: Pareto distribution — a small number of entries cover common technique 
patterns and are retrieved frequently, while many entries are rarely used.

### 7C: Strategy Extraction Quality

For problems where we have ground truth Codeforces tags, check if the extracted 
strategy's algorithm_tags match.

**Metric:** Tag extraction accuracy (does the extracted tag set overlap with the 
ground truth tag set?). Compute precision and recall.


## Summary: What Makes a Strong Paper

### Minimum Viable Results

These are the results you NEED for a top-venue submission:

1. Learning curve shows clear upward trend for the full system (Exp 1)
2. Strategy-only outperforms full-solution by ≥ 5% absolute (Exp 2)
3. Fine-tuned encoder outperforms base encoder by ≥ 15pp in precision@3 (Exp 3A)
4. Two-step adaptation outperforms single-step by ≥ 3% absolute (Exp 4)
5. Improvement visible across at least 3 of 4 difficulty buckets (Exp 5A)
6. Memory helps at least one API model by ≥ 3% absolute (Exp 6)

### Ideal Results (Would Make the Paper Very Strong)

7. "Technique seen" vs "technique unseen" analysis shows clear transfer (Exp 5C)
8. Fallback distribution shifts toward adapted_1 over time (Exp 5D)
9. Error analysis reveals actionable failure taxonomy (Exp 5E)
10. Strategy extraction accuracy > 70% (Exp 7C)


## Recommended Implementation Order for Experiments

1. **Experiment 1** (Learning Curve) — run this first, it's the headline result
2. **Experiment 2** (Granularity Ablation) — your strongest novelty claim
3. **Experiment 5A-5D** (Breakdowns) — comes "free" from logging during Exp 1
4. **Experiment 3A** (Retrieval Quality) — fast to run, important for understanding
5. **Experiment 4** (Adaptation Ablation) — validates the two-step design
6. **Experiment 6** (Multi-Model) — requires API budget, run after main results solid
7. **Experiment 3B-3C** (End-to-End Retrieval) — requires multiple full runs
8. **Experiment 7** (Memory Quality) — analysis of Exp 1 artifacts
9. **Experiment 5E** (Error Analysis) — manual, do last

Total compute estimate:
- Experiments 1-5 with 7B model: ~5 full pipeline runs × ~500 problems × ~3 LLM calls each = ~7,500 LLM calls per run = ~37,500 total. On a single A100 with vLLM: ~3-5 days.
- Experiment 6 API models: ~2,000 API calls per model = $200-500 per model.
- Total budget: ~1 week of GPU time + $500-1000 in API costs.
