# Experiments Guide

This document describes each experiment, what it measures, and how the training/evaluation pipeline works end-to-end.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  CodeContests Dataset (HF)                              │
│        │                                                │
│        ▼                                                │
│  load_codecontests.py ──► dataset_cc/                   │
│    6,589 CF problems       ├── problems/*.json          │
│    with solutions,         └── splits/                  │
│    generated tests              ├── seed.json (4,986)   │
│                                 ├── eval.json (860)     │
│                                 └── test.json (743)     │
│        │                                                │
│        ▼                                                │
│  train_encoder.py ──► models/technique_encoder_v2/      │
│    Contrastive fine-tuning     Sentence transformer     │
│    of all-MiniLM-L6-v2        for technique-aware       │
│    using algorithm tags        retrieval                │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Inference Pipeline                     │
│                                                         │
│  vLLM Server (Qwen3-8B on H100)                        │
│        │                                                │
│        ▼                                                │
│  run_experiments.py                                     │
│    1. Seed memory: extract strategies from seed          │
│       solutions using LLM                               │
│    2. For each eval problem:                            │
│       a. Encode problem → retrieve top-k from memory    │
│       b. Align strategy to new problem (LLM call)       │
│       c. Generate code (LLM call)                       │
│       d. Verify against test cases (subprocess)         │
│       e. Update memory on success/failure               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Dataset

**Source**: DeepMind's CodeContests dataset (HuggingFace), filtered to Codeforces problems with ratings 800-2500.

| Split | Problems | Contest IDs | Purpose |
|-------|----------|-------------|---------|
| Seed  | 4,986    | < 1200      | Bootstrap memory with reference solutions |
| Eval  | 860      | 1200-1400   | Main evaluation (measure accuracy) |
| Test  | 743      | > 1400 + CC test/valid | Held-out for final reporting |

Each problem includes:
- Problem statement, rating, algorithm tags (33 canonical tags)
- Sample tests (1-4), private tests, generated tests (avg ~87 per problem)
- Up to 5 reference solutions in multiple languages

## Encoder Training (Step 1)

**Script**: `scripts/train_encoder_h100.sbatch`

**What it does**: Fine-tunes `all-MiniLM-L6-v2` (22M params) using contrastive learning so that problems requiring the same algorithmic technique cluster together in embedding space.

**Training data**: 50,000 (anchor, positive) pairs generated from seed problems:
- Positive = shares algorithm tag with anchor
- Hard negatives (30%) = embedding-close but technique-different, mined from the base model

**Loss**: MultipleNegativesRankingLoss (MNRL) — uses in-batch negatives automatically.

**Duration**: ~10 min on H100, ~23 min on A40.

**Output**: `models/technique_encoder_v2/` — loaded by the agent for retrieval.

**Results** (Precision@k for technique-aware retrieval):

| Encoder | P@1 | P@3 | P@5 | P@10 |
|---------|-----|-----|-----|------|
| Random baseline | 45.2% | 45.2% | 45.2% | 45.2% |
| Base (all-MiniLM-L6-v2) | 78.5% | 70.3% | 67.1% | 66.1% |
| Fine-tuned (ours) | 80.0% | 71.8% | 70.4% | 69.1% |

The encoder is a supporting component, not a main contribution. The big gap is random (45%) vs any encoder (70%+), showing retrieval matters. The fine-tuning adds a modest +2-4%.

## Experiment 1: Learning Curve

**Script**: `scripts/run_exp1_learning_curve.sbatch`
**Question**: Does the agent improve over time as it accumulates experience?

**Design**: Run 5 methods on the same 300 eval problems. Seed memory with 200 problems. Plot accuracy as a function of problems seen (cumulative accuracy over time).

**Methods compared**:

| Method | Description |
|--------|-------------|
| **Strategy Adaptation (ours)** | Full system: retrieve → align strategy → generate code → update memory |
| No Memory | Pure LLM code generation, no retrieval or memory |
| Random Retrieval | Retrieve random (not technique-similar) entries from memory |
| Full History | Inject raw solution code instead of extracted strategies |
| Tag Oracle | Retrieval uses ground-truth tags (perfect match) — upper bound |

**Expected outcome**: Strategy Adaptation should show an upward learning curve (accuracy improves as memory grows). No Memory is a flat line. The gap between ours and No Memory quantifies the value of memory. The gap between ours and Tag Oracle shows room for retrieval improvement.

**Estimated runtime**: ~14 hours on H100 (5 methods x ~3h each for 300 eval problems).

## Experiment 2: Granularity Ablation (KEY EXPERIMENT)

**Script**: `scripts/run_exp2_granularity.sbatch`
**Question**: Is strategy-level the right abstraction for memory entries?

**Design**: Run 6 granularity modes on the same 300 eval problems. Each mode controls what information from retrieved entries is shown to the LLM.

| Mode | What's in the prompt | Token budget |
|------|---------------------|-------------|
| G1 | Nothing (free generation, no retrieval) | 0 |
| G2 | Algorithm tag names only | ~20 |
| G3 | Strategy: technique chain + key insight + preconditions | ~200 |
| G4 | Strategy + first 400 chars of solution code | ~500 |
| G5 | Full reference solution (up to 1000 chars) | ~1000 |
| G6 | 3 full reference solutions | ~3000 |

**Expected outcome**: Inverted-U curve. Too little info (G1-G2) hurts because the LLM has no guidance. Too much info (G5-G6) hurts because the LLM copies irrelevant implementation details instead of reasoning about the algorithm. G3 (strategy-level) should be the sweet spot.

**This is the core contribution** — it validates that strategy-level abstraction is better than both "no context" and "full code". If G3 > G5, it shows that abstraction beats copying.

**Estimated runtime**: ~21 hours on H100 (6 modes x ~3.5h each).

**Partial results from first run** (before time limit):
- G1 (No retrieval): 15.7%
- G2 (Tag hints): 37.7%
- G3 (Strategy): 34.7%

## Experiment 3: Retrieval Quality

**Script**: `scripts/run_exp3_retrieval.sbatch`
**Question**: Does technique-aware retrieval matter for downstream accuracy?

**Design**: Run the full agent with 4 different retrieval methods and measure solve rate.

| Method | Retrieval mechanism |
|--------|-------------------|
| Base Encoder | Off-the-shelf all-MiniLM-L6-v2 cosine similarity |
| Fine-tuned Encoder | Our contrastive-trained encoder |
| Random | Random entries from memory (no similarity) |
| Tag Oracle | Perfect retrieval using ground-truth algorithm tags |

**Expected outcome**: Tag Oracle > Fine-tuned >= Base >> Random. The gap between Random and Base quantifies the value of semantic retrieval. The gap between Base and Fine-tuned shows the contrastive training benefit.

**Estimated runtime**: ~11 hours on H100 (4 methods x ~3h each).

## Experiment 4: Adaptation Ablation (Not yet scheduled)

**Question**: Does explicit alignment reasoning help vs direct code generation?

Compares two-step (alignment reasoning + code gen) vs single-step (strategy injected directly into code gen prompt). Tests whether the "think about how to adapt the strategy" step adds value.

## How a Single Eval Problem is Processed

```
1. RETRIEVE: Encode problem statement → cosine similarity → top-3 memory entries
       │
2. ALIGN: LLM call with (problem, retrieved strategies)
       │  "Which of these strategies applies? How should it be adapted?"
       │  Uses Qwen3 thinking mode (/think) for chain-of-thought
       │
3. GENERATE: LLM call with (problem, alignment reasoning)
       │  "Write Python code to solve this problem"
       │  Uses /no_think for deterministic output
       │
4. VERIFY: Run generated code against test cases (subprocess, 10s timeout)
       │  Uses up to 50 tests (public + private + generated from CodeContests)
       │
5. RETRY: If verification fails, try next retrieved strategy (up to 3 attempts)
       │  If all strategies fail, try free generation (no retrieval)
       │
6. UPDATE MEMORY:
       ├── On success: Extract strategy from the working code → add to memory
       └── On failure: Extract diagnosis → add failure entry to memory
```

## How to Run

```bash
# 1. Load dataset (2 min, no GPU needed)
python data_collection/load_codecontests.py --output_dir dataset_cc/

# 2. Train encoder (10 min on H100)
sbatch scripts/train_encoder_h100.sbatch

# 3. Run experiments (submit to 3 H100s in parallel)
sbatch scripts/run_exp1_learning_curve.sbatch
sbatch scripts/run_exp2_granularity.sbatch
sbatch scripts/run_exp3_retrieval.sbatch

# Monitor progress
squeue -u xinchi
tail -f logs/exp1_*.err | grep -E "accuracy|Problem|mode"
```

## Infrastructure

- **LLM**: Qwen3-8B served via vLLM (v0.18.0) on H100
  - ~160 tokens/s generation throughput
  - 15.3 GiB GPU memory, leaving 61 GiB for KV cache
  - Max 54 concurrent 8K-token requests
- **Encoder**: all-MiniLM-L6-v2 fine-tuned, runs on CPU (fast enough for retrieval)
- **Verification**: Sandboxed subprocess with 10s timeout per test case, max 50 tests
- **Storage**: Dataset + models on NFS (`/nfs-share/xinchi/`), HF cache redirected to NFS
