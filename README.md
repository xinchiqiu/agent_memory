# Strategy-Only Adaptation for Competitive Programming

A memory-augmented coding agent that solves competitive programming problems by retrieving and adapting **strategies** (not code) from similar past problems.

## Overview

When facing a new problem the agent:
1. **Retrieves** the most structurally similar solved problems from memory
2. **Adapts** their strategies to the new problem via explicit alignment reasoning
3. **Generates** fresh code guided by the adapted strategic blueprint
4. **Verifies** the solution against test cases
5. **Updates** memory with the new experience (or diagnoses the failure)

The core hypothesis: giving the model a *reasoning path* (technique chain + key insight) from similar problems — but not the code — leads to better generalisation than raw example replay.

---

## Repository Structure

```
agent_memory/
├── data_collection/            # Part A: Codeforces data pipeline
│   ├── cf_api.py               # Codeforces REST API client
│   ├── cf_scraper.py           # Web scraper (statements, solutions, editorials)
│   ├── dataset_utils.py        # Tag normalisation, splits, I/O helpers
│   └── collect.py              # Main collection CLI script
├── encoder_training/           # Part B: Contrastive retrieval encoder
│   ├── pair_generation.py      # Training pair + hard negative generation
│   ├── train_encoder.py        # Fine-tuning script (MNRL / Triplet loss)
│   └── evaluate_encoder.py     # Precision@k evaluation table
└── (agent modules)
```

**Agent modules:**

```
agent_memory/
├── config.py               # All hyperparameters, model names, backend selector
├── data_structures.py      # Strategy, EpisodicEntry, Memory, Problem, FailureAnnotation
├── llm_client.py           # LLM backends: HF local, vLLM, OpenAI API, Anthropic API
├── encoder.py              # Sentence-transformer problem encoder
├── retriever.py            # Cosine similarity retriever + random + tag-oracle variants
├── strategy_extraction.py  # AST analysis + LLM → Strategy object
├── adaptation.py           # Two-step alignment + code generation pipeline
├── verifier.py             # Subprocess-sandboxed test case runner
├── memory_update.py        # Memory entry creation and failure diagnosis
├── agent.py                # Main agent loop with seeding and checkpointing
├── baselines.py            # Four comparison baselines
└── evaluation.py           # Metrics computation and matplotlib figures
```

---

## Installation

```bash
conda activate cp-agent   # or: /nfs-share/xinchi/bin/conda activate cp-agent
# all dependencies are already installed in the cp-agent environment
```

The `cp-agent` conda environment (at `/nfs-share/xinchi/envs/cp-agent`) contains:
`sentence-transformers`, `transformers`, `torch`, `accelerate`, `openai`, `matplotlib`, `seaborn`, `numpy`, `requests`, `beautifulsoup4`, `lxml`

---

## Data Collection (Part A)

### Collect the dataset

```bash
# Full collection (~4 hours, 3000 problems)
python data_collection/collect.py --output_dir dataset/

# Quick test (50 problems, no solutions)
python data_collection/collect.py --output_dir dataset/ --max_problems 50 --skip_solutions

# Resume an interrupted run
python data_collection/collect.py --output_dir dataset/ --resume
```

Dataset output structure:
```
dataset/
  index.json            # lightweight index of all problems
  checkpoint.json       # progress tracker for resuming
  problems/
    1850E.json          # full data per problem
    ...
  splits/
    seed.json           # older problems for seeding memory (~2000)
    eval.json           # middle period for evaluation (~500)
    test.json           # newest problems, held-out (~200)
```

**Temporal split** (contamination-aware for NeurIPS):

| Split | Contest dates | Purpose |
|---|---|---|
| `seed` | before 2023-07-01 | Bootstrap agent memory |
| `eval` | 2023-07-01 → 2024-07-01 | Main evaluation |
| `test` | after 2024-07-01 | Held-out final results |

### Using an existing dataset instead

Before scraping, check if [CodeContests](https://github.com/google-deepmind/code_contests) (DeepMind, ~13K problems) covers your needs — it saves significant scraping time. You can combine it with Codeforces API calls for algorithm tags and ratings.

---

## Contrastive Encoder Training (Part B)

### Why fine-tune?

The base `all-MiniLM-L6-v2` measures surface semantic similarity. We want a model where problems that share **algorithmic techniques** are close in embedding space regardless of narrative framing. Contrastive training on Codeforces algorithm tags achieves this.

Expected improvement after fine-tuning:

| Encoder | P@3 | P@5 |
|---|---|---|
| Random | ~14% | ~14% |
| Base (all-MiniLM-L6-v2) | ~37% | ~33% |
| **Fine-tuned (technique encoder)** | **~60%** | **~55%** |
| Tag Oracle (upper bound) | 100% | 100% |

### Training

```bash
# Standard training (MNRL loss + hard negatives, ~1 hour on 1 GPU)
python encoder_training/train_encoder.py \
    --dataset_dir dataset/ \
    --output_dir models/technique_encoder

# Triplet loss variant (uses explicit hard negatives)
python encoder_training/train_encoder.py \
    --dataset_dir dataset/ \
    --output_dir models/technique_encoder_triplet \
    --loss triplet

# Quick smoke test
python encoder_training/train_encoder.py \
    --dataset_dir dataset/ --output_dir models/test \
    --num_pairs 1000 --epochs 1
```

Key hyperparameters in `train_encoder.py`:

| Argument | Default | Notes |
|---|---|---|
| `--base_model` | `all-MiniLM-L6-v2` | Starting checkpoint |
| `--loss` | `mnrl` | `mnrl` or `triplet` |
| `--num_pairs` | 50000 | Training triplets |
| `--epochs` | 5 | |
| `--batch_size` | 64 | Larger = more in-batch negatives for MNRL |
| `--hard_neg_ratio` | 0.3 | Fraction from hard negative mining |

### Evaluation

```bash
python encoder_training/evaluate_encoder.py \
    --dataset_dir dataset/ \
    --base_model all-MiniLM-L6-v2 \
    --finetuned_model models/technique_encoder \
    --output_json results/encoder_eval.json
```

### Using the fine-tuned encoder in the agent

`create_encoder()` automatically picks the fine-tuned model if it exists:

```python
from encoder import create_encoder
from agent import StrategyAdaptationAgent

encoder = create_encoder()          # uses models/technique_encoder if available
agent = StrategyAdaptationAgent(
    llm_client=client,
    encoder=encoder,
)
```

---

## Quick Start

### 1. Configure the backend

Edit `config.py` and set `BACKEND` to your preferred inference backend:

```python
BACKEND = "hf_local"    # load Qwen3-7B directly via HuggingFace (no server needed)
BACKEND = "vllm"        # call a running vLLM server
BACKEND = "openai"      # call the OpenAI API
BACKEND = "anthropic"   # call the Anthropic API
```

### 2. (Optional) Start a vLLM server for `backend = "vllm"`

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-7B \
    --port 8000 \
    --enable-reasoning
```

### 3. Run the agent

```python
from llm_client import create_llm_client
from agent import StrategyAdaptationAgent
from data_structures import Problem

client = create_llm_client()          # picks backend from config.py
agent  = StrategyAdaptationAgent(llm_client=client, log_dir="logs/run_001")

# Seed memory with pre-solved problems (list of (Problem, solution_code) pairs)
agent.seed_memory(seed_problems)

# Run on evaluation problems (list of Problem objects)
results = agent.run(eval_problems)
print(f"Accuracy: {results['overall_accuracy']:.1%}")
```

### 4. Run baselines

```python
from baselines import NoMemoryBaseline, RandomRetrievalBaseline
from baselines import FullHistoryBaseline, TagOracleBaseline

no_mem = NoMemoryBaseline(llm_client=client)
results_no_mem = no_mem.run(eval_problems)
```

### 5. Generate evaluation plots

```python
from evaluation import generate_all_plots

generate_all_plots(
    main_log_path="logs/run_001/final_results.json",
    baseline_log_paths={
        "No Memory":      "logs/baseline_no_memory/final_results.json",
        "Random":         "logs/baseline_random/final_results.json",
        "Full History":   "logs/baseline_full_history/final_results.json",
        "Tag Oracle":     "logs/baseline_tag_oracle/final_results.json",
    },
    output_dir="plots/",
)
```

---

## Models

| Model | HF identifier | Notes |
|---|---|---|
| Qwen3-7B (default) | `Qwen/Qwen3-7B` | Thinking-capable; used for alignment + code gen |
| Qwen3-8B | `Qwen/Qwen3-8B` | Slightly larger variant |
| Qwen2.5-Coder-7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | Code-specialised, no thinking mode |

Change the active model in `config.py`:

```python
EXTRACTION_MODEL  = QWEN3_7B
ALIGNMENT_MODEL   = QWEN3_7B
GENERATION_MODEL  = QWEN3_7B
DIAGNOSIS_MODEL   = QWEN3_7B
```

### Qwen3 Thinking Mode

Qwen3 supports explicit chain-of-thought reasoning via `/think` / `/no_think` suffixes. This is controlled per-role in `config.py`:

```python
"qwen3_thinking_for_alignment":  True,   # deep reasoning when adapting strategies
"qwen3_thinking_for_generation": False,  # fast deterministic code output
```

The `RoleAwareLLMClient` applies these automatically based on the call site.

---

## Pipeline Details

### Strategy Representation

Each solved problem is stored as a `Strategy` — not code:

```python
Strategy(
    technique_chain  = ["sort intervals by end time",
                        "greedy: pick earliest-ending non-overlapping interval",
                        "track last selected endpoint"],
    key_insight      = "Choosing the earliest-ending interval always maximises remaining space "
                       "(exchange argument).",
    preconditions    = ["maximise count of non-overlapping selections",
                        "items are independent",
                        "objective is cardinality, not weighted sum"],
    complexity       = "O(n log n) time, O(1) space",
    algorithm_tags   = ["greedy", "sorting"],
)
```

### Two-Step Adaptation

1. **Alignment** (temperature 0.3, thinking enabled for Qwen3): the LLM matches retrieved preconditions to the new problem, identifies differences, and writes an adapted plan.
2. **Code Generation** (temperature 0, thinking disabled): the LLM implements the adapted plan as a complete Python program.

### Fallback Chain

If the first adapted solution fails verification, the agent tries the next retrieved strategy, then falls back to free generation (chain-of-thought, no retrieved context).

### Memory Update

- **Success**: extract a `Strategy` from the working code (AST + LLM), embed the problem, add `EpisodicEntry` to memory.
- **Failure**: generate a `FailureAnnotation` diagnosing the structural mismatch between the strategy's preconditions and this problem.

---

## Baselines

| Baseline | Description |
|---|---|
| **No Memory** | Free generation for every problem; no retrieval |
| **Random Retrieval** | Randomly sample k entries from memory; same adaptation pipeline |
| **Full History** | Dump raw text + code of k most recent solved problems into the prompt |
| **Tag Oracle** | Retrieve by ground-truth algorithm tags (upper bound; not available to real agent) |

---

## Evaluation Metrics

- **Overall accuracy** — fraction of problems solved
- **Rolling accuracy** (window=50) — the primary learning curve plot
- **Accuracy by difficulty** — buckets: 800–1199, 1200–1599, 1600–1999, 2000+
- **Method breakdown** — fraction solved by adapted_1/2/3, free generation, or failed
- **Retrieval precision** — how often the retrieved entry shares algorithm tags with the target problem

### Figures

| Figure | Description |
|---|---|
| `fig1_learning_curves.png` | Rolling accuracy over time for all methods |
| `fig2_difficulty_accuracy.png` | Per-difficulty accuracy bar chart |
| `fig3_method_breakdown.png` | Stacked area chart of solution method over time |
