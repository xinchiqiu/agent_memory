# Encoder Training: Technique-Aware Contrastive Fine-Tuning

## What It Does

The retrieval encoder is the component that finds **algorithmically similar** problems in memory when the agent faces a new problem. Off-the-shelf sentence embeddings (e.g. `all-MiniLM-L6-v2`) match problems by **surface text similarity** — two problems about "arrays" will be close even if one requires dynamic programming and the other requires greedy. 

We fine-tune the encoder using **contrastive learning** so that problems requiring the **same algorithmic technique** are close in embedding space, regardless of their narrative framing. After training, a binary search problem about "finding a position in a sorted array" will be closer to a binary search problem about "minimizing the maximum distance" than to a brute force problem about arrays.

## Why It Matters

This is a core contribution of the paper. Without technique-aware retrieval, the agent retrieves irrelevant strategies that waste adaptation attempts. The expected improvement:

| Encoder | P@1 | P@3 | P@5 | P@10 |
|---------|-----|-----|-----|------|
| Random baseline | 53.9% | 53.9% | 53.9% | 53.9% |
| Base (all-MiniLM-L6-v2) | 82.0% | 74.3% | 71.2% | 70.9% |
| **Fine-tuned (ours)** | **81.5%** | **77.8%** | **76.2%** | **74.2%** |
| Improvement | -0.5% | **+3.5%** | **+5.0%** | **+3.2%** |

Note: The random baseline is relatively high because common tags like `implementation` (49% of problems) and `math_number_theory` (27%) create significant tag overlap. The fine-tuned encoder shows meaningful gains at P@3-10, which is the range that matters for top-k retrieval in the agent (k=3).

## How It Works

### Step 1: Training Pair Generation (`pair_generation.py`)

We create training triplets (anchor, positive, negative) using Codeforces algorithm tags as supervision:

- **Positive pair**: Two problems that share at least one algorithm tag (e.g., both tagged `dp`)
  - *Hard positives* (50%): Share the **primary** tag (first in the list) — these are the most important matches
  - *Soft positives* (50%): Share **any** tag — captures cross-technique overlap
- **Negative pair**: A problem that shares **no** tags with the anchor
- **Hard negatives** (30% of training data): Problems that are **embedding-close but technique-different** — mined using the base encoder before training. These are the most informative negatives because they force the model to distinguish structural similarity from surface similarity.

Example triplet:
```
Anchor:   "Find minimum spanning tree of weighted graph"     [graph_mst]
Positive: "Connect cities with minimum total cable cost"     [graph_mst, greedy]
Negative: "Count substrings matching a pattern"              [string_hashing]
```

### Step 2: Contrastive Training (`train_encoder.py`)

We fine-tune `all-MiniLM-L6-v2` (22M parameters) using one of two loss functions:

**MNRL (MultipleNegativesRankingLoss)** — recommended, default:
- Takes (anchor, positive) pairs
- Uses all other positives in the batch as implicit negatives
- With batch_size=64, each example gets 63 automatic negatives
- Efficient: no explicit negative sampling at training time

**Triplet Loss** — alternative:
- Takes explicit (anchor, positive, negative) triplets
- Uses hard-mined negatives directly
- Margin-based: pushes negative at least 0.5 further than positive

Training hyperparameters:
- 50,000 training pairs from ~5,000 seed problems
- 5 epochs, batch size 64, learning rate 2e-5
- 10% warmup steps
- ~30 min on a single GPU (A40 or H100)

### Step 3: Evaluation (`evaluate_encoder.py`)

Measures **Precision@k**: for each eval problem, retrieve the k nearest neighbors from the full dataset and check how many share at least one algorithm tag.

The evaluation compares:
1. **Random baseline** — expected precision from random retrieval (lower bound)
2. **Base model** — off-the-shelf `all-MiniLM-L6-v2` (no fine-tuning)
3. **Fine-tuned model** — our contrastive encoder (should show clear improvement)

## How It Connects to the Agent

```
New Problem → [Encoder] → embedding → cosine similarity search → top-k retrieved
                                                                       ↓
                                                          Strategy Adaptation
                                                                       ↓
                                                            Code Generation
```

The encoder is used in two places:
1. **Memory seeding**: Encode all seed problem statements into embeddings stored in memory
2. **Retrieval**: Encode the new problem, find the closest entries in memory by cosine similarity

A better encoder means the retrieved strategies are more likely to be **algorithmically relevant**, which directly improves the adaptation success rate.

## Running

```bash
# Full training + evaluation (~30-60 min on GPU)
python encoder_training/train_encoder.py \
    --dataset_dir dataset_cc/ \
    --output_dir models/technique_encoder

python encoder_training/evaluate_encoder.py \
    --dataset_dir dataset_cc/ \
    --encoder_path models/technique_encoder

# Quick smoke test (~5 min)
python encoder_training/train_encoder.py \
    --dataset_dir dataset_cc/ \
    --output_dir models/test \
    --num_pairs 1000 --epochs 1

# Via Slurm (A40)
sbatch scripts/train_encoder.sbatch

# Via Slurm (H100)
sbatch scripts/train_encoder_h100.sbatch
```

## Output

After training, `models/technique_encoder/` contains:
- The fine-tuned sentence-transformer model (loadable via `SentenceTransformer("models/technique_encoder")`)
- `training_metadata.json` with hyperparameters and dataset stats

The agent automatically picks up the fine-tuned encoder if it exists at this path (via `encoder.py:create_encoder()`).
