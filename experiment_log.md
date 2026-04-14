# SAGE Experiment Log

## Dataset

- **Source**: DeepMind CodeContests (HuggingFace), filtered to Codeforces problems rated 800-2500
- **Total problems**: 6,589 (with Python-preferred reference solutions)
- **Splits**: Seed 4,986 / Eval 860 / Test 743
- **Tags**: 33 canonical algorithm tags
- **Tests per problem**: avg ~87 (public + private + generated)

---

## Encoder Training

### v1 (all-MiniLM-L6-v2, old 18-tag taxonomy)
- Discarded — tag taxonomy too coarse, only 18 tags

### v2 (all-MiniLM-L6-v2, 33-tag taxonomy)
- P@1=80.0%, P@3=71.8%, P@5=70.4%, P@10=69.1%
- Improvement over base: +1.5% to +3.0%
- **Problem discovered**: Embedding collapse — mean pairwise cosine sim 0.92, same-tag vs diff-tag gap only 0.003. Retrieval is effectively random.

### v3 (all-mpnet-base-v2, 33-tag taxonomy + statement compression)
- P@1=80.4%, P@3=72.7%, P@5=70.5%, P@10=69.0%
- Same-tag vs diff-tag gap: 0.011 (3x better than v2 but still small)
- **Still collapsed**: mean pairwise sim 0.93

### LLM Fingerprinting (current approach)
- LLM generates 2-3 sentence algorithmic fingerprint per problem
- Fingerprints are distinctive: "Greedy with sorting", "Segment tree with lazy propagation", etc.
- Cached to disk (444 fingerprints generated)
- **Result**: Fingerprints are qualitatively better but downstream accuracy still shows no retrieval benefit (see Exp 3)

---

## Experiment 1: Learning Curve (Qwen3-8B)

**Question**: Does the agent improve over time with memory?

| Method | Accuracy | 800-1200 | 1200-1600 | 1600-2000 | 2000+ |
|--------|----------|----------|-----------|-----------|-------|
| Random Retrieval | **37.0%** | 71% | 38% | 29% | 18% |
| Tag Oracle | 33.7% | 69% | 41% | 22% | 12% |
| **Strategy Adaptation (ours)** | **33.3%** | 65% | 39% | 24% | 14% |
| No Memory | 16.3% | 44% | 13% | 8% | 5% |
| Full History | 13.0% | 29% | 13% | 8% | 5% |

### Key findings

1. **Memory doubles performance**: Any memory method (33-37%) >> No Memory (16.3%). Memory helped on 52 extra problems, hurt on only 1.
2. **Full History is worst**: 13.0% — injecting raw code actively confuses the model. Worse than no memory.
3. **Retrieval quality doesn't differentiate**: Random (37.0%) ≈ Tag Oracle (33.7%) ≈ Strategy Adaptation (33.3%). Even perfect retrieval (Tag Oracle) doesn't help because the ~35% ceiling is set by Qwen3-8B's generation ability.
4. **Random slightly beats structured retrieval**: Likely because random entries provide more diversity — structured retrieval picks similar entries that may all fail on the same aspects.

---

## Experiment 2: Granularity Ablation (Qwen3-8B)

**Question**: Is strategy-level the right abstraction for memory entries?

| Mode | Description | Accuracy | 800-1200 | 1200-1600 | 1600-2000 | 2000+ |
|------|-------------|----------|----------|-----------|-----------|-------|
| G5 | Full solution (~1000 tok) | **38.0%** | **74%** | 43% | **28%** | **17%** |
| G4 | Strategy + snippet (~500 tok) | 36.0% | 71% | **46%** | 20% | 17% |
| G2 | Tag hints (~20 tok) | 34.3% | 66% | 39% | 24% | 16% |
| G3 | Strategy (~200 tok) [OURS] | 34.0% | 63% | **46%** | 24% | 12% |
| G6 | 3 full solutions (~3000 tok) | 21.7% | 49% | 25% | 12% | 8% |
| G1 | No retrieval | 17.0% | 43% | 13% | 11% | 6% |

### Key findings

1. **Inverted-U shape confirmed**: G1 (17%) < G2-G5 (34-38%) > G6 (22%). Both no context and too much context hurt.
2. **G5 (full solution) beats G3 (strategy)**: 38.0% vs 34.0%. The extracted strategy loses useful information that the full code retains.
3. **Sweet spot is broad**: G2 through G5 (34-38%) all perform similarly. The model is robust to abstraction level within this range.
4. **G3 shines on mid-difficulty**: At 1200-1600, G3 and G4 both hit 46% vs G5's 43%. Strategy helps when problems are "hard enough to need guidance but solvable with the right approach."
5. **G6 collapse is dramatic**: 3 full solutions (21.7%) is worse than 1 solution (38.0%). Information overload is real.
6. **G3 vs G5 head-to-head**: 92 problems solved by both, 10 by G3 only, 22 by G5 only. G5 strictly dominates.

### Root cause: Strategy extraction quality
- 46% of seed entries had empty `technique_chain` (fixed in later runs — now ~9% empty)
- But even after fixing, G3 ≈ G2, suggesting the extracted strategies don't add much beyond tag-level information
- The strategy extraction may be too generic: "use DP" vs "use interval DP with prefix optimization" — the former is tag-level, the latter would be truly useful

---

## Experiment 3: Retrieval Quality (Qwen3-8B)

**Question**: Does technique-aware retrieval matter for downstream accuracy?

| Method | Accuracy |
|--------|----------|
| Random retrieval | **36.7%** |
| Tag Oracle | 36.3% |
| Base encoder | 33.3% |

### Key findings

1. **Retrieval method doesn't matter**: All methods within ~4% of each other.
2. **Head-to-head overlap is massive**: 90+ problems solved by all methods, only 5-20 unique solves per method.
3. **Root cause**: Embedding collapse — all problems have cosine similarity 0.93-0.97, so "similar" retrieval is effectively random.
4. **Even Tag Oracle doesn't help**: Perfect tag-matched retrieval (36.3%) ≈ Random (36.7%). The bottleneck is downstream generation, not retrieval.

---

## Experiment: Qwen3-32B (partial, 284/300 problems)

**Question**: Is the ~35% accuracy ceiling a model capacity bottleneck?

| Method | Qwen3-8B | Qwen3-32B | Delta |
|--------|----------|-----------|-------|
| Strategy Adaptation | 33.3% | **46.5%** | **+13.2%** |

### Difficulty breakdown

| Difficulty | 8B | 32B | Delta |
|------------|-----|-----|-------|
| 800-1200 | 65% | 74% | +10% |
| 1200-1600 | 39% | 54% | +14% |
| 1600-2000 | 24% | 39% | +15% |
| 2000+ | 14% | 27% | +13% |

### Key findings

1. **The 35% ceiling was a model bottleneck**: 32B breaks through to 46.5% — a 13.2% absolute improvement.
2. **Improvement is uniform**: +10-15% at every difficulty level.
3. **Biggest gain on 1600-2000**: +15% — these are problems where strategy adaptation matters most.
4. **Still missing**: No Memory and Random Retrieval baselines with 32B. Needed to determine if retrieval quality matters with a stronger model.

---

## Running Experiments

- **Qwen3-32B**: Job 39582 — completing No Memory and Random Retrieval baselines
- **Gemma-4-31B-it**: Job 39629 — first run, multi-model comparison

---

## Summary of Robust Findings

| Finding | Evidence | Strength |
|---------|----------|----------|
| Memory doubles performance | 16% → 33-37% across all configs | Strong |
| Raw code injection fails | Full History 13% < No Memory 16% | Strong |
| Information overload hurts | G6 (22%) vs G5 (38%) | Strong |
| Any context helps | G1 (17%) vs G2-G5 (34-38%) | Strong |
| Sweet spot is broad, not sharp | G2-G5 within 4% of each other | Strong |
| Stronger model raises ceiling | 8B 33% → 32B 47% | Strong |
| Retrieval quality doesn't matter (with 8B) | Random ≈ Tag Oracle ≈ Encoder | Strong |
| Strategy < Full solution (with 8B) | G3 34% vs G5 38% | Moderate |

## Open Questions

1. Does retrieval quality matter with a stronger model (32B)?
2. Does the method generalize across model families (Gemma vs Qwen)?
3. Can better strategy extraction (more specific, less generic) close the G3-G5 gap?
