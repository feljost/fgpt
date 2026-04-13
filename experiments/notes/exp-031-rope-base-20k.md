# exp-031-rope-base-20k

**Description:** RoPE base freq 20k  
**Val loss:** 3.9634  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 09:27  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 20000 |
| n_head | 8 |

## Reasoning

Phase 1 showed RoPE base 100k was marginal vs 10k default. 20k hasn't been tested and sits between the two. A higher base frequency means the rotary embeddings decay more slowly with distance, giving tokens slightly more positional sensitivity at longer ranges. At T=1024, the difference between 10k and 20k may be more meaningful than 10k vs 100k. Worth bracketing to see if there is a sweet spot below 100k.

## Outcome

**Better than baseline** (3.9634 vs 5.5235, -1.5601 better than baseline).
