# exp-039-geglu

**Description:** GeGLU activation instead of SwiGLU  
**Val loss:** 3.4646  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-13 19:02  

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

SwiGLU uses silu gating; GeGLU uses gelu gating. Some papers report GeGLU marginally better. Low-risk swap worth testing at phase 3 scale.

## Outcome

**Better than baseline** (3.4646 vs 5.5235, -2.0589 better than baseline).
