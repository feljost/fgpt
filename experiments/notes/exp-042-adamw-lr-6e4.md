# exp-042-adamw-lr-6e4

**Description:** AdamW LR 6e-4 (vs 5e-4 baseline)  
**Val loss:** 3.4425  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-14 02:34  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0006 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 20000 |
| n_head | 8 |

## Reasoning

Phase 2 found AdamW LR 5e-4 better than 2e-4 (exp-029). Testing a further bump to 6e-4 — at 2.5h scale we have more steps so the model can handle slightly higher LR before the cosine decay brings it down. If the loss landscape is still favorable, a higher peak LR could extract more from the longer run.

## Outcome

**Better than baseline** (3.4425 vs 5.5235, -2.0809 better than baseline).
