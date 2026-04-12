# exp-020-adamw-lr-5e4

**Description:** AdamW LR 5e-4 on compound baseline  
**Val loss:** 4.1243  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-12 23:06  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 16 |

## Reasoning

Phase 2 experiment 4. In phase 1, adamw_lr swept 2e-4 < 3e-4 with 3e-4 winning. Testing 5e-4 on the compound baseline (parallel+n_head=16, val_loss=4.2090). The more expressive architecture may benefit from a higher AdamW LR to make better use of the embedding/norm parameters that AdamW controls.

## Outcome

**Better than baseline** (4.1243 vs 5.5235, -1.3992 better than baseline).
