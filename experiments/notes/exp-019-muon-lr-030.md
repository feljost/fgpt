# exp-019-muon-lr-030

**Description:** Muon LR 0.030 on compound baseline  
**Val loss:** 4.2114  
**Duration:** 60.4 min  
**Seed:** 42  
**Date:** 2026-04-12 22:05  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.03 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 16 |

## Reasoning

Phase 2 experiment 3. In phase 1, muon_lr swept 0.015 < 0.020 < 0.025 with 0.025 winning. The compound baseline (parallel+n_head=16) is a more expressive architecture — it may benefit from a higher Muon LR. Testing 0.030 to see if the sweep continues upward on the stronger baseline (compound val_loss=4.2090).

## Outcome

**Better than baseline** (4.2114 vs 5.5235, -1.3120 better than baseline).
