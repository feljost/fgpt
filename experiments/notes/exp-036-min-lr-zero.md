# exp-036-min-lr-zero

**Description:** Full cosine decay to LR=0  
**Val loss:** 3.9664  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 11:29  

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

Baseline keeps a 5% LR floor at the end of the cosine schedule. Decaying fully to 0 squeezes out a few more steps of fine-tuning at the tail end. The risk is that the very low LR region near zero contributes little useful signal and just wastes steps. At 60 minutes (~600 optimizer steps), the cosine tail is short anyway — this tests whether the floor helps or hurts at this run length.

## Outcome

**Better than baseline** (3.9664 vs 5.5235, -1.5570 better than baseline).
