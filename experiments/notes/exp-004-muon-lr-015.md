# exp-004-muon-lr-015

**Description:** Muon LR 0.015 (lower)  
**Val loss:** 5.4718  
**Duration:** 30.5 min  
**Seed:** 42  
**Date:** 2026-04-12 14:11  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.015 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |

## Reasoning

Bracketing the Muon LR optimum. exp-003 showed 0.025 beats baseline 0.02, but we need the lower side too to know where the sweet spot is. 0.015 is -25% from baseline. If worse than 0.025 and worse than baseline, the optimum is above 0.02. If better than baseline but worse than 0.025, the curve peaks somewhere between 0.02 and 0.025. Either way this data point shapes the next LR sweep.

## Outcome

**Better than baseline** (5.4718 vs 5.5235, -0.0516 better than baseline).
