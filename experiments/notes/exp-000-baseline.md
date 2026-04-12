# exp-000-baseline

**Description:** current config unchanged  
**Val loss:** 5.5235  
**Duration:** 30.6 min  
**Seed:** 42  
**Date:** 2026-04-12 12:06  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0002 |
| Muon LR | 0.02 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |

## Reasoning

Establishes the baseline val loss for all future comparisons. No changes from the working autoresearch setup: B=64, acc=8, gradient_checkpointing=True, cosine LR schedule scaled to 30-min run.

## Changes from Baseline

_Fill in: what was changed vs exp-000-baseline_
