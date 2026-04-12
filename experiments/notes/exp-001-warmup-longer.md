# exp-001-warmup-longer

**Description:** 10% warmup instead of 5%  
**Val loss:** 5.5596  
**Duration:** 30.5 min  
**Seed:** 42  
**Date:** 2026-04-12 12:37  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0002 |
| Muon LR | 0.02 |
| Warmup | 10.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |

## Reasoning

With only ~278 optimizer updates in 30 minutes, the baseline 5% warmup gives ~14 steps of warmup. Doubling to 10% gives ~28 warmup steps, which should produce a smoother early loss curve and potentially reduce the high initial gradient norms (norm ~2.5-2.8 seen in baseline). The hypothesis is that a gentler LR ramp helps the optimizer find a better trajectory before the cosine decay kicks in.

## Outcome

**Worse than baseline** (5.5596 vs 5.5235, +0.036). With only ~278 optimizer steps total, spending a larger fraction on warmup means less time at peak LR. The baseline 5% warmup is better for short 30-min windows. Hypothesis rejected — do not increase warmup fraction further.
