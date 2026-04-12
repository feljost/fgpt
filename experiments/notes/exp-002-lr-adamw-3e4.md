# exp-002-lr-adamw-3e4

**Description:** AdamW LR 3e-4 vs 2e-4  
**Val loss:** 5.4101  
**Duration:** 30.6 min  
**Seed:** 42  
**Date:** 2026-04-12 13:08  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.02 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |

## Reasoning

The baseline uses AdamW LR=2e-4, which was tuned for a 1M-step run. In a 30-minute window (~278 optimizer steps) the schedule decays aggressively, so a slightly higher peak LR could drive faster early learning. 3e-4 is a common sweet spot for AdamW on transformer pretraining. If this helps, it suggests the baseline LR is slightly conservative for short runs.

## Outcome

**Better than baseline** (5.4101 vs 5.5235, -0.1134 better than baseline). Clear win. AdamW LR 3e-4 drives significantly faster early learning than 2e-4. The baseline LR was tuned for 1M steps; at 30-min scale a higher peak LR is better. AdamW LR 3e-4 should become the new default.
