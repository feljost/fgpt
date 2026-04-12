# exp-003-muon-lr-025

**Description:** Muon LR 0.025 (higher)  
**Val loss:** 5.3898  
**Duration:** 30.5 min  
**Seed:** 42  
**Date:** 2026-04-12 13:40  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |

## Reasoning

exp-002 showed AdamW LR 3e-4 is better than 2e-4, suggesting the baseline LRs are generally conservative for 30-min runs. Testing Muon LR 0.025 (vs baseline 0.02, +25%) with the winning AdamW LR. Muon handles the bulk of the model weights (all 2D linear layers), so its LR has large leverage. If too high it will destabilise training (watch grad norm); if better it confirms the same conservative-LR pattern.

## Outcome

**Better than baseline** (5.3898 vs 5.5235, -0.1336 better than baseline).
