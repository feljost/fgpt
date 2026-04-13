# exp-028-adamw-beta2-099

**Description:** Slower grad² EMA in AdamW (beta2=0.99)  
**Val loss:** 3.9751  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 06:24  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 8 |

## Reasoning

AdamW default beta2=0.95 is aggressive — the second moment estimate decays quickly, making the effective LR noisy on bursty gradients. Slowing it to 0.99 (more common in large model pretraining) smooths the adaptive denominator. This matters more at higher LR (5e-4 compound baseline), where the gradient variance is larger. Potential downside: slower adaptation to gradient scale changes early in training. Note: exp-027 (Nesterov Muon) was skipped — nesterov=True is already the Muon default in this codebase.

## Outcome

**Better than baseline** (3.9751 vs 5.5235, -1.5483 better than baseline).
