# exp-045-logit-softcap

**Description:** Logit soft-cap at 30 (tanh)  
**Val loss:** 3.4099  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-14 11:36  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0006 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 50000 |
| n_head | 8 |

## Reasoning

exp-010 (phase 1) showed logit soft-cap winning vs baseline (5.3939 vs 5.5235) but it was never merged into the compound. Gemini uses soft-capping to stabilize large logit magnitudes. Testing at phase 3 scale with the full compound config to see if it holds up.

## Outcome

**Better than baseline** (3.4099 vs 5.5235, -2.1136 better than baseline).
