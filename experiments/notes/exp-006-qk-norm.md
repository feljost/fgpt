# exp-006-qk-norm

**Description:** QK normalization per-head  
**Val loss:** 5.6449  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 15:14  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |

## Reasoning

QK normalization applies RMSNorm to Q and K per-head before attention, preventing attention logit blow-up as training progresses. Used in Chameleon, PaLM-2, and recent stabilized transformers. At short 30-min runs the model is still in early training where logit norms grow fast — QK norm may help by keeping attention distributions sharper and more uniform across heads.

## Outcome

**Worse than baseline** (5.6449 vs 5.5235, +0.1214 worse than baseline).
