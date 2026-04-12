# exp-010-logit-softcap

**Description:** Soft-cap logits at 30 (Gemma 2 style)  
**Val loss:** 5.3939  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 17:22  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 24 |

## Reasoning

Gemma 2 applies tanh(logits/30)*30 before the cross-entropy loss, soft-capping logit magnitude. This prevents logit blow-up during early training when the model may produce very large unnormalized scores, stabilizing the loss surface. The tanh squashes extreme values while leaving moderate logits nearly unchanged (tanh(x)~x for small x).

## Outcome

**Better than baseline** (5.3939 vs 5.5235, -0.1296 better than baseline).
