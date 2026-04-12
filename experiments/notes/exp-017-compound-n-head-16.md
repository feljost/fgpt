# exp-017-compound-n-head-16

**Description:** n_head=16 on parallel baseline  
**Val loss:** 4.2090  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-12 20:02  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 16 |

## Reasoning

Phase 2 experiment 1. Re-validates the n_head=16 win (5.3510 in phase 1) on the new compound baseline that already includes parallel attn+mlp. With parallel blocks, attn and MLP share one pre-norm and their residuals are summed — this changes the gradient landscape for attention significantly. n_head=16 gives head_dim=78 vs baseline head_dim=52 (n_head=24). The isolated win may not hold on the stronger baseline.

## Outcome

**Better than baseline** (4.2090 vs 5.5235, -1.3145 better than baseline).
