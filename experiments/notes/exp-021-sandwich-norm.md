# exp-021-sandwich-norm

**Description:** Post-norm after sublayer (OLMo 2)  
**Val loss:** 4.2302  
**Duration:** 60.4 min  
**Seed:** 42  
**Date:** 2026-04-13 00:07  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 16 |

## Reasoning

Phase 2 experiment 5. OLMo 2 found post-norm (norming sublayer output before residual add) significantly improved training stability and final loss. On the parallel block, this means applying RMSNorm to (attn_out + mlp_out) before adding to the residual stream. With adamw_lr=5e-4 (compound baseline now 4.1243), the slightly higher LR might benefit from the extra stabilization that post-norm provides.

## Outcome

**Better than baseline** (4.2302 vs 5.5235, -1.2932 better than baseline).
