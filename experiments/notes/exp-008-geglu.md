# exp-008-geglu

**Description:** GeGLU instead of SwiGLU  
**Val loss:** 5.3875  
**Duration:** 30.5 min  
**Seed:** 42  
**Date:** 2026-04-12 16:19  

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

SwiGLU uses SiLU (sigmoid linear unit) as the gate activation; GeGLU uses GELU instead. Both are gated MLP variants but GELU has slightly different curvature near zero. Some work (e.g. GLU Variants paper, Noam Shazeer) finds GeGLU slightly outperforms SwiGLU on language modeling. This is a minimal one-line change with zero parameter count difference.

## Outcome

**Better than baseline** (5.3875 vs 5.5235, -0.1360 better than baseline).
