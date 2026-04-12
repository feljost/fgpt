# exp-018-compound-geglu

**Description:** GeGLU on parallel+n16 baseline  
**Val loss:** 4.2176  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-12 21:03  

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

Phase 2 experiment 2. Tests GeGLU (GELU gate instead of SiLU) on the compound baseline of parallel attn+MLP + n_head=16 (val_loss=4.2090). In phase 1, GeGLU won at 5.3875 in isolation. Now testing whether it compounds. GELU has slightly different curvature near zero compared to SiLU; the difference may be amplified or diminished on the stronger parallel baseline.

## Outcome

**Better than baseline** (4.2176 vs 5.5235, -1.3059 better than baseline).
