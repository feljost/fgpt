# exp-037-phase3-baseline

**Description:** Phase 3 baseline (2.5h, no changes)  
**Val loss:** 3.4504  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-13 14:05  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 20000 |
| n_head | 8 |

## Reasoning

Establishes the phase 3 reference point. All phase 1+2 compound wins are baked into the defaults: parallel attn+MLP, n_head=8, n_kv_heads=4, adamw_lr=5e-4, beta2=0.99, warmup_frac=0.025, rope_base=20000. No additional changes. The 2.5h run gives ~1100 optimizer steps (vs ~600 in phase 2), reducing noise and lowering val loss compared to the 60-min phase 2 baseline of 3.9634. All subsequent phase 3 experiments compare against this.

## Outcome

**Better than baseline** (3.4504 vs 5.5235, -2.0730 better than baseline).
