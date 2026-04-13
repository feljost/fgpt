# exp-040-muon-lr-030

**Description:** Muon LR 0.030 (vs 0.025 baseline)  
**Val loss:** 3.4740  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-13 21:33  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.03 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 20000 |
| n_head | 8 |

## Reasoning

Phase 2 baseline uses muon_lr=0.025 (merged from earlier experiments). Testing a slightly higher Muon LR of 0.030 — Muon handles larger LRs well due to its Newton-Schulz orthogonalization. If the optimizer is slightly under-stepped at 0.025, 0.030 could drive faster convergence.

## Outcome

**Better than baseline** (3.4740 vs 5.5235, -2.0494 better than baseline).
