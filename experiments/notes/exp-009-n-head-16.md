# exp-009-n-head-16

**Description:** 16 heads instead of 24 (larger head_dim)  
**Val loss:** 5.3510  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 16:51  

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

The baseline uses n_head=24, giving head_dim=52 (1248/24). Reducing to n_head=16 gives head_dim=78, keeping n_embd=1248 and total params roughly the same. Larger head_dim means each head has more capacity to represent its subspace. Some recent work (e.g. MLA, GQA ablations) suggests fewer but wider heads can outperform many narrow heads at this model scale.

## Outcome

**Better than baseline** (5.3510 vs 5.5235, -0.1724 better than baseline).
