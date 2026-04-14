# exp-044-mqa

**Description:** MQA: n_kv_heads=1 (extreme GQA)  
**Val loss:** 3.4038  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-14 07:36  

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

Current baseline uses GQA with n_kv_heads=4 (2:1 ratio). MQA (n_kv_heads=1) is the extreme case: all query heads share a single KV head. Reduces KV projection params significantly, potentially acting as a regularizer. Phase 2 found the 2:1 GQA ratio better than MHA — testing whether pushing further to MQA continues that trend.

## Outcome

**Better than baseline** (3.4038 vs 5.5235, -2.1196 better than baseline).
