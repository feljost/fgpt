# exp-026-deeper-narrower

**Description:** 40 layers, n_embd=1120 (~same params)  
**Val loss:** 4.1115  
**Duration:** 60.4 min  
**Seed:** 42  
**Date:** 2026-04-13 05:22  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 8 |

## Reasoning

Test depth vs width tradeoff. 40 layers × n_embd=1120 vs 32 layers × n_embd=1248 — same parameter count (~611M). Deeper networks can learn more compositional hierarchical representations. With the parallel attn+MLP block, extra depth costs less compute per layer than in sequential designs. n_head=8 and n_kv_heads=4 maintained at the same ratio, head_dim=140 (vs 156 in baseline).

## Outcome

**Better than baseline** (4.1115 vs 5.5235, -1.4120 better than baseline).
