# exp-022-diff-attn

**Description:** Differential attention (ICLR 2025)  
**Val loss:** 4.5714  
**Duration:** 60.5 min  
**Seed:** 42  
**Date:** 2026-04-13 01:13  

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

Differential attention cancels attention noise by computing two softmax attention maps and subtracting a learned-weighted second from the first. Each head has q,k,v from c_attn plus a separate q2,k2 pair from c_attn2 (extra 2*n_embd params). The subtraction suppresses irrelevant context that both heads attend to, sharpening signal-to-noise. No constraint on logit scale unlike qk-norm which hurt badly in phase 1. ICLR 2025 result shows strong improvements on language modeling benchmarks.

## Outcome

**Better than baseline** (4.5714 vs 5.5235, -0.9521 better than baseline).
