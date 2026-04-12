# exp-007-parallel-attn-mlp

**Description:** Parallel attn+MLP (PaLM style)  
**Val loss:** 5.1336  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 15:47  

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

PaLM-style parallel blocks compute attention and MLP on the same normed input and add both residuals in one step. This removes one RMSNorm per block (32 fewer norms total) and can improve throughput by fusing the two matmul paths. The shared input norm may also provide a regularizing effect. Used in PaLM, GPT-J, and several recent efficient architectures.

## Outcome

**Better than baseline** (5.1336 vs 5.5235, -0.3899 better than baseline).
