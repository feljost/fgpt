# exp-023-gqa-8kv

**Description:** GQA: 8 KV heads (halved)  
**Val loss:** 4.0400  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 02:16  

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

Grouped Query Attention halves the KV heads from 16 to 8. Each KV head is shared by 2 Q heads. This saves ~50M params in KV projections (QKV goes from 3x to 2x n_embd per layer) and reduces KV cache memory. The question is whether the model can maintain representation quality with fewer KV heads, or whether having more independent KV representations per Q head matters. With n_head=16 and head_dim=78, each KV head still has plenty of capacity. GQA is standard in modern LLMs (LLaMA 2+, Mistral) with minimal quality loss vs MHA.

## Outcome

**Better than baseline** (4.0400 vs 5.5235, -1.4834 better than baseline).
