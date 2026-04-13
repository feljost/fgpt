# exp-025-sliding-window

**Description:** Alternating local/global attention  
**Val loss:** 4.2402  
**Duration:** 60.4 min  
**Seed:** 42  
**Date:** 2026-04-13 04:20  

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

Alternate even layers (window=512 local attention) with odd layers (full causal attention). Half the layers cheaply handle local patterns; the other half maintain full context. With T=1024, window=512 means even layers see the nearest 512 tokens instead of all 1024. No parameter change — purely a sparsity pattern. This mirrors Mistral's sliding window design. If local patterns dominate lower frequencies of features, this should lose little quality while reducing compute on even layers.

## Outcome

**Better than baseline** (4.2402 vs 5.5235, -1.2832 better than baseline).
