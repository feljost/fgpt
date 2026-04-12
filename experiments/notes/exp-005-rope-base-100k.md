# exp-005-rope-base-100k

**Description:** RoPE base 100k (LLaMA-3 style)  
**Val loss:** 5.4097  
**Duration:** 30.5 min  
**Seed:** 42  
**Date:** 2026-04-12 14:43  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 100000 |

## Reasoning

The default RoPE base of 10000 was set for GPT-NeoX. LLaMA-3 uses 500000 and many modern models use 100000+. A higher base stretches the positional encoding period, giving better length generalization and potentially better within-context representations. At 30-minute scale this affects every single token's positional encoding, so the effect should be visible quickly.

## Outcome

**Better than baseline** (5.4097 vs 5.5235, -0.1138 better than baseline).
