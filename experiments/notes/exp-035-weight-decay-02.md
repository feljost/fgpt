# exp-035-weight-decay-02

**Description:** Higher weight decay (0.2)  
**Val loss:** 3.9741  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 10:28  

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

Baseline uses weight_decay=0.1. With a more expressive compound architecture (parallel block, n_head=8, GQA, higher LR), the model may benefit from stronger regularisation. Phase 1 showed lower weight decay (0.05) was marginally better in isolation, but the compound baseline is now more complex. Stronger decay (0.2) may prevent overfitting to early training data and improve generalisation. This is a simple test — if it wins, it also suggests the model is slightly over-parameterised relative to the data seen in 60 min.

## Outcome

**Better than baseline** (3.9741 vs 5.5235, -1.5493 better than baseline).
