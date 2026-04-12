# exp-015-weight-decay-005

**Description:** Weight decay 0.05 instead of 0.1  
**Val loss:** 5.3914  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 18:28  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 24 |

## Reasoning

The baseline uses AdamW weight_decay=0.1. In a 30-minute run (~278 optimizer steps), weight decay acts as an L2 penalty that continuously shrinks weights. A lower value of 0.05 reduces this shrinkage, potentially allowing weights to grow larger and fit the data more aggressively in early training. The baseline value of 0.1 was tuned for long runs; at short scale, lighter regularization may win.

## Outcome

**Better than baseline** (5.3914 vs 5.5235, -0.1321 better than baseline).
