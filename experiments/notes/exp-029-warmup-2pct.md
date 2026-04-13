# exp-029-warmup-2pct

**Description:** 2.5% warmup (shorter)  
**Val loss:** 3.9667  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 07:25  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 8 |

## Reasoning

Baseline uses 5% warmup. exp-001 showed 10% warmup hurt (-0.04 vs baseline). 2.5% hasn't been tested. With the parallel arch and higher LR (5e-4), the model may warm up faster — less warmup means more steps at peak LR. At ~600 optimizer steps per 60-min run, 5% warmup = ~30 steps, 2.5% = ~15 steps. Potentially beneficial if the model doesn't need much warmup at small batch sizes.

## Outcome

**Better than baseline** (3.9667 vs 5.5235, -1.5567 better than baseline).
