# exp-030-batch-larger

**Description:** Effective batch ~1M tokens (accum=16)  
**Val loss:** 4.6031  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 08:26  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0005 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 16 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 8 |

## Reasoning

Doubling accumulation steps from 8 to 16 doubles the effective batch size to ~1M tokens per optimizer update. Larger batches produce lower-variance gradient estimates, which may stabilise training and allow the optimizer to take better steps. Downside: half as many optimizer updates in the same wall-clock time (~300 vs ~600 updates), so the model sees fewer LR schedule steps. Net effect depends on which matters more: gradient quality or update frequency.

## Outcome

**Better than baseline** (4.6031 vs 5.5235, -0.9204 better than baseline).
