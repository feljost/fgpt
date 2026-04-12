# exp-011-z-loss

**Description:** Auxiliary z-loss on logits (1e-4)  
**Val loss:** 5.7478  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 17:57  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0003 |
| Muon LR | 0.025 |
| Warmup | 5.0% |
| Accum steps | 16 |
| Min LR ratio | 0.05 |
| RoPE base | 10000 |
| n_head | 24 |

## Reasoning

Z-loss adds a penalty on the log-partition function: z_loss = 1e-4 * logsumexp(logits)^2. This discourages the model from growing logit magnitude unnecessarily, acting as an implicit regularizer on the output distribution. Used in PaLM and other large-scale models to prevent logit drift during training. Run at B=32/accum=16 (same effective batch) to fit z_loss backward within 80 GB GPU memory.

## Outcome

**Worse than baseline** (5.7478 vs 5.5235, +0.2243 worse than baseline).
