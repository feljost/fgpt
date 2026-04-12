# exp-016-no-weight-tying

**Description:** Untied lm_head and wte weights  
**Val loss:** 5.3643  
**Duration:** 30.4 min  
**Seed:** 42  
**Date:** 2026-04-12 19:00  

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

The baseline ties lm_head and wte (input/output embedding weight sharing, GPT-2 style). Untying gives lm_head its own 63M parameters, letting input and output representations specialize independently. Some work finds untied weights help at larger scales. The tradeoff is more parameters to optimize, but the model may benefit from decoupled input/output token spaces.

## Outcome

**Better than baseline** (5.3643 vs 5.5235, -0.1591 better than baseline).
