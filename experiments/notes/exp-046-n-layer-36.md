# exp-046-n-layer-36

**Description:** Deeper model: n_layer=36  
**Val loss:** 3.4521  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-14 14:07  

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| AdamW LR | 0.0006 |
| Muon LR | 0.025 |
| Warmup | 2.5% |
| Accum steps | 8 |
| Min LR ratio | 0.05 |
| RoPE base | 50000 |
| n_head | 8 |

## Reasoning

More layers increases model depth and representational capacity. Current best is 32 layers. Adding 4 layers (~12% deeper) keeps parameter count similar to prior depth experiments. Phase 2 exp-026 (deeper+narrower) lost, but that also changed width. This tests depth alone with same n_embd=1248.

## Outcome

**Better than baseline** (3.4521 vs 5.5235, -2.0713 better than baseline).
