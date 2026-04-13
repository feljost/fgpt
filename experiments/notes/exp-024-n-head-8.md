# exp-024-n-head-8

**Description:** 8 attention heads (head_dim 156)  
**Val loss:** 3.9780  
**Duration:** 60.3 min  
**Seed:** 42  
**Date:** 2026-04-13 03:17  

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

n_head went 24->16 and gave a strong win (exp-017). Testing whether going further to 8 heads (head_dim=156) helps. With GQA already in baseline, n_kv_heads scales proportionally to 4 (maintaining the 2:1 ratio). Larger heads capture longer-range dependencies per head, but fewer heads means less diverse attention patterns. This continues the sweep of the head count vs head_dim tradeoff.

## Outcome

**Better than baseline** (3.9780 vs 5.5235, -1.5455 better than baseline).
