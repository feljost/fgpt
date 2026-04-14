# exp-043-rope-base-50k

**Description:** RoPE base freq 50k (vs 20k baseline)  
**Val loss:** 3.4391  
**Duration:** 150.3 min  
**Seed:** 42  
**Date:** 2026-04-14 05:05  

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

Current rope_base=20k was merged in exp-031. Testing 50k — higher base frequencies extend effective context length and can improve generalization on longer sequences. LLaMA-3 uses 500k; 50k is a modest step up that may help without destabilizing training.

## Outcome

**Better than baseline** (3.4391 vs 5.5235, -2.0844 better than baseline).
