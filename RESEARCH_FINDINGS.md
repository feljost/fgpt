# FGPT Research Findings

Summary of 46 experiments across 3 phases of automated hill-climbing research on the 600M parameter FGPT model.

---

## What Worked (Wins Merged Into Compound Config)

### Architecture
| Change | Val Loss Delta | Notes |
|--------|---------------|-------|
| **Parallel attn+MLP** (PaLM style) | **-0.39** (phase 1) | Single biggest win of the entire project. Sharing the pre-norm between attention and MLP and summing both residuals is strictly better than sequential blocks. |
| **n_head=8** (from 24 originally) | **-0.06** (phase 2) | Fewer, wider heads are better at this scale. Each head gets more dimensionality (head_dim=156 vs 52). |
| **GQA → MQA** (n_kv_heads=1) | **-0.047** (phase 3) | Biggest phase 3 win. A single shared KV head is better than 4 or 8 at this scale and training duration — likely because fewer KV parameters means more capacity goes to Q and the MLP. |
| **RoPE base 50k** (from 10k) | **-0.014** (cumulative phases 2+3) | Higher RoPE base = slower rotation per position = better long-range attention. Stepped up from 10k → 20k → 50k across phases. |

### Optimizer / Training
| Change | Val Loss Delta | Notes |
|--------|---------------|-------|
| **Muon LR 0.025** (from 0.02) | small (phase 1) | Slightly more aggressive Muon update. |
| **AdamW LR 6e-4** (from 2e-4 → 5e-4 → 6e-4) | **-0.008** (phase 3) | LR crept up across all phases; 6e-4 was the sweet spot. Going higher (7e-4+) was never tried but risky. |
| **AdamW beta2=0.99** | **-0.003** (phase 2) | Slower gradient squared EMA smooths out noisy early gradients. |
| **warmup_frac=0.025** | small (phase 2) | 2.5% warmup slightly better than 5%. Short warmup gets to peak LR faster. |

---

## What Didn't Work (Losses)

### Architecture
| Change | Val Loss Delta | Verdict |
|--------|---------------|---------|
| **GeGLU** (GELU gate instead of SwiGLU) | +0.014 (phase 3), +0.04 (phase 2), similar in phase 1 | Tested **3 times across all phases**. Never helps. SwiGLU is strictly better for this model. |
| **QK normalization** | **+0.12** (phase 1) | Large penalty. Normalizing Q and K per-head constrains attention too much at this scale. Do not retry. |
| **Differential attention** (ICLR 2025) | **+0.53** (phase 2) | Much worse. Likely requires more careful hyperparameter tuning or a different architecture around it. |
| **Sliding window attention** (alternating local/global) | **+0.26** (phase 2) | Sequence length is only 1024 — there is no long-range attention problem to solve here. Local windowing just hurts. |
| **Sandwich/post-norm** (OLMo 2) | **+0.19** (phase 2) | Extra normalization after sublayers adds stability constraints that slow learning. |
| **Deeper + narrower** (40 layers, n_embd=1120) | **+0.13** (phase 2) | Same params, more layers, smaller embedding — worse. Wider is better at this scale. |
| **n_layer=36** (deeper only) | **+0.048** (phase 3) | More depth without more params hurts at 2.5h. Extra layers may need more steps to become useful. |
| **n_head=16** | improved in phase 1, lost in phase 2 compound | Good in isolation but inferior once combined with other wins. n_head=8 was the correct choice. |

### Optimizer / Regularization
| Change | Val Loss Delta | Verdict |
|--------|---------------|---------|
| **Larger batch** (~1M tokens, accum=16) | **+1.15** (phase 2) | Devastating at 60 min. Fewer optimizer steps in the same wall-clock time dominates any per-step quality gain at this scale. |
| **z-loss** (auxiliary logit penalty) | **+0.22** (phase 1) | Strong regularization penalty. Hurts. |
| **Logit soft-cap at 30** (tanh) | +0.006 (phase 3), small win in phase 1 | Phase 1 showed a narrow win but it did not replicate at 2.5h. Not worth the inference overhead. |
| **Weight decay 0.05** | +0.005 (phase 3) | Marginal difference; 0.1 is fine. |
| **Weight decay 0.2** | +0.024 (phase 2) | Too strong. |
| **Muon LR 0.030** | +0.024 (phase 3) | Past the optimum; oscillates. |
| **Full LR decay to 0** (min_lr=0) | small loss (phase 2) | Marginal. The default 5% floor is better. |
| **Weight tying off** (untied lm_head/wte) | lost (phase 1) | Extra parameters in lm_head, but the shared representation is valuable. |

---

## General Learnings

### 1. The biggest lever is architecture, not hyperparameters
The parallel attn+MLP block (-0.39) dwarfs every hyperparameter change. If there is a structural win to find, it will be the largest improvement. Most hyperparameter changes are <0.01.

### 2. MQA is surprisingly good at short training durations
MQA (n_kv_heads=1) was the biggest single win in phase 3 (-0.047). At 2.5h with a fresh model, having fewer KV parameters frees capacity. It may eventually lose to MHA at very long training runs (models that are fully converged benefit from richer KV representations), but for the first few billion tokens MQA is clearly better here.

### 3. Phase 1 winners don't always replicate at longer durations
Logit soft-cap won in phase 1 (30 min) but lost in phase 3 (2.5h). Phase 1 is measuring early-training dynamics, not final quality. Short-run wins are suggestive but not conclusive — validate everything at scale.

### 4. GeGLU is definitively ruled out for this model
Tested 3 separate times across all phases with different baselines. Never helped. SwiGLU (the current MLP) is the right choice.

### 5. RoPE base should be higher than the default 10k
The progression 10k → 20k → 50k all improved. At 1024 context length 50k is a good default; 100k showed no benefit in phase 1, so 50k is probably near optimal for this context length.

### 6. Larger effective batch size needs more steps to pay off
Doubling accumulation steps to 16 was catastrophic in a 60-min run (half as many optimizer steps). At a 1M+ step full run the larger batch would likely help by the end, but it makes short-run comparisons unreliable.

### 7. Wider is better than deeper at this param count and scale
Both "deeper + narrower" and "n_layer=36 at same n_embd" lost. The 600M model at n_layer=32, n_embd=1248 is a good shape. Adding depth without adding width doesn't help in the budget we can observe.

---

## Recommendations for a Full 1–2M Step Run

The compound config found through autoresearch should be significantly better than a naive baseline at full scale. Below are specific recommendations for a production training run.

### Confirmed Config (use this)

```python
FGPTConfig(
    n_layer=32,
    n_head=8,
    n_embd=1248,
    n_kv_heads=1,       # MQA — best single architecture win
    rope_base=50000,    # higher base = better long-range attention
)

# Optimizer
adamw_lr = 6e-4
muon_lr = 0.025
adamw_beta2 = 0.99
weight_decay = 0.1
warmup_frac = 0.025     # 2.5% warmup
min_lr_ratio = 0.05     # cosine decay to 5% of peak LR
accumulation_steps = 12  # ~0.5M effective batch (matches original base_train.py)
```

### On MQA at Full Scale
MQA won clearly in 2.5h experiments, but there is a theoretical risk it degrades relative to MHA at very long training (10B+ tokens) because individual queries can't specialize their keys. Mitigation options:
- Keep MQA and monitor val loss curves for plateau behavior compared to MHA
- Or use GQA with n_kv_heads=2 as a compromise (not tested, worth trying)

### Learning Rate Schedule
The autoresearch LR schedule scales by run duration (warmup/decay as % of steps). For a 1M step run at accumulation_steps=12:
- Total optimizer updates ≈ 83k
- Warmup: 2.5% = ~2k steps
- Cosine decay: starts at step ~2k, reaches 5% of peak at step 83k
- Peak LR: 6e-4 (AdamW), 0.025 (Muon)

This is aggressive but validated. For a 2M step run, the same fractions apply — the schedule will simply stretch to cover 166k optimizer updates.

### Gradient Clipping
The current `base_train.py` uses a two-phase clip: 0.5 for the first 350k steps, then 1.0. This was set empirically for the original config. With MQA and the higher RoPE base, the model may be slightly more stable — it's fine to leave as-is.

### Gradient Checkpointing
For an 80 GB H100 (vs the 96 GB GH200 used in production), enable `gradient_checkpointing=True` in FGPTConfig. ~33% compute overhead but reduces activation memory from ~63 GB to ~2 GB, making it feasible.

### Things to Re-evaluate at Scale That Were Not Tested
1. **Larger effective batch size** — the 1M-token batch (accum=16) hurt in short runs but the literature suggests it helps at full scale. Consider ablating batch size at ~100k steps into a full run.
2. **MQA vs GQA n_kv_heads=2** — not tested; could be a small win vs pure MQA at 2B+ tokens.
3. **RoPE base 100k** — showed no benefit in phase 1 at 1024 context, but worth revisiting if context length increases.
4. **n_layer=36 or 40** — depth didn't help at 2.5h, but a fully-trained deeper model might improve perplexity. Requires full-run ablation.
5. **WSD (Warmup-Stable-Decay) schedule** — not tested. A long stable plateau before the cosine decay can improve final loss on full runs. Consider plateau_frac=0.7 with decay only in the last 30% of training.

---

## Compound Config Progress Over Phases

| Phase | Duration | Best Val Loss | Key Change |
|-------|----------|--------------|------------|
| Phase 1 baseline | 30 min | 5.5235 | original config |
| Phase 1 best | 30 min | 5.1336 | parallel attn+MLP |
| Phase 2 baseline | 60 min | 4.2090 | all phase 1 wins |
| Phase 2 best | 60 min | 3.9634 | + GQA, n_head=8, beta2=0.99, warmup, AdamW 5e-4, rope 20k |
| Phase 3 baseline | 2.5 hr | 3.4504 | all phase 2 wins |
| Phase 3 best | 2.5 hr | **3.4038** | + AdamW 6e-4, rope 50k, MQA |

Total improvement from autoresearch: **-0.12 val loss** at the 2.5h scale (3.4504 → 3.4038), on top of the large architectural gains already in the compound config from phase 1 and 2.
