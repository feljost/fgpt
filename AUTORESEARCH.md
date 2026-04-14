# FGPT Autoresearch

Automated research loop for improving the fgpt base model. Each experiment makes one small change, trains for a fixed wall-clock duration with identical data, and records the final validation loss. Over time, winning ideas compound.

---

## Goal

Improve the base model's validation loss (currently ~2.58 after 45B tokens of full training) by systematically trying small, well-motivated changes to the training setup — learning rates, architecture tweaks, optimizer settings, and regularization.

---

## Research Phases

### Phase 1 — Isolated single-variable tests (exp-000 to exp-016)
Test one change at a time against the same original baseline. Every experiment starts from a fresh model with identical seeds and data. Code changes are always reverted after recording the result — no changes accumulate. The goal is to identify which individual changes are beneficial *at all*, and to build a ranked list of ideas.

**Key findings from Phase 1:**
- **Parallel attn+MLP** (PaLM style): -0.39 val loss vs baseline — dominant win by a large margin
- **n_head=16** (larger heads): -0.17 — strong architectural win (later superseded by n_head=8 in phase 2)
- **GeGLU** (GELU gate in MLP): -0.14 — modest win (did not hold up in phase 2 or 3)
- **muon_lr=0.025, adamw_lr=3e-4**: both improved early learning speed
- **logit_softcap**: modest win in phase 1 (5.3939) — never merged into compound, retesting in phase 3
- **qk_norm, z_loss**: hurt — constraining attention/logit growth is bad at this scale

### Phase 2 — Compounding hill-climbing (exp-017 to exp-036)
Starting baseline: **parallel attn+MLP + adamw_lr=3e-4 + muon_lr=0.025** (val loss ~5.1336).

Each experiment runs on top of the current best compound config. If it wins, the change is permanently merged. If it loses, it is reverted.

**Phase 2 runs for 60 minutes** (~600 optimizer steps).

**Key wins merged from Phase 2:**
- n_head=8, n_kv_heads=4 (GQA 2:1)
- adamw_beta2=0.99
- warmup_frac=0.025
- adamw_lr=5e-4
- rope_base=20000

### Phase 3 — Compounding experiments (exp-037+)
Same hill-climbing approach. **Runs for 2.5 hours** (~11k microsteps, ~1100 optimizer steps).

**Key wins merged from Phase 3:**
- adamw_lr=6e-4 (exp-042)
- rope_base=50000 (exp-043)
- n_kv_heads=1 / MQA (exp-044) — biggest phase 3 win (-0.047)

---

## Current Compound Config (as of exp-044)

| Parameter | Value | Merged after |
|-----------|-------|-------------|
| Architecture | Parallel attn+MLP (PaLM) | exp-007 |
| n_layer | 32 | — |
| n_embd | 1248 | — |
| n_head | 8 | exp-024 |
| n_kv_heads | **1** (MQA) | exp-044 |
| head_dim | 156 | — |
| rope_base | **50000** | exp-043 |
| adamw_lr | **6e-4** | exp-042 |
| muon_lr | 0.025 | exp-003 |
| adamw_beta2 | 0.99 | exp-028 |
| weight_decay | 0.1 | — |
| warmup_frac | 0.025 | exp-029 |
| min_lr_ratio | 0.05 | — |
| accumulation_steps | 8 | — |

**Current best val loss (2.5h run): 3.4038** (exp-044) — queue empty, no active experiments

---

## How It Works

Every experiment:
1. Initializes a fresh model from the same random seed (42)
2. Draws training data in the same deterministic order (seeded shard shuffle)
3. Evaluates on **fixed validation batches** (`experiments/fixed_val_batches.pt`) — identical every run
4. Records results to `experiments/results.jsonl` and creates a git tag
5. Generates a notes file in `experiments/notes/` with reasoning and results

This means every experiment is directly comparable: same model init, same data, same val set.

---

## Environment

Always use the uv-managed venv at `/root/fgpt/.venv`:

```bash
# Activate first:
source /root/fgpt/.venv/bin/activate

# Or use full path directly:
/root/fgpt/.venv/bin/python -m fgpt.autoresearch.run_experiment ...
```

**Never use the system Python** (`/usr/bin/python`) — it doesn't have the required packages.

---

## Model Scale

Experiments always use the **full 600M parameter model** (`n_layer=32, n_head=8, n_embd=1248`). Do not change the number of parameters — the goal is insights applicable to the actual production model.

**Architecture changes are allowed** (e.g. different attention mechanisms, activation functions, normalization placement), as long as the total parameter count stays roughly the same.

---

## One-Time Setup

Generate the fixed val batches (only needs to happen once):

```bash
source /root/fgpt/.venv/bin/activate
python -m fgpt.autoresearch.fixed_val
```

---

## Running an Experiment

```bash
cd /root/fgpt
.venv/bin/python -m fgpt.autoresearch.run_experiment \
    --tag exp-045-logit-softcap \
    --description "Logit soft-cap at 30" \
    --duration-minutes 150 \
    --reasoning "..."
```

All flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--tag` | required | Experiment ID, e.g. `exp-045-logit-softcap` |
| `--description` | required | 3–5 words for the plot label |
| `--duration-minutes` | 150 | Wall-clock training time (phase 3 = 150) |
| `--seed` | 42 | RNG seed for model init + data |
| `--accumulation-steps` | 8 | Gradient accumulation steps |
| `--adamw-lr` | 6e-4 | AdamW learning rate |
| `--muon-lr` | 0.025 | Muon learning rate |
| `--warmup-frac` | 0.025 | Warmup as fraction of total schedule |
| `--rope-base` | 50000 | RoPE base frequency |
| `--n-head` | 8 | Number of attention heads |
| `--n-kv-heads` | 1 | KV heads (1 = MQA) |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--adamw-beta2` | 0.99 | AdamW beta2 |
| `--reasoning` | "" | Explanation written into notes file |

---

## Visualize Progress

```bash
cd /root/fgpt
.venv/bin/python -m fgpt.autoresearch.visualize
```

Output: `experiments/plots/progress.png`

---

## Ideas Queue

### Phase 1 — Isolated experiments (exp-000 to exp-016)

| # | Tag | Description | Status |
|---|-----|-------------|--------|
| 0 | `exp-000-baseline` | current config unchanged | ✅ done |
| 1 | `exp-001-warmup-longer` | 10% warmup | ✅ done |
| 2 | `exp-002-lr-adamw-3e4` | AdamW LR 3e-4 | ✅ done |
| 3 | `exp-003-muon-lr-025` | Muon LR 0.025 | ✅ done |
| 4 | `exp-004-muon-lr-015` | Muon LR 0.015 | ✅ done |
| 5 | `exp-005-rope-base-100k` | RoPE base 100k | ✅ done |
| 6 | `exp-006-qk-norm` | QK normalization | ✅ done |
| 7 | `exp-007-parallel-attn-mlp` | Parallel attn+MLP (PaLM) | ✅ done |
| 8 | `exp-008-geglu` | GeGLU instead of SwiGLU | ✅ done |
| 9 | `exp-009-n-head-16` | 16 attention heads | ✅ done |
| 10 | `exp-010-logit-softcap` | Soft-cap logits at 30 | ✅ done |
| 11 | `exp-011-z-loss` | Auxiliary z-loss | ✅ done |
| 15 | `exp-015-weight-decay-005` | weight_decay=0.05 | ✅ done |
| 16 | `exp-016-no-weight-tying` | Untied lm_head/wte | ✅ done |

---

### Phase 2 — Compounding experiments (exp-017 to exp-036)

| # | Tag | Description | Status |
|---|-----|-------------|--------|
| 17 | `exp-017-compound-n-head-16` | n_head=16 on parallel baseline | ✅ done — lost |
| 18 | `exp-018-compound-geglu` | GeGLU on parallel baseline | ✅ done — lost |
| 19 | `exp-019-muon-lr-030` | Muon LR 0.030 | ✅ done — lost |
| 20 | `exp-020-adamw-lr-5e4` | AdamW LR 5e-4 | ✅ done — **merged** |
| 21 | `exp-021-sandwich-norm` | Post-norm (OLMo 2 style) | ✅ done — lost |
| 22 | `exp-022-diff-attn` | Differential attention | ✅ done — lost |
| 23 | `exp-023-gqa-8kv` | GQA: 8 KV heads | ✅ done — **merged** |
| 24 | `exp-024-n-head-8` | 8 attention heads | ✅ done — **merged** |
| 25 | `exp-025-sliding-window` | Alternating local/global attention | ✅ done — lost |
| 26 | `exp-026-deeper-narrower` | 40 layers, n_embd=1120 | ✅ done — lost |
| 27 | `exp-027-muon-nesterov` | Muon Nesterov=True | ✅ skipped (already default) |
| 28 | `exp-028-adamw-beta2-099` | AdamW beta2=0.99 | ✅ done — **merged** |
| 29 | `exp-029-warmup-2pct` | warmup_frac=0.025 | ✅ done — **merged** |
| 30 | `exp-030-batch-larger` | Effective batch ~1M tokens | ✅ done — lost |
| 31 | `exp-031-rope-base-20k` | rope_base=20000 | ✅ done — **merged** |
| 32 | `exp-032-head-dim-64` | head_dim=64 | ✅ skipped (inapplicable) |
| 33 | `exp-033-muon-lr-035` | Muon LR 0.035 | ✅ skipped (conditional) |
| 34 | `exp-034-adamw-lr-4e4` | AdamW LR 4e-4 | ✅ skipped (conditional) |
| 35 | `exp-035-weight-decay-02` | weight_decay=0.2 | ✅ done — lost |
| 36 | `exp-036-min-lr-zero` | Full cosine decay to LR=0 | ✅ done — lost |

---

### Phase 3 — Compounding experiments (exp-037+)

| # | Tag | Description | Status |
|---|-----|-------------|--------|
| 37 | `exp-037-phase3-baseline` | Phase 3 baseline (2.5h) | ✅ done — 3.4504 (reference) |
| 38 | `exp-038-n-head-4` | n_head=4, n_kv_heads=2 | ✅ done — lost (looked worse at halfway) |
| 39 | `exp-039-geglu` | GeGLU activation | ✅ done — lost (+0.014) |
| 40 | `exp-040-muon-lr-030` | Muon LR 0.030 | ✅ done — lost (+0.024) |
| 41 | `exp-041-wd-005` | weight_decay=0.05 | ✅ done — lost (+0.005) |
| 42 | `exp-042-adamw-lr-6e4` | AdamW LR 6e-4 | ✅ done — **merged** (-0.008) |
| 43 | `exp-043-rope-base-50k` | RoPE base 50k | ✅ done — **merged** (-0.011) |
| 44 | `exp-044-mqa` | MQA: n_kv_heads=1 | ✅ done — **merged** (-0.047) |
| 45 | `exp-045-logit-softcap` | Logit soft-cap at 30 | ✅ done — lost (+0.006) |
| 46 | `exp-046-n-layer-36` | n_layer=36 (deeper model) | ✅ done — lost (+0.048) |

---

## Results Summary

| Rank | Tag | Val Loss | Description |
|------|-----|----------|-------------|
| 1 | `exp-044-mqa` | 3.4038 | MQA: n_kv_heads=1 (extreme GQA) |
| 2 | `exp-045-logit-softcap` | 3.4099 | Logit soft-cap at 30 (tanh) |
| 3 | `exp-043-rope-base-50k` | 3.4391 | RoPE base freq 50k (vs 20k baseline) |
| 4 | `exp-042-adamw-lr-6e4` | 3.4425 | AdamW LR 6e-4 (vs 5e-4 baseline) |
| 5 | `exp-037-phase3-baseline` | 3.4504 | Phase 3 baseline (2.5h, no changes) |
| 6 | `exp-046-n-layer-36` | 3.4521 | Deeper model: n_layer=36 |
| 7 | `exp-041-wd-005` | 3.4558 | Lower weight decay 0.05 (vs 0.1 baseline) |
| 8 | `exp-039-geglu` | 3.4646 | GeGLU activation instead of SwiGLU |
| 9 | `exp-040-muon-lr-030` | 3.4740 | Muon LR 0.030 (vs 0.025 baseline) |
| 10 | `exp-031-rope-base-20k` | 3.9634 | RoPE base freq 20k |
| 11 | `exp-036-min-lr-zero` | 3.9664 | Full cosine decay to LR=0 |
| 12 | `exp-029-warmup-2pct` | 3.9667 | 2.5% warmup (shorter) |
| 13 | `exp-035-weight-decay-02` | 3.9741 | Higher weight decay (0.2) |
| 14 | `exp-028-adamw-beta2-099` | 3.9751 | Slower grad² EMA in AdamW (beta2=0.99) |
| 15 | `exp-024-n-head-8` | 3.9780 | 8 attention heads (head_dim 156) |
| 16 | `exp-023-gqa-8kv` | 4.0400 | GQA: 8 KV heads (halved) |
| 17 | `exp-026-deeper-narrower` | 4.1115 | 40 layers, n_embd=1120 (~same params) |
| 18 | `exp-020-adamw-lr-5e4` | 4.1243 | AdamW LR 5e-4 on compound baseline |
| 19 | `exp-017-compound-n-head-16` | 4.2090 | n_head=16 on parallel baseline |
| 20 | `exp-019-muon-lr-030` | 4.2114 | Muon LR 0.030 on compound baseline |
| 21 | `exp-018-compound-geglu` | 4.2176 | GeGLU on parallel+n16 baseline |
| 22 | `exp-021-sandwich-norm` | 4.2302 | Post-norm after sublayer (OLMo 2) |
| 23 | `exp-025-sliding-window` | 4.2402 | Alternating local/global attention |
| 24 | `exp-022-diff-attn` | 4.5714 | Differential attention (ICLR 2025) |
| 25 | `exp-030-batch-larger` | 4.6031 | Effective batch ~1M tokens (accum=16) |
| 26 | `exp-007-parallel-attn-mlp` | 5.1336 | Parallel attn+MLP (PaLM style) |
| 27 | `exp-009-n-head-16` | 5.3510 | 16 heads instead of 24 (larger head_dim) |
| 28 | `exp-016-no-weight-tying` | 5.3643 | Untied lm_head and wte weights |
| 29 | `exp-008-geglu` | 5.3875 | GeGLU instead of SwiGLU |
| 30 | `exp-003-muon-lr-025` | 5.3898 | Muon LR 0.025 (higher) |
| 31 | `exp-015-weight-decay-005` | 5.3914 | Weight decay 0.05 instead of 0.1 |
| 32 | `exp-010-logit-softcap` | 5.3939 | Soft-cap logits at 30 (Gemma 2 style) |
| 33 | `exp-005-rope-base-100k` | 5.4097 | RoPE base 100k (LLaMA-3 style) |
| 34 | `exp-002-lr-adamw-3e4` | 5.4101 | AdamW LR 3e-4 vs 2e-4 |
| 35 | `exp-004-muon-lr-015` | 5.4718 | Muon LR 0.015 (lower) |
| 36 | `exp-000-baseline` | 5.5235 | current config unchanged |
| 37 | `exp-001-warmup-longer` | 5.5596 | 10% warmup instead of 5% |
| 38 | `exp-006-qk-norm` | 5.6449 | QK normalization per-head |
| 39 | `exp-011-z-loss` | 5.7478 | Auxiliary z-loss on logits (1e-4) |

*Auto-updated by `run_experiment.py` after each run — see `experiments/results.jsonl` for full data.*

---

## Methodology Notes

- **Phase 1: 30 min** (~300 optimizer steps). **Phase 2: 60 min** (~600 steps). **Phase 3: 2.5 hours** (~1100 optimizer steps). Comparisons are only valid within the same phase (same duration).
- **LR schedule scales to run duration** — warmup and cosine decay are expressed as fractions of total steps, so every run sees a full warmup→peak→decay cycle regardless of length.
- **Code changes must be reverted** if an experiment loses. If it wins, the change is kept and becomes part of the permanent baseline.
- **Git tags** mark each experiment's code state. Use `git show <tag>` to see exactly what changed.
- **Things definitively ruled out:** GeGLU (tested 3× across phases), QK-norm, z-loss, sandwich/post-norm, differential attention, sliding window attention, larger batch, full LR decay to 0, logit soft-cap (lost in phase 3 despite winning in phase 1), n_layer=36 (worse than 32 at 2.5h).

---

## File Structure

```
experiments/
├── fixed_val_batches.pt     # 64 fixed val batches — never regenerate
├── results.jsonl            # one JSON line per experiment
├── notes/
│   ├── exp-000-baseline.md
│   └── ...
└── plots/
    └── progress.png

src/fgpt/autoresearch/
├── fixed_val.py             # generate / load fixed val batches
├── run_experiment.py        # CLI experiment runner
└── visualize.py             # plot progress
```
