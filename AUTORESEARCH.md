# FGPT Autoresearch

Automated research loop for improving the fgpt base model. Each experiment makes one small change, trains for 30 minutes with identical data, and records the final validation loss. Over time, winning ideas compound.

---

## Goal

Improve the base model's validation loss (currently ~2.58 after 45B tokens of full training) by systematically trying small, well-motivated changes to the training setup — learning rates, architecture tweaks, optimizer settings, and regularization — within short 30-minute trial windows.

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

Always use the uv-managed venv at `/root/fgpt/.venv`. Either activate it first or prefix commands with the full python path:

```bash
source /root/fgpt/.venv/bin/activate
# or use the full path directly:
/root/fgpt/.venv/bin/python -m fgpt.autoresearch.run_experiment ...
```

The venv is managed by `uv`. To install/sync dependencies: `uv sync` from the repo root.

## Model Scale

Experiments always use the **full 600M parameter model** (`n_layer=32, n_head=24, n_embd=1248`). Do not change the number of parameters — the goal is insights applicable to the actual production model.

**Architecture changes are allowed** (e.g. QK-norm, different attention mechanisms, activation functions, normalization placement), as long as the total parameter count stays roughly the same.

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
cd src/fgpt
python -m fgpt.autoresearch.run_experiment \
    --tag exp-001-warmup-longer \
    --description "2% warmup instead 1%" \
    --duration-minutes 20 \
    --warmup-frac 0.02 \
    --reasoning "Longer warmup may reduce early instability and lead to a smoother loss descent."
```

All flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--tag` | required | Experiment ID, e.g. `exp-001-warmup-longer` |
| `--description` | required | 3–5 words for the plot label |
| `--duration-minutes` | 30 | Wall-clock training time |
| `--seed` | 42 | RNG seed for model init + data |
| `--accumulation-steps` | 12 | Gradient accumulation steps |
| `--adamw-lr` | 2e-4 | AdamW learning rate |
| `--muon-lr` | 0.02 | Muon learning rate |
| `--warmup-frac` | 0.01 | Warmup as fraction of total schedule |
| `--reasoning` | "" | Explanation written into notes file |

---

## Visualize Progress

```bash
cd src/fgpt
python -m fgpt.autoresearch.visualize
```

Output: `experiments/plots/progress.png` — val loss annotated with tag and description.

---

## Ideas Queue

Experiments are ordered by expected impact and ease. After each run, update the status column.

| # | Tag | Description | Change | Status |
|---|-----|-------------|--------|--------|
| 0 | `exp-000-baseline` | current config unchanged | None — establishes baseline | ✅ done |
| 1 | `exp-001-warmup-longer` | 10% warmup instead of 5% | `warmup_frac=0.10` | ✅ done |
| 2 | `exp-002-lr-adamw-3e4` | AdamW LR 3e-4 | `adamw_lr=3e-4` | 🔄 running |
| 3 | `exp-003-muon-lr-025` | Muon LR higher (0.025) | `muon_lr=0.025` | ⏳ todo |
| 4 | `exp-004-muon-lr-015` | Muon LR lower (0.015) | `muon_lr=0.015` | ⏳ todo |
| 5 | `exp-005-rope-base-100k` | RoPE base freq 100k (LLaMA-3 style) | `RotaryEmbedding(base=100000)` | ⏳ todo |
| 6 | `exp-006-qk-norm` | QK normalization | Add `RMSNorm` on Q and K before attention | ⏳ todo |
| 7 | `exp-007-parallel-attn-mlp` | Parallel attention + MLP (PaLM style) | Compute attn and MLP in parallel, sum residuals | ⏳ todo |
| 8 | `exp-008-geglu` | GeGLU instead of SwiGLU | Replace `F.silu` gate with `F.gelu` in MLP | ⏳ todo |
| 9 | `exp-009-n-head-16` | Fewer, larger heads (head_dim 78) | `n_head=16` (same param count) | ⏳ todo |
| 10 | `exp-010-logit-softcap` | Soft-cap logits (Gemma 2 style) | `logits = tanh(logits/30)*30` before CE loss | ⏳ todo |
| 11 | `exp-011-z-loss` | auxiliary logit z-loss | Add `z_loss = 1e-4 * logits.logsumexp(-1).pow(2).mean()` | ⏳ todo |
| 12 | `exp-012-min-lr-01` | Higher min LR ratio (0.1 vs 0.05) | `min_lr_ratio=0.1` — less aggressive decay | ⏳ todo |
| 13 | `exp-013-plateau-10pct` | 10% plateau at peak LR | `plateau_frac=0.10` before cosine decay | ⏳ todo |
| 14 | `exp-014-grad-clip-03` | lower grad clip throughout | `norm_clip=0.3` always (vs 0.5/1.0 schedule) | ⏳ todo |
| 15 | `exp-015-weight-decay-005` | lower weight decay (0.05) | `weight_decay=0.05` in AdamW | ⏳ todo |
| 16 | `exp-016-no-weight-tying` | untie lm_head from wte | Separate `lm_head` and `wte` weights | ⏳ todo |
| 17 | `exp-017-deeper-narrower` | 40 layers, n_embd=1120 | `n_layer=40, n_embd=1120` (~same params) | ⏳ todo |
| 18 | `exp-018-batch-larger` | effective batch 1M tokens | `accumulation_steps=16` (~1M tokens/update) | ⏳ todo |
| 19 | `exp-019-adamw-beta2-099` | slower grad² EMA in AdamW | `betas=(0.9, 0.99)` instead of `(0.9, 0.95)` | ⏳ todo |
| 20 | `exp-020-rope-base-20k` | RoPE base freq 20k | `RotaryEmbedding(base=20000)` | ⏳ todo |

---

## Results Summary

| Rank | Tag | Val Loss | Description |
|------|-----|----------|-------------|
| 1 | `exp-000-baseline` | 5.5235 | current config unchanged |
| 2 | `exp-001-warmup-longer` | 5.5596 | 10% warmup instead of 5% |

*Auto-updated by `run_experiment.py` after each run — see `experiments/results.jsonl` for full data.*

---

## Methodology Notes

- **30 minutes from scratch** gives ~300 optimizer steps for the 600M model. Loss at this point is in the 5–7 range (still early). Comparisons are valid as long as they use identical seeds and data — we're measuring *relative improvement*, not absolute final loss.
- **Small changes only.** One variable at a time. If a run beats baseline, merge it into the baseline config for the next experiment.
- **If signal is too noisy**, consider switching to an 85M model config for quick iteration (add it as `FGPTConfigSmall`), then validate winners on the full 600M model.
- **Git tags** mark each experiment's code state. Use `git show <tag>` to see exactly what changed.

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
