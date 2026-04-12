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
| 0 | `exp-000-baseline` | current config unchanged | None — establishes baseline | ⏳ todo |
| 1 | `exp-001-warmup-longer` | 2% warmup instead 1% | `warmup_frac=0.02` | ⏳ todo |
| 2 | `exp-002-lr-adamw-3e4` | AdamW LR slightly higher | `adamw_lr=3e-4` | ⏳ todo |
| 3 | `exp-003-muon-lr-015` | Muon LR slightly lower | `muon_lr=0.015` | ⏳ todo |
| 4 | `exp-004-rope-base-20k` | higher RoPE base freq | `RotaryEmbedding(base=20000)` | ⏳ todo |
| 5 | `exp-005-qk-norm` | QK normalization | Add RMSNorm on Q and K | ⏳ todo |
| 6 | `exp-006-z-loss` | auxiliary logit z-loss | Add `z_loss = 1e-4 * logits.exp().mean()` | ⏳ todo |
| 7 | `exp-007-batch-larger` | effective batch 1M tokens | `accumulation_steps=24` | ⏳ todo |
| 8 | `exp-008-grad-clip-03` | lower grad clip throughout | `norm_clip=0.3` always | ⏳ todo |
| 9 | `exp-009-weight-decay-05` | half weight decay | `weight_decay=0.05` | ⏳ todo |

---

## Results Summary

| Rank | Tag | Val Loss | Description |
|------|-----|----------|-------------|
| — | *(run exp-000-baseline first)* | — | — |

*Auto-updated by `visualize.py` — see `experiments/results.jsonl` for full data.*

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
