# FGPT Autoresearch

Automated research loop for improving the fgpt base model. Each experiment makes one small change, trains for 30 minutes with identical data, and records the final validation loss. Over time, winning ideas compound.

---

## Goal

Improve the base model's validation loss (currently ~2.58 after 45B tokens of full training) by systematically trying small, well-motivated changes to the training setup — learning rates, architecture tweaks, optimizer settings, and regularization — within short 30-minute trial windows.

---

## Research Phases

### Phase 1 — Isolated single-variable tests (exp-000 to exp-016)
Test one change at a time against the same original baseline. Every experiment starts from a fresh model with identical seeds and data. Code changes are always reverted after recording the result — no changes accumulate. The goal is to identify which individual changes are beneficial *at all*, and to build a ranked list of ideas.

**Key findings from Phase 1:**
- **Parallel attn+MLP** (PaLM style): -0.39 val loss vs baseline — dominant win by a large margin
- **n_head=16** (larger heads): -0.17 — strong architectural win
- **GeGLU** (GELU gate in MLP): -0.14 — modest win
- **muon_lr=0.025, adamw_lr=3e-4**: both improved early learning speed
- **qk_norm, logit_softcap, z_loss**: all hurt or were neutral — constraining attention/logit growth is bad at this scale

### Phase 2 — Compounding hill-climbing (exp-017+)
Starting baseline: **parallel attn+MLP + adamw_lr=3e-4 + muon_lr=0.025** (val loss ~5.1336).

Each experiment runs on top of the current best compound config. If it wins, the change is permanently merged and becomes the new baseline for all subsequent experiments. If it loses, it is reverted. This greedy hill-climbing approach finds synergistic combinations that isolated tests miss.

**Phase 2 runs for 60 minutes** (instead of 30) — ~600 optimizer steps rather than ~300, reducing noise and making margins more reliable.

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

### Phase 1 — Isolated experiments (exp-000 to exp-016)

Each experiment tested one variable against the original baseline in isolation. No compounding. Results established which changes are independently beneficial.

| # | Tag | Description | Change | Status |
|---|-----|-------------|--------|--------|
| 0 | `exp-000-baseline` | current config unchanged | None — establishes baseline | ✅ done |
| 1 | `exp-001-warmup-longer` | 10% warmup instead of 5% | `warmup_frac=0.10` | ✅ done |
| 2 | `exp-002-lr-adamw-3e4` | AdamW LR 3e-4 | `adamw_lr=3e-4` | ✅ done |
| 3 | `exp-003-muon-lr-025` | Muon LR higher (0.025) | `muon_lr=0.025` | ✅ done |
| 4 | `exp-004-muon-lr-015` | Muon LR lower (0.015) | `muon_lr=0.015` | ✅ done |
| 5 | `exp-005-rope-base-100k` | RoPE base freq 100k (LLaMA-3 style) | `RotaryEmbedding(base=100000)` | ✅ done |
| 6 | `exp-006-qk-norm` | QK normalization | Add `RMSNorm` on Q and K before attention | ✅ done |
| 7 | `exp-007-parallel-attn-mlp` | Parallel attention + MLP (PaLM style) | Compute attn and MLP in parallel, sum residuals | ✅ done |
| 8 | `exp-008-geglu` | GeGLU instead of SwiGLU | Replace `F.silu` gate with `F.gelu` in MLP | ✅ done |
| 9 | `exp-009-n-head-16` | Fewer, larger heads (head_dim 78) | `n_head=16` (same param count) | ✅ done |
| 10 | `exp-010-logit-softcap` | Soft-cap logits (Gemma 2 style) | `logits = tanh(logits/30)*30` before CE loss | ✅ done |
| 11 | `exp-011-z-loss` | auxiliary logit z-loss | Add `z_loss = 1e-4 * logits.logsumexp(-1).pow(2).mean()` | ✅ done |
| 15 | `exp-015-weight-decay-005` | lower weight decay (0.05) | `weight_decay=0.05` in AdamW | ✅ done |
| 16 | `exp-016-no-weight-tying` | untie lm_head from wte | Separate `lm_head` and `wte` weights | ✅ done |

---

### Phase 2 — Compounding experiments (exp-017+)

**Starting baseline for phase 2:**
- Architecture: **parallel attn+MLP** (exp-007, biggest phase 1 win at 5.1336)
- Optimizers: **adamw_lr=3e-4**, **muon_lr=0.025** (exp-002, exp-003)
- Plus **exp-016 result** if it beats phase 2 baseline

**Rules:** Each experiment runs on top of the current best compound config. If it wins, the change is permanently merged into the baseline for all subsequent experiments. If it loses, it is reverted. The compound baseline val loss is updated after each winning run.

The first two experiments (017, 018) re-validate the other phase 1 architectural winners on the new parallel baseline — their isolated wins may not hold now that the residual structure has changed.

| # | Tag | Description | Change | Rationale |
|---|-----|-------------|--------|-----------|
| 17 | `exp-017-compound-n-head-16` | n_head=16 on parallel baseline | `n_head=16` | Phase 1 rank-2 win (5.3510) — must re-validate; parallel arch changes residual structure |
| 18 | `exp-018-compound-geglu` | GeGLU on parallel baseline | `geglu=True` | Phase 1 rank-3 win (5.3875) — test if it compounds with parallel |
| 19 | `exp-019-muon-lr-030` | Muon LR 0.030 | `muon_lr=0.030` | 0.025 won over 0.020 and 0.015; more expressive arch may need higher LR |
| 20 | `exp-020-adamw-lr-5e4` | AdamW LR 5e-4 | `adamw_lr=5e-4` | 3e-4 won over 2e-4; continue sweep upward on compound baseline |
| 21 | `exp-021-sandwich-norm` | Post-norm after sublayer output (OLMo 2) | `RMSNorm` after parallel output before residual add | OLMo 2 found this very effective; with parallel arch there's only one pre-norm, post-norm may stabilize |
| 22 | `exp-022-diff-attn` | Differential attention (ICLR 2025) | `attn = softmax(Q·K1ᵀ)V1 − λ·softmax(Q·K2ᵀ)V2` | Cancels attention noise by subtracting a second attention head; no constraint on logit scale (unlike qk-norm) |
| 23 | `exp-023-gqa-8kv` | GQA: 8 KV heads | `n_kv_heads=8` (assuming n_head=16 in baseline) | Halve KV heads; reduces memory and adds parameter budget elsewhere; natural complement to n_head=16 |
| 24 | `exp-024-n-head-8` | 8 attention heads (head_dim 156) | `n_head=8` | n_head went 24→16 and improved; test if going further to 8 (head_dim=156) helps |
| 25 | `exp-025-sliding-window` | Alternating local/global attention | Even layers: window=512; odd: full attention | Reduce compute on half the layers; local patterns handled cheaply, global context preserved |
| 26 | `exp-026-deeper-narrower` | 40 layers, n_embd=1120 | `n_layer=40, n_embd=1120` (~same params) | Test depth vs width tradeoff on compound baseline; parallel arch may favour depth more |
| 27 | `exp-027-muon-nesterov` | Muon with Nesterov momentum | `nesterov=True` in Muon | Nesterov look-ahead often improves convergence; zero-overhead optimizer change |
| 28 | `exp-028-adamw-beta2-099` | Slower grad² EMA in AdamW | `betas=(0.9, 0.99)` instead of `(0.9, 0.95)` | More stable second moment; useful when LR is pushed higher (as in compound baseline) |
| 29 | `exp-029-warmup-2pct` | 2.5% warmup (shorter than baseline) | `warmup_frac=0.025` | 10% warmup hurt (exp-001); 5% is baseline; 2.5% may be better for fast-learning parallel arch |
| 30 | `exp-030-batch-larger` | Effective batch 1M tokens | `accumulation_steps=16` | More stable gradient estimates per update; parallel arch has higher per-step throughput |
| 31 | `exp-031-rope-base-20k` | RoPE base freq 20k | `theta=20000` | 100k was marginal; 20k not yet tested; lower than 10k may hurt, worth bracketing |
| 32 | `exp-032-head-dim-64` | head_dim=64 (n_embd=1152, n_head=18) | `n_embd=1152, n_head=18` | Power-of-2 head dim for hardware alignment; slight param increase (~same order) |
| 33 | `exp-033-muon-lr-035` | Muon LR 0.035 | `muon_lr=0.035` | Only run if exp-019 (0.030) wins — continue sweep |
| 34 | `exp-034-adamw-lr-4e4` | AdamW LR 4e-4 | `adamw_lr=4e-4` | Only run if exp-020 (5e-4) loses — fine-tune between 3e-4 and 5e-4 |
| 35 | `exp-035-weight-decay-02` | Higher weight decay (0.2) | `weight_decay=0.2` | Parallel arch is more expressive; stronger regularisation may help generalisation |
| 36 | `exp-036-min-lr-zero` | Fully decay LR to 0 | `min_lr_ratio=0.0` | Baseline keeps 5% floor; aggressive full decay may squeeze out a few more steps of learning |

---

### Phase 3 — Compounding experiments (exp-037+)

**Phase 3 compound baseline** (all phase 1+2 wins merged):
- Architecture: parallel attn+MLP, n_head=8, n_kv_heads=4 (GQA 2:1), head_dim=156
- Optimizers: adamw_lr=5e-4, muon_lr=0.025, beta2=0.99, weight_decay=0.1
- Schedule: warmup_frac=0.025, min_lr_ratio=0.05, rope_base=20000
- **Phase 2 final val loss: 3.9634** (60-min reference)

**Rules:** Same compounding hill-climbing as phase 2. Each experiment adds one change on top of the current best config. Win → merge permanently. Loss → revert.

**Phase 3 runs for 2.5 hours** (~11k microsteps, ~1100 optimizer steps) — longer runs further reduce noise and reveal effects that only emerge with more training.

| # | Tag | Description | Change | Rationale |
|---|-----|-------------|--------|-----------|
| 37 | `exp-037-phase3-baseline` | Phase 3 baseline (no changes) | None — establishes 2.5h reference | Must re-anchor before comparing; 2.5h gives a lower val loss than the 60-min phase 2 baseline |
| 38 | `exp-038-n-head-4` | 4 attention heads (head_dim 312) | `n_head=4, n_kv_heads=2` | 24→16→8 won every step; test if trend continues to n_head=4 |
| 39 | `exp-039-geglu` | GeGLU activation | Replace `F.silu` with `F.gelu` gate in MLP | Won in phase 1 (-0.14) but failed in early phase 2 (tested on old arch). Architecture is now very different — worth retesting |
| 40 | `exp-040-muon-lr-030` | Muon LR 0.030 | `muon_lr=0.030` | Lost narrowly in phase 2 (Δ=0.002). 2.5h run gives cleaner signal — may flip |
| 41 | `exp-041-weight-decay-005` | Lower weight decay (0.05) | `weight_decay=0.05` | Phase 1 showed 0.05 beat 0.1. Never tested on compound baseline |
| 42 | `exp-042-adamw-lr-6e4` | AdamW LR 6e-4 | `adamw_lr=6e-4` | 3e-4 → 5e-4 both won. Continue upward sweep with beta2=0.99 smoothing |
| 43 | `exp-043-rope-base-50k` | RoPE base 50k | `rope_base=50000` | 10k → 20k won narrowly. Test if trend continues above 20k |
| 44 | `exp-044-mqa` | MQA: 1 shared KV head | `n_kv_heads=1` | GQA 16→8→4 KV heads won each step. Test the extreme — single shared KV head |

---

## Results Summary

| Rank | Tag | Val Loss | Description |
|------|-----|----------|-------------|
| 1 | `exp-037-phase3-baseline` | 3.4504 | Phase 3 baseline (2.5h, no changes) |
| 2 | `exp-041-wd-005` | 3.4558 | Lower weight decay 0.05 (vs 0.1 baseline) |
| 3 | `exp-039-geglu` | 3.4646 | GeGLU activation instead of SwiGLU |
| 4 | `exp-040-muon-lr-030` | 3.4740 | Muon LR 0.030 (vs 0.025 baseline) |
| 5 | `exp-031-rope-base-20k` | 3.9634 | RoPE base freq 20k |
| 6 | `exp-036-min-lr-zero` | 3.9664 | Full cosine decay to LR=0 |
| 7 | `exp-029-warmup-2pct` | 3.9667 | 2.5% warmup (shorter) |
| 8 | `exp-035-weight-decay-02` | 3.9741 | Higher weight decay (0.2) |
| 9 | `exp-028-adamw-beta2-099` | 3.9751 | Slower grad² EMA in AdamW (beta2=0.99) |
| 10 | `exp-024-n-head-8` | 3.9780 | 8 attention heads (head_dim 156) |
| 11 | `exp-023-gqa-8kv` | 4.0400 | GQA: 8 KV heads (halved) |
| 12 | `exp-026-deeper-narrower` | 4.1115 | 40 layers, n_embd=1120 (~same params) |
| 13 | `exp-020-adamw-lr-5e4` | 4.1243 | AdamW LR 5e-4 on compound baseline |
| 14 | `exp-017-compound-n-head-16` | 4.2090 | n_head=16 on parallel baseline |
| 15 | `exp-019-muon-lr-030` | 4.2114 | Muon LR 0.030 on compound baseline |
| 16 | `exp-018-compound-geglu` | 4.2176 | GeGLU on parallel+n16 baseline |
| 17 | `exp-021-sandwich-norm` | 4.2302 | Post-norm after sublayer (OLMo 2) |
| 18 | `exp-025-sliding-window` | 4.2402 | Alternating local/global attention |
| 19 | `exp-022-diff-attn` | 4.5714 | Differential attention (ICLR 2025) |
| 20 | `exp-030-batch-larger` | 4.6031 | Effective batch ~1M tokens (accum=16) |
| 21 | `exp-007-parallel-attn-mlp` | 5.1336 | Parallel attn+MLP (PaLM style) |
| 22 | `exp-009-n-head-16` | 5.3510 | 16 heads instead of 24 (larger head_dim) |
| 23 | `exp-016-no-weight-tying` | 5.3643 | Untied lm_head and wte weights |
| 24 | `exp-008-geglu` | 5.3875 | GeGLU instead of SwiGLU |
| 25 | `exp-003-muon-lr-025` | 5.3898 | Muon LR 0.025 (higher) |
| 26 | `exp-015-weight-decay-005` | 5.3914 | Weight decay 0.05 instead of 0.1 |
| 27 | `exp-010-logit-softcap` | 5.3939 | Soft-cap logits at 30 (Gemma 2 style) |
| 28 | `exp-005-rope-base-100k` | 5.4097 | RoPE base 100k (LLaMA-3 style) |
| 29 | `exp-002-lr-adamw-3e4` | 5.4101 | AdamW LR 3e-4 vs 2e-4 |
| 30 | `exp-004-muon-lr-015` | 5.4718 | Muon LR 0.015 (lower) |
| 31 | `exp-000-baseline` | 5.5235 | current config unchanged |
| 32 | `exp-001-warmup-longer` | 5.5596 | 10% warmup instead of 5% |
| 33 | `exp-006-qk-norm` | 5.6449 | QK normalization per-head |
| 34 | `exp-011-z-loss` | 5.7478 | Auxiliary z-loss on logits (1e-4) |

*Auto-updated by `run_experiment.py` after each run — see `experiments/results.jsonl` for full data.*

---

## Methodology Notes

- **Phase 1: 30 minutes** (~300 optimizer steps). Phase 2: **60 minutes** (~600 optimizer steps). Phase 3: **2.5 hours** (~1100 optimizer steps) — longer runs further reduce noise and reveal effects that only emerge with more training. Comparisons are valid as long as they use identical seeds and data — we're measuring *relative improvement*, not absolute final loss.
- **Phase 1 (exp-000–016): isolated tests.** One variable at a time against the original baseline. Changes are always reverted after recording the result — no compounding.
- **Phase 2 (exp-017+): compounding hill-climbing.** Each experiment runs on top of the current best compound config. If it wins, the change is permanently merged into the baseline. If it loses, it is reverted. The compound baseline val loss should be tracked manually after each winning merge.
- **If signal is too noisy**, consider switching to an 85M model config for quick iteration (add it as `FGPTConfigSmall`), then validate winners on the full 600M model.
- **Git tags** mark each experiment's code state. Use `git show <tag>` to see exactly what changed.
- **Revert code changes after each run.** If an experiment requires modifying source files (e.g. model architecture, attention, MLP), those changes **must be reverted** before launching the next experiment if the experiment lost. If the experiment won, the change is kept and becomes part of the permanent baseline.

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
