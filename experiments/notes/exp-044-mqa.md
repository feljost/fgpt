# exp-044-mqa — Best Compound Config (val_loss 3.4038)

**This is the best result produced by the autoresearch loop.** It is the
phase-3 compound config (`exp-037`) plus three merged phase-3 wins: AdamW
LR 6e-4 (`exp-042`), RoPE base 50000 (`exp-043`), and MQA / `n_kv_heads=1`
(`exp-044` itself). This file is self-contained — it includes every code
diff and command needed to reproduce the result starting from `main`,
without needing the `autoresearch` branch.

| | |
|---|---|
| **Tag** | `exp-044-mqa` |
| **Val loss (fixed val batches)** | **3.4038** |
| **Baseline for comparison** | `exp-000-baseline` (original `main` config, unmodified): 5.5235 |
| **Improvement** | -2.1196 (-38%) over the unmodified `main` config, at the same 150-min wall-clock budget |
| **Run duration** | 150.3 min wall-clock (~11k microsteps, ~1100 optimizer steps) |
| **Seed** | 42 |
| **Base commit (autoresearch branch)** | `9d18b2f` |

---

## 1. What changed, and why (lineage)

The autoresearch loop runs a hill-climbing search: each experiment changes
one thing on top of the current best config; if val loss improves, the
change is kept permanently, otherwise it's reverted. `exp-044`'s config is
the sum of every win merged through phase 1, 2, and 3:

| Win | Merged at | Δ val loss | What it is |
|---|---|---|---|
| Parallel attn+MLP (PaLM block) | `exp-007` | **-0.39** | Attention and MLP share one pre-norm; their outputs are summed into the residual instead of being applied sequentially. By far the largest single win in the whole project. |
| GQA → `n_kv_heads=8` | `exp-023` | -0.08 | First step of grouped-query attention. |
| `n_head=8` (from 24) | `exp-024` | -0.062 | Fewer, wider heads (head_dim 156 instead of 52). |
| GQA → `n_kv_heads=4` (2:1 ratio) | `exp-024` | (combined with above) | Halved KV heads again. |
| `adamw_beta2=0.99` (from 0.95) | `exp-028` | -0.003 | Slower gradient² EMA, smooths noisy early gradients. |
| `warmup_frac=0.025` (from 0.05) | `exp-029` | -0.008 | Shorter warmup reaches peak LR faster. |
| `rope_base=20000` (from 10000) | `exp-031` | -0.003 | Slower RoPE rotation per position. |
| `adamw_lr=6e-4` (from 5e-4) | `exp-042` | -0.008 | LR crept up across all 3 phases; this is the sweet spot found so far. |
| `rope_base=50000` (from 20000) | `exp-043` | -0.011 | Stepped up again; 100k showed no further benefit in phase 1. |
| **`n_kv_heads=1` (MQA, from 4)** | **`exp-044`** | **-0.047** | **Biggest phase-3 win.** A single shared KV head beats 4 or 8 at this scale/duration — fewer KV params frees capacity for Q and the MLP. |

Full experiment-by-experiment detail (including everything that was
*tried and reverted*, e.g. GeGLU, QK-norm, z-loss, differential attention,
sliding window, sandwich norm) is in `RESEARCH_FINDINGS.md` and
`experiments/notes/exp-*.md` on the `autoresearch` branch.

---

## 2. Final config (everything you need)

### Architecture (`FGPTConfig`)

| Parameter | `main` (original) | exp-044 value |
|---|---|---|
| `n_layer` | 32 | 32 (unchanged) |
| `n_embd` | 1248 | 1248 (unchanged) |
| Block structure | Sequential (attn then MLP, separate norms) | **Parallel** (shared norm, summed residuals) |
| `n_head` | 24 | **8** |
| `n_kv_heads` | n/a (full MHA) | **1 (MQA)** |
| `rope_base` | n/a (library default 10000) | **50000** |

### Optimizer / schedule

| Parameter | `main` (original) | exp-044 value |
|---|---|---|
| AdamW LR | 2e-4 | **6e-4** |
| Muon LR | 0.02 | **0.025** |
| AdamW beta2 | 0.95 | **0.99** |
| Weight decay | 0.1 | 0.1 (unchanged) |
| Warmup fraction | 1% (hardcoded `total_updates * 0.01`) | **2.5%** |
| `min_lr_ratio` (cosine floor) | 0.05 | 0.05 (unchanged) |
| `accumulation_steps` | 12 (B=40 → ~491,520 tok/step) | 12 for production (unchanged); autoresearch comparison runs used 8 with B=64 → ~524,288 tok/step, a wash within noise — **use 12/B=40 for production**, it's the better-tested production batch size |

---

## 3. Code changes required in `main`

Two files change: `src/fgpt/model.py` and `src/fgpt/base_train.py`. Both
diffs below apply cleanly to `main` at commit `441ad37`.

### 3.1 `src/fgpt/model.py`

```diff
diff --git a/src/fgpt/model.py b/src/fgpt/model.py
index 5baad49..2f6988d 100644
--- a/src/fgpt/model.py
+++ b/src/fgpt/model.py
@@ -1,7 +1,8 @@
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 import torch
 import torch.nn as nn
 from torch.nn import functional as F
+from torch.utils.checkpoint import checkpoint as grad_checkpoint
 from rotary_embedding_torch import RotaryEmbedding
 
 B = 40  # batch size
@@ -15,10 +16,25 @@ class FGPTConfig:
         50304  # GPT-2's vocab size 50257 --> set to power of 2 for faster cuda
     )
     n_layer: int = 32
-    n_head: int = 24
+    n_head: int = 8  # merged after exp-024: n_head=8 → 3.9780 (-0.062 vs 4.0400 baseline)
     n_embd: int = (
         1248  # embedding dimension -> number of features in each token embedding
     )
+    # Recompute activations during backward instead of storing all 32 layers.
+    # Reduces activation memory from ~63 GB to ~2 GB at ~33% compute cost.
+    # Enable for autoresearch runs on 80 GB H100 (production used 96 GB GH200).
+    gradient_checkpointing: bool = False
+    # RoPE base frequency. Default 10000 matches GPT-NeoX / original RoPE paper.
+    # LLaMA-3 uses 500000; common alternatives are 20000, 100000.
+    # Merged after exp-031: rope_base=20000 → 3.9634 (-0.003 vs 3.9667 baseline).
+    # Merged after exp-043: rope_base=50000 → 3.4391 (-0.011 vs 3.4504 baseline).
+    rope_base: int = 50000
+    # Grouped Query Attention: number of KV heads. Must divide n_head evenly.
+    # Set equal to n_head for standard MHA. Set to 1 for MQA.
+    # Merged after exp-023: GQA n_kv_heads=8 → 4.0400 (-0.08 vs 4.1243 baseline).
+    # Merged after exp-024: n_kv_heads=4 (2:1 ratio with n_head=8) → 3.9780 (-0.062 vs 4.0400).
+    # Merged after exp-044: MQA n_kv_heads=1 → 3.4038 (-0.047 vs 3.4504 baseline).
+    n_kv_heads: int = 1
 
 
 class CausalSelfAttention(nn.Module):
@@ -30,41 +46,46 @@ class CausalSelfAttention(nn.Module):
 
         # Key parameters
         self.n_head = config.n_head
+        self.n_kv_heads = config.n_kv_heads
         self.n_embd = config.n_embd
         self.head_dim = config.n_embd // config.n_head
+        assert config.n_head % config.n_kv_heads == 0
+        self.n_groups = config.n_head // config.n_kv_heads
 
-        # We combine Key, Query, and Value into a single linear layer for efficiency
-        # This replaces the internal mechanics of nn.MultiheadAttention
-        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
+        kv_dim = config.n_kv_heads * self.head_dim
+        # Q projects to n_embd; K+V each project to kv_dim (= n_embd when n_kv_heads == n_head)
+        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * kv_dim, bias=False)
 
         # Output projection
         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
 
         # rotary embedding object
-        self.rotary_emb = RotaryEmbedding(dim = self.head_dim)
+        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, theta=config.rope_base)
 
     def forward(self, x):
         B, T, C = x.size()
 
         # 1. Calculate Query, Key, Value
-        # Result of c_attn is (B, T, 3 * C)
+        kv_dim = self.n_kv_heads * self.head_dim
         qkv = self.c_attn(x)
 
-        # Split into q, k, v -> Each is (B, T, C)
-        q, k, v = qkv.split(self.n_embd, dim=2)
+        # Split into q (n_embd), k (kv_dim), v (kv_dim)
+        q, k, v = qkv.split([self.n_embd, kv_dim, kv_dim], dim=2)
 
-        # 2. Reshape for Multi-head attention
-        # We need to transform (B, T, C) -> (B, n_head, T, head_dim)
-        # The 'transpose' is physically moving memory, putting heads in the 2nd dimension
-        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
-        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
-        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
+        # 2. Reshape for multi-head / grouped-query attention
+        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, nh, T, hs)
+        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)
+        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)
 
         # 3. Apply rotary embeddings to Q and K
-        # rotary_embedding_torch expects shape (B, n_head, T, head_dim)
         q = self.rotary_emb.rotate_queries_or_keys(q)
         k = self.rotary_emb.rotate_queries_or_keys(k)
 
+        # 4. Expand K/V heads for GQA (no-op when n_groups == 1, i.e. standard MHA)
+        if self.n_groups > 1:
+            k = k.repeat_interleave(self.n_groups, dim=1)  # (B, nh, T, hs)
+            v = v.repeat_interleave(self.n_groups, dim=1)  # (B, nh, T, hs)
+
         # PyTorch automatically selects the fastest kernel (FlashAttention V2, etc.)
         y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
 
@@ -82,7 +103,7 @@ class MLP(nn.Module):
         super().__init__()
         hidden_dim = int(8 / 3 * config.n_embd)
         hidden_dim = ((hidden_dim + 255) // 256) * 256
-        
+
         # Fused gate and up projection
         self.w13 = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
         self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
@@ -95,18 +116,18 @@ class MLP(nn.Module):
 
 
 class Block(nn.Module):
+    """PaLM-style parallel block: attn and MLP share one pre-norm, residuals summed together.
+    Merged as permanent baseline after exp-007 (val_loss 5.1336, -0.39 vs sequential baseline).
+    """
     def __init__(self, config):
         super().__init__()
         self.ln_1 = nn.RMSNorm(config.n_embd)
         self.attn = CausalSelfAttention(config)
-        self.ln_2 = nn.RMSNorm(config.n_embd)
         self.mlp = MLP(config)
 
     def forward(self, x):
-        # first we go through layer norm, that is fed into attention
-        # then we go through layer norm again, that is fed into MLP
-        x = x + self.attn(self.ln_1(x))
-        x = x + self.mlp(self.ln_2(x))
+        normed = self.ln_1(x)
+        x = x + self.attn(normed) + self.mlp(normed)
         return x
 
 
@@ -157,7 +178,10 @@ class FGPT(nn.Module):
         x = self.transformer.wte(idx)  # (B, T, n_embd) token embeddings
 
         for block in self.transformer.h:
-            x = block(x)
+            if self.config.gradient_checkpointing and self.training:
+                x = grad_checkpoint(block, x, use_reentrant=False)
+            else:
+                x = block(x)
 
         x = self.transformer.ln_f(x)
         logits = self.lm_head(x)
```

No new dependencies are needed — `rotary-embedding-torch` is already a
`main` dependency (used for RoPE before this change too).

### 3.2 `src/fgpt/base_train.py`

```diff
diff --git a/src/fgpt/base_train.py b/src/fgpt/base_train.py
index 4866a32..a182514 100644
--- a/src/fgpt/base_train.py
+++ b/src/fgpt/base_train.py
@@ -1,5 +1,6 @@
 import time
 import math
+import random
 import torch
 from torch import optim
 import json
@@ -36,6 +37,7 @@ def log_train_metrics(
     dataloader_val,
     val_batches,
     now_str=now_str,
+    disable_heavy_evals=False,
 ):
     pbar.write(
         f"Step: {step} | Loss: {loss:.4f} | norm {norm:.2f} | "
@@ -57,7 +59,7 @@ def log_train_metrics(
     if step % 512 == 0:
         metrics["val_loss"] = calculate_val_loss(model, val_batches)
 
-    if step % 25_000 == 0:
+    if not disable_heavy_evals and step % 25_000 == 0:
         metrics["hellaswag_acc"] = hellaswag_eval_base(model, pbar)
 
     if step % 12 == 0 or step % 25_000 == 0 or step % 256 == 0:
@@ -84,11 +86,17 @@ def calculate_val_loss(model, val_batches):
     losses = []
     with torch.no_grad():
         for x_val, y_val in val_batches:
-            x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
+            x_val = x_val.to("cuda")
+            y_val = y_val.to("cuda")
             with torch.autocast("cuda", dtype=torch.bfloat16):
-                _, val_loss = model(x_val, y_val)
+                logits_val, val_loss = model(x_val, y_val)
             losses.append(val_loss.item())
+            del x_val, y_val, logits_val, val_loss  # free GPU tensors immediately
     model.train()
+    # Release cached allocator blocks so the training forward pass isn't starved.
+    # Without this, 64 val batches leave ~20 GB cached, causing OOM at the next
+    # training step when the logits buffer (3.84 GB) can't be allocated.
+    torch.cuda.empty_cache()
     return sum(losses) / len(losses)
 
 
@@ -96,6 +104,8 @@ def configure_optimizers(
     model,
     adamw_lr: float,
     muon_lr: float,
+    weight_decay: float = 0.1,
+    adamw_beta2: float = 0.99,  # merged after exp-028: beta2=0.99 → 3.9751 (-0.003 vs 3.9780)
 ):
     muon_params = []
     adamw_params = []
@@ -129,9 +139,9 @@ def configure_optimizers(
     opt_adamw = optim.AdamW(
         adamw_params,
         lr=adamw_lr,
-        betas=(0.9, 0.95),
+        betas=(0.9, adamw_beta2),
         eps=1e-8,
-        weight_decay=0.1,
+        weight_decay=weight_decay,
         fused=True,
     )
 
@@ -181,9 +191,24 @@ def train(
     sched_adamw,
     current_step,
     accumulation_steps,
+    max_time_seconds=None,
+    disable_heavy_evals=False,
 ):
+    """Train the model.
+
+    Args:
+        max_time_seconds: If set, stop training after this many wall-clock seconds
+            instead of running to num_steps.
+        disable_heavy_evals: Skip HellaSwag eval and sample generation. Use this
+            for short autoresearch runs where only val_loss matters and the heavy
+            evals would fragment GPU memory and waste time.
+
+    Returns:
+        final_val_loss: Val loss on the fixed val_batches at the end of training.
+    """
     print(f"Starting training for {num_steps} steps...")
     norm_val = 0
+    train_start = time.time()
 
     pbar = tqdm(
         range(current_step, num_steps),
@@ -192,6 +217,11 @@ def train(
         dynamic_ncols=True,
     )
     for i in pbar:
+        # Time-based early stop
+        if max_time_seconds is not None and (time.time() - train_start) >= max_time_seconds:
+            pbar.write(f"Reached time limit ({max_time_seconds}s), stopping at step {i}.")
+            break
+
         t0 = time.time()
         x, y = dataloader_train.next_batch()
         x, y = x.to("cuda"), y.to("cuda")
@@ -224,6 +254,10 @@ def train(
             # Zero grad for BOTH
             opt_muon.zero_grad()
             opt_adamw.zero_grad()
+            # Muon's Newton-Schulz iteration leaves large temporary matrices
+            # (G.T @ G etc.) in the PyTorch cache. Release them immediately so
+            # the next accumulation cycle's forward pass can allocate freely.
+            torch.cuda.empty_cache()
 
         torch.cuda.synchronize()
         t1 = time.time()
@@ -246,9 +280,10 @@ def train(
             dataloader_val=dataloader_val,
             val_batches=val_batches,
             now_str=now_str,
+            disable_heavy_evals=disable_heavy_evals,
         )
 
-        if i % 2048 == 0:
+        if not disable_heavy_evals and i % 2048 == 0:
             log_sample_output(model, pbar, step=i, now_str=now_str)
 
         if i % 10_000 == 0 and i > 0:
@@ -270,8 +305,15 @@ def train(
     print("Training complete.")
     torch.save(model.state_dict(), f"model_weights_{now_str}.pth")
 
+    final_val_loss = calculate_val_loss(model, val_batches)
+    return final_val_loss
+
 
 if __name__ == "__main__":
+    seed = 42
+    torch.manual_seed(seed)
+    random.seed(seed)
+
     model = FGPT(FGPTConfig())
     model.to("cuda")
     accumulation_steps = 12 # -> effective batch size of roughly 0.5m tokens
@@ -324,8 +366,8 @@ if __name__ == "__main__":
         # current_step = checkpoint["step"] + 1
 
     torch.set_float32_matmul_precision("medium")
-    dataloader_train = BaseDataLoader(B, T, split="train")
-    dataloader_val = BaseDataLoader(B, T, split="val")
+    dataloader_train = BaseDataLoader(B, T, split="train", seed=seed)
+    dataloader_val = BaseDataLoader(B, T, split="val", seed=seed + 1)
 
     # Preload fixed validation samples once
     val_batches = [dataloader_val.next_batch() for _ in range(64)]  # 64 mini-batches
```

### 3.3 Production hyperparameters — edit `if __name__ == "__main__":` in `base_train.py`

The diff above only changes function *signatures and defaults*; it does
**not** change the literal hyperparameter values hardcoded in the
production entry point at the bottom of `base_train.py`. After applying
the diff, also edit these lines in the `if __name__ == "__main__":` block:

```python
# before:
start_lr_adamw = 2e-4
start_lr_muon = 0.02
...
warmup_steps = total_updates * 0.01

# after (exp-044 compound config):
start_lr_adamw = 6e-4
start_lr_muon = 0.025
...
warmup_steps = total_updates * 0.025
```

`configure_optimizers(model, adamw_lr=start_lr_adamw, muon_lr=start_lr_muon)`
does not need to change — `weight_decay=0.1` and `adamw_beta2=0.99` are
now its defaults, matching exp-044.

`accumulation_steps = 12` and `FGPTConfig()` (now defaulting to `n_head=8`,
`n_kv_heads=1`, `rope_base=50000`) need no further edits — the new
dataclass defaults already encode the winning architecture.

If running on an 80 GB GPU instead of the 96 GB GH200 used in production,
also pass `gradient_checkpointing=True` to `FGPTConfig()` (see §5, Caveats).

---

## 4. How to reproduce the exact number (3.4038)

There are two distinct reproductions — pick based on what you need.

### 4a. Reproduce the *autoresearch comparison run* (gets you 3.4038 on fixed val batches)

This is the literal experiment that produced 3.4038. It needs the
autoresearch harness (`src/fgpt/autoresearch/`), which doesn't exist on
`main`. Easiest path: cherry-pick those files from the `autoresearch`
branch, or copy `run_experiment.py` / `fixed_val.py` directly.

```bash
# from a checkout that already has the model.py / base_train.py diffs above applied
git checkout autoresearch -- src/fgpt/autoresearch/
source .venv/bin/activate

# one-time setup (only if experiments/fixed_val_batches.pt doesn't exist)
python -m fgpt.autoresearch.fixed_val

python -m fgpt.autoresearch.run_experiment \
    --tag exp-044-mqa-repro \
    --description "MQA: n_kv_heads=1 (extreme GQA)" \
    --duration-minutes 150 \
    --seed 42 \
    --accumulation-steps 8 \
    --adamw-lr 6e-4 \
    --muon-lr 0.025 \
    --warmup-frac 0.025 \
    --rope-base 50000 \
    --n-head 8 \
    --n-kv-heads 1 \
    --weight-decay 0.1 \
    --adamw-beta2 0.99
```

This trains a fresh 600M model from seed 42 for 150 wall-clock minutes
(~1100 optimizer steps), evaluates on the 64 fixed val batches in
`experiments/fixed_val_batches.pt`, and should land at val loss ≈3.40
(small variance from GPU/kernel nondeterminism is expected — the original
run got 3.4038).

Note `run_experiment.py` runs with `B=64` micro-batch and
`accumulation_steps=8` (≈524k token effective batch) with
`gradient_checkpointing=True`, to fit the 600M model + Muon optimizer
state in 80 GB without `torch.compile`. This was an autoresearch-only
constraint, not part of the architectural win — see §5.

### 4b. Apply the win to a real production run

Apply §3.1–3.3 to `main`, then run as before:

```bash
source .venv/bin/activate
python -m fgpt.base_train
```

This runs the full `1_000_000`-step schedule at `B=40`,
`accumulation_steps=12` (≈491,520 token effective batch), with
HellaSwag eval and checkpointing enabled, on the original 96 GB GH200
production hardware. This is the recommended way to apply the win — the
short autoresearch comparison run is a research signal, not the
production training recipe.

---

## 5. Caveats / risks (read before scaling up)

- **MQA risk at long training horizons.** MQA won clearly in 2.5h /
  ~1B-token experiments, but a single shared KV head means queries can't
  specialize their keys. There's a theoretical risk it degrades relative
  to MHA/GQA at very long training (10B+ tokens). Mitigation: monitor val
  loss curves for a plateau/inflection relative to where MQA would be
  expected to fall behind, or use GQA `n_kv_heads=2` as an untested
  middle ground.
- **`accumulation_steps`/batch size differs between the autoresearch run
  and production.** The autoresearch comparison used `B=64`,
  `accumulation_steps=8` (≈524k tokens/step) to fit memory without
  `torch.compile`. Production `main` uses `B=40`, `accumulation_steps=12`
  (≈491k tokens/step). These are close enough that the architectural/LR
  wins should transfer, but batch size itself was never validated at
  production scale by this search (a 2x larger batch, accum=16, was
  tested in phase 2 and was catastrophic *only because it halved
  optimizer steps in a fixed wall-clock budget* — not because the larger
  batch is bad per se. Irrelevant for a step-limited production run).
- **`torch.compile` was disabled for all autoresearch runs** (it added
  ~25 GB of inductor buffer overhead that caused OOM with the 600M model +
  Muon state on 80 GB GPUs). Production `main` still uses
  `torch.compile` — fine, since it doesn't change numerics, only speed.
- **Gradient checkpointing was only needed because autoresearch ran on an
  80 GB H100** instead of the 96 GB GH200 used for production. Leave
  `gradient_checkpointing=False` (the `FGPTConfig` default) for the GH200
  production run unless you hit OOM; enabling it costs ~33% more compute
  in exchange for far lower activation memory.
- **RoPE base 100k showed no benefit over 50k** in phase 1 at the current
  1024-token context length — don't push higher without first increasing
  context length.
- **Don't re-try:** GeGLU (lost 3× across all 3 phases), QK-norm
  (+0.12), z-loss (+0.22), sandwich/post-norm (+0.19), differential
  attention (+0.53), sliding window attention (+0.26, no long-range
  problem exists at T=1024), `n_layer=36` deeper-only (+0.048 at this
  scale/duration).

---

## 6. Provenance

- `experiments/results.jsonl` — raw JSON record for `exp-044-mqa`:
  `val_loss=3.403844155371189`, `duration_s=9017.9`, `seed=42`,
  `accumulation_steps=8`, `start_lr_adamw=0.0006`, `start_lr_muon=0.025`,
  `warmup_frac=0.025`, timestamp `2026-04-14T07:36:42`.
- Git tag `exp-044-mqa` on the `autoresearch` branch marks the exact code
  state this result was produced from.
- Full ranked results table and methodology: `AUTORESEARCH.md`.
- Full findings writeup (what worked, what didn't, why): `RESEARCH_FINDINGS.md`.
