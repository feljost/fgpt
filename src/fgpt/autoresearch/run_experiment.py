"""
Experiment runner for fgpt autoresearch.

Usage:
    python -m fgpt.autoresearch.run_experiment \
        --tag exp-000-baseline \
        --description "current config unchanged" \
        --duration-minutes 30

Each run:
  1. Seeds model init and data loading deterministically
  2. Trains for the specified wall-clock duration
  3. Evaluates on fixed val batches (same every run)
  4. Appends result to experiments/results.jsonl
  5. Creates experiments/notes/{tag}.md
  6. Creates a git tag for the current commit
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

# Must be set before importing torch to take effect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch import optim

# Add src/fgpt/ to path so bare imports like `from model import ...` resolve correctly
_SRC_FGPT = Path(__file__).resolve().parents[1]  # src/fgpt/
sys.path.insert(0, str(_SRC_FGPT))

from model import FGPT, FGPTConfig, B, T
from fgpt.data.loaders import BaseDataLoader
from fgpt.autoresearch.fixed_val import load_fixed_val_batches, FIXED_VAL_PATH
from base_train import (
    configure_optimizers,
    get_cosine_schedule_with_warmup_and_plateau,
    calculate_val_loss,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"
NOTES_DIR = EXPERIMENTS_DIR / "notes"
AUTORESEARCH_MD = REPO_ROOT / "AUTORESEARCH.md"


def _update_autoresearch_md(new_result: dict):
    """Update Results Summary and Ideas Queue status in AUTORESEARCH.md."""
    text = AUTORESEARCH_MD.read_text()

    # ── Results Summary table ─────────────────────────────────────
    # Rebuild from all results in results.jsonl, sorted by val_loss
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    results_sorted = sorted(results, key=lambda r: r["val_loss"])
    rows = []
    for rank, r in enumerate(results_sorted, 1):
        rows.append(
            f"| {rank} | `{r['tag']}` | {r['val_loss']:.4f} | {r['description']} |"
        )
    table_body = "\n".join(rows)

    new_table = (
        "| Rank | Tag | Val Loss | Description |\n"
        "|------|-----|----------|-------------|\n"
        + table_body
    )

    # Replace everything between the header row and the italics note
    import re
    text = re.sub(
        r"\| Rank \| Tag \| Val Loss \| Description \|.*?\n\*Auto-updated",
        new_table + "\n\n*Auto-updated",
        text,
        flags=re.DOTALL,
    )

    # ── Ideas Queue: mark tag as done ────────────────────────────
    tag = new_result["tag"]
    text = re.sub(
        rf"(\| \d+ \| `{re.escape(tag)}`[^\n]+)\| [⏳✅❌][^\n]*\|",
        r"\1| ✅ done |",
        text,
    )

    AUTORESEARCH_MD.write_text(text)
    print(f"AUTORESEARCH.md updated.")


def _git(args, check=True):
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT)] + args,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


# Observed throughput on GH200 (tokens/sec). Used only to estimate schedule length.
_TOKENS_PER_SEC = 81_000


_B_TRAIN = 64  # gradient checkpointing keeps mem ~const; B=64 uses ~49 GB on H100

def _estimate_microsteps(duration_minutes: float, T: int) -> int:
    """Estimate how many microsteps fit in duration_minutes at observed throughput."""
    tokens_per_microstep = _B_TRAIN * T
    return int(duration_minutes * 60 * _TOKENS_PER_SEC / tokens_per_microstep)


def run_experiment(
    tag: str,
    description: str,
    duration_minutes: float = 30.0,
    seed: int = 42,
    # Training hyperparams — override these per experiment
    accumulation_steps: int = 8,  # 64*1024*8 = 524,288 tokens ≈ original 491,520
    start_lr_adamw: float = 2e-4,
    start_lr_muon: float = 0.02,
    min_lr_ratio: float = 0.05,
    warmup_frac: float = 0.05,
    plateau_frac: float = 0.0,
    rope_base: int = 10000,
    n_head: int = 24,
    batch_size: int = 64,
    weight_decay: float = 0.1,
    untie_lm_head: bool = False,
    # Per-experiment reasoning (written into the notes file)
    reasoning: str = "",
):
    # Scale the LR schedule to the actual run length so warmup/decay are meaningful.
    # With 1M-step defaults, every 30-min run sits entirely inside warmup.
    total_schedule_steps = _estimate_microsteps(duration_minutes, T)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {tag}")
    print(f"Description: {description}")
    print(f"Duration: {duration_minutes} min | Seed: {seed} | "
          f"~{total_schedule_steps} microsteps | warmup {warmup_frac*100:.0f}%")
    print(f"{'='*60}\n")

    # ── Seeding ──────────────────────────────────────────────────
    torch.manual_seed(seed)
    random.seed(seed)

    # ── Model ────────────────────────────────────────────────────
    # Enable gradient checkpointing to fit in 80 GB (production ran on 96 GB GH200).
    # Recomputes each block during backward instead of storing all 32 layers.
    cfg = FGPTConfig(gradient_checkpointing=True, rope_base=rope_base, n_head=n_head, untie_lm_head=untie_lm_head)
    model = FGPT(cfg)
    model.to("cuda")
    torch.set_float32_matmul_precision("medium")

    # ── Data ─────────────────────────────────────────────────────
    # With gradient checkpointing, activation memory is ~constant (one layer at
    # a time). What scales with B is only the logits + CE-backward tensors.
    # At B=64: ~49 GB used (vs 38.7 GB at B=32), well within 80 GB.
    # Effective batch: 64*1024*8 = 524,288 tokens ≈ original 491,520.
    B_TRAIN = batch_size
    dataloader_train = BaseDataLoader(B_TRAIN, T, split="train", seed=seed)
    val_batches = load_fixed_val_batches(FIXED_VAL_PATH)

    # ── Optimizers & Schedulers ───────────────────────────────────
    opt_muon, opt_adamw = configure_optimizers(
        model, adamw_lr=start_lr_adamw, muon_lr=start_lr_muon, weight_decay=weight_decay
    )

    total_updates = math.ceil(total_schedule_steps / accumulation_steps)
    warmup_steps = int(total_updates * warmup_frac)
    plateau_steps = int(total_updates * plateau_frac)

    scheduler_lambda = partial(
        get_cosine_schedule_with_warmup_and_plateau,
        warmup_steps=warmup_steps,
        plateau_steps=plateau_steps,
        total_steps=total_updates,
        min_ratio=min_lr_ratio,
    )

    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw, scheduler_lambda)
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon, scheduler_lambda)

    # ── Train ─────────────────────────────────────────────────────
    # Note: torch.compile is intentionally omitted for autoresearch runs.
    # Compile adds ~25 GB of inductor buffer overhead on H100 that causes OOM
    # with the 600M model + Muon optimizer state. Uncompiled eager mode fits
    # in 80 GB comfortably and all experiments are equally affected, so
    # relative comparisons remain valid. Production training still uses compile.

    wall_start = time.time()
    final_val_loss = train(
        num_steps=total_schedule_steps,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=None,
        val_batches=val_batches,
        opt_muon=opt_muon,
        opt_adamw=opt_adamw,
        sched_muon=sched_muon,
        sched_adamw=sched_adamw,
        current_step=0,
        accumulation_steps=accumulation_steps,
        max_time_seconds=duration_minutes * 60,
        disable_heavy_evals=True,
    )
    wall_end = time.time()
    duration_s = wall_end - wall_start

    tokens_per_second = (B * T * accumulation_steps) / (duration_s / max(1, 1))
    # rough tokens seen estimate
    steps_trained = int(duration_s / (B * T * accumulation_steps / 81_000))

    print(f"\nFinal val loss: {final_val_loss:.4f}")

    # ── Record result ─────────────────────────────────────────────
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "tag": tag,
        "description": description,
        "val_loss": final_val_loss,
        "duration_s": round(duration_s, 1),
        "duration_minutes": round(duration_minutes, 1),
        "seed": seed,
        "accumulation_steps": accumulation_steps,
        "start_lr_adamw": start_lr_adamw,
        "start_lr_muon": start_lr_muon,
        "warmup_frac": warmup_frac,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Result appended to {RESULTS_FILE}")

    # ── Notes file ────────────────────────────────────────────────
    # Compute delta vs baseline for the Outcome section
    baseline_loss = None
    delta_str = ""
    try:
        with open(RESULTS_FILE) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry["tag"] == "exp-000-baseline":
                    baseline_loss = entry["val_loss"]
                    break
        if baseline_loss is not None and tag != "exp-000-baseline":
            delta = final_val_loss - baseline_loss
            sign = "+" if delta > 0 else ""
            verdict = "worse than" if delta > 0 else "better than"
            delta_str = (
                f"**{'Worse' if delta > 0 else 'Better'} than baseline** "
                f"({final_val_loss:.4f} vs {baseline_loss:.4f}, {sign}{delta:.4f} {verdict} baseline)."
            )
        elif tag == "exp-000-baseline":
            delta_str = f"Baseline established. Val loss **{final_val_loss:.4f}** is the reference for all future experiments."
    except Exception:
        delta_str = "_Could not compute delta vs baseline._"

    notes_path = NOTES_DIR / f"{tag}.md"
    notes_path.write_text(
        f"# {tag}\n\n"
        f"**Description:** {description}  \n"
        f"**Val loss:** {final_val_loss:.4f}  \n"
        f"**Duration:** {duration_s/60:.1f} min  \n"
        f"**Seed:** {seed}  \n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n"
        f"## Hyperparameters\n\n"
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| AdamW LR | {start_lr_adamw} |\n"
        f"| Muon LR | {start_lr_muon} |\n"
        f"| Warmup | {warmup_frac*100:.1f}% |\n"
        f"| Accum steps | {accumulation_steps} |\n"
        f"| Min LR ratio | {min_lr_ratio} |\n"
        f"| RoPE base | {rope_base} |\n"
        f"| n_head | {n_head} |\n\n"
        f"## Reasoning\n\n"
        f"{reasoning if reasoning else '_No reasoning provided._'}\n\n"
        f"## Outcome\n\n"
        f"{delta_str}\n"
    )
    print(f"Notes written to {notes_path}")

    # ── Update AUTORESEARCH.md ────────────────────────────────────
    try:
        _update_autoresearch_md(result)
    except Exception as e:
        print(f"Warning: could not update AUTORESEARCH.md: {e}")

    # ── Git tag + push ────────────────────────────────────────────
    try:
        _git(["tag", tag, "-m", f"{description} | val_loss={final_val_loss:.4f}"])
        print(f"Git tag created: {tag}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not create git tag '{tag}': {e.stderr.strip()}")

    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        # Stage results, notes, AUTORESEARCH.md, plus any modified tracked source files
        # (model changes, run_experiment.py changes, etc. are included so the git tag
        # captures the full diff of what was different for this experiment)
        _git(["add",
              str(AUTORESEARCH_MD),
              str(NOTES_DIR / f"{tag}.md"),
              str(RESULTS_FILE),
        ])
        _git(["add", "-u"])  # stage all other modified tracked files (e.g. model.py)
        _git(["commit", "-m", f"results: {tag} | val_loss={final_val_loss:.4f}"])
        _git(["push", "origin", branch])
        _git(["push", "origin", tag])
        print(f"Committed all changes, pushed branch '{branch}' and tag '{tag}' to origin.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not commit/push to origin: {e.stderr.strip()}")

    return final_val_loss


def main():
    parser = argparse.ArgumentParser(description="Run a 30-minute fgpt training experiment")
    parser.add_argument("--tag", required=True, help="Experiment tag, e.g. exp-001-warmup")
    parser.add_argument("--description", required=True, help="3-5 word description")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--adamw-lr", type=float, default=2e-4)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--rope-base", type=int, default=10000)
    parser.add_argument("--n-head", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--untie-lm-head", action="store_true", default=False)
    parser.add_argument("--reasoning", type=str, default="")
    args = parser.parse_args()

    run_experiment(
        tag=args.tag,
        description=args.description,
        duration_minutes=args.duration_minutes,
        seed=args.seed,
        accumulation_steps=args.accumulation_steps,
        start_lr_adamw=args.adamw_lr,
        start_lr_muon=args.muon_lr,
        warmup_frac=args.warmup_frac,
        rope_base=args.rope_base,
        n_head=args.n_head,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        untie_lm_head=args.untie_lm_head,
        reasoning=args.reasoning,
    )


if __name__ == "__main__":
    main()
