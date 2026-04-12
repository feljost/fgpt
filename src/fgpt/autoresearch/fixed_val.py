"""
Generate and cache fixed validation batches for reproducible experiment comparison.

Run once before any experiments:
    python -m fgpt.autoresearch.fixed_val

All subsequent experiments load these same batches, ensuring val loss is always
computed on identical data regardless of when or in what order experiments run.
"""

import sys
import torch
from pathlib import Path

# Add src/fgpt/ to path so bare imports like `from model import ...` resolve correctly
_SRC_FGPT = Path(__file__).resolve().parents[1]  # src/fgpt/
sys.path.insert(0, str(_SRC_FGPT))

from fgpt.data.loaders import BaseDataLoader
from model import B, T

FIXED_VAL_PATH = Path(__file__).resolve().parents[3] / "experiments" / "fixed_val_batches.pt"
NUM_VAL_BATCHES = 64
VAL_SEED = 9999  # separate seed from train seed pool


def generate_fixed_val_batches(n=NUM_VAL_BATCHES, seed=VAL_SEED, out_path=FIXED_VAL_PATH):
    """Draw n batches from the val split with a fixed seed and save to disk.

    Batches are stored as CPU tensors to be device-agnostic.
    """
    print(f"Generating {n} fixed val batches (seed={seed})...")
    loader = BaseDataLoader(B, T, split="val", seed=seed)
    batches = []
    for _ in range(n):
        x, y = loader.next_batch()
        batches.append((x.cpu(), y.cpu()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batches, out_path)
    print(f"Saved {n} val batches to {out_path}")
    return batches


def load_fixed_val_batches(path=FIXED_VAL_PATH):
    """Load pre-generated val batches from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fixed val batches not found at {path}.\n"
            "Run:  python -m fgpt.autoresearch.fixed_val"
        )
    batches = torch.load(path, weights_only=True)
    print(f"Loaded {len(batches)} fixed val batches from {path}")
    return batches


if __name__ == "__main__":
    generate_fixed_val_batches()
