"""
This fineweb dataset loader is different to Karpathy's implementation
by it taking each document of the dataset and assigning ttrainc / val
split on document level and not on shard level. This is because we don't
want document overlap in between train and validation sets.
"""

import os
import hashlib
import multiprocessing as mp
import numpy as np
from tokenizer import tokenizer
from tokenizer import special_tokens
from datasets import load_dataset
from tqdm import tqdm
import random


local_dir = "edu_fineweb100B"
remote_name = "sample-100BT"
shard_size = 100_000_000
val_fraction = 0.05

shuffle_seed = 42

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")


fw = fw.shuffle(seed=shuffle_seed)  # shuffle dataset for better mix


def tokenize(doc):
    # returns np.uint16 token array for one document (with leading EOT)
    tokens = [special_tokens["<|endoftext|>"]]
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.int32)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "uint16 overflow"
    return tokens_np.astype(np.uint16, copy=False)


class SplitBuffer:
    def __init__(self, split_name):
        self.split = split_name
        self.shard_idx = 0
        self.buf = np.empty((shard_size,), dtype=np.uint16)
        self.count = 0
        self.pbar = None

    def _filename(self):
        return os.path.join(
            DATA_CACHE_DIR, f"edufineweb_{self.split}_{self.shard_idx:06d}"
        )

    def _ensure_pbar(self):
        if self.pbar is None:
            self.pbar = tqdm(
                total=shard_size,
                unit="tokens",
                desc=f"{self.split} shard {self.shard_idx}",
            )

    def add_tokens(self, tok_arr):
        i = 0
        n = len(tok_arr)
        while i < n:
            space = shard_size - self.count
            take = min(space, n - i)
            # fill
            self.buf[self.count : self.count + take] = tok_arr[i : i + take]
            self.count += take
            self._ensure_pbar()
            self.pbar.update(take)
            i += take

            # flush if full
            if self.count == shard_size:
                np.save(self._filename(), self.buf)
                if self.pbar:
                    self.pbar.close()
                self.pbar = None
                self.shard_idx += 1
                self.count = 0


# Actual loop
nprocs = max(1, os.cpu_count() // 2)
train_buf = SplitBuffer("train")
val_buf = SplitBuffer("val")

with mp.Pool(nprocs) as pool:
    for tok in pool.imap(tokenize, fw, chunksize=16):
        split = "val" if random.random() < val_fraction else "train"
        (val_buf if split == "val" else train_buf).add_tokens(tok)
