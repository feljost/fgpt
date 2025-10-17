import numpy as np
import torch
import os
import random

def load_tokens(file_path):
    np_tokens = np.load(file_path)
    return torch.tensor(np_tokens, dtype=torch.long)

class DataLoader:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        assert split in ["train", "val"], "split must be 'train' or 'val'"

        data_root = "edu_fineweb100B"
        all_files = os.listdir(data_root)
        self.shards = [os.path.join(data_root, s) for s in all_files
                       if s.startswith(f"edufineweb_{split}")]
        assert len(self.shards) > 0, f"No shards found for split={split}"

        # Keep one shard loaded at a time to save memory
        self.loaded_shards = {}
        self.device = "cpu"  # adjust if you preload to GPU

    def _get_tokens(self, shard_path):
        # simple cache to avoid reloading every time
        if shard_path not in self.loaded_shards:
            self.loaded_shards[shard_path] = load_tokens(shard_path)
        return self.loaded_shards[shard_path]

    def next_batch(self):
        B, T = self.B, self.T

        # Randomly pick a shard each call
        shard_path = random.choice(self.shards)
        tokens = self._get_tokens(shard_path)

        # Randomly pick a start index
        start = random.randint(0, len(tokens) - (B * T + 1))
        buf = tokens[start : start + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Return shard id for logging
        shard_id = os.path.basename(shard_path)
        return x, y, shard_id
