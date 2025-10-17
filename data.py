"""
Simple DataLoader for large text datasets (FineWeb, FineWebEdu, OpenWebText, etc.).

Instead of reading each shard in order (like Karpathy's GPT loader), this version
picks random text chunks from random shards every batch. That helps avoid domain
drift — where the model overfits to early shards and doesn't generalize well.
Random sampling keeps training data more diverse and validation loss in line with 
training loss. (This was actually a game changer in my experiments.)

It also uses mmap and a small in-memory cache to speed things up. 
For huge datasets (e.g. 100B tokens), caching won't help much since there are 
hundreds of shards, but for smaller ones it can make a difference. 
"""
import os
import numpy as np
import random
import torch

class DataLoader:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        data_root = "edu_fineweb100B"
        all_files = os.listdir(data_root)
        self.shards = [os.path.join(data_root, s) for s in all_files
                       if s.startswith(f"edufineweb_{split}")]
        assert len(self.shards) > 0

        # Cache mmap objects instead of full np arrays
        self.loaded_shards = {}
        self.max_cache = 4  # keep a few shards in memory
        self.shard_queue = []  # track LRU

    def _get_tokens(self, shard_path):
        if shard_path not in self.loaded_shards:
            if len(self.loaded_shards) >= self.max_cache:
                # evict oldest shard
                old = self.shard_queue.pop(0)
                del self.loaded_shards[old]
            # mmap for lazy loading
            self.loaded_shards[shard_path] = np.load(shard_path, mmap_mode='r')
            self.shard_queue.append(shard_path)
        return torch.from_numpy(self.loaded_shards[shard_path]).long()

    def next_batch(self):
        B, T = self.B, self.T
        shard_path = random.choice(self.shards)
        tokens = self._get_tokens(shard_path)
        start = random.randint(0, len(tokens) - (B * T + 1))
        buf = tokens[start : start + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        shard_id = os.path.basename(shard_path)
        return x, y, shard_id