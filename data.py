
import numpy as np
import torch
import tiktoken
import os

def load_tokens(file_path):
    # Load tokens from a file and return as a tensor
    np_tokens = np.load(file_path)
    pptokens = torch.tensor(np_tokens, dtype=torch.long)
    return pptokens


class DataLoader:
    def __init__(self, B, T, process_rank = 0, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        assert split in ['train', 'val'], "split must be either 'train' or 'val'"

        data_root = "fgpt/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if s.startswith(f"edufineweb_{split}")]
        self.shards = shards
        
        self.current_shard_index = 0
        self.tokens = load_tokens(os.path.join(data_root, self.shards[self.current_shard_index]))
        self.current_position = self.B * self.T * self.process_rank  # start at a different position for each process
        self.current_position = 0 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # labels
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank  # reset to the start of the next shard
            self.current_shard_index += 1
            if self.current_shard_index >= len(self.shards):
                self.current_shard_index = 0
        
        return x, y