"""
Simple DataLoader for large text datasets (FineWeb, FineWebEdu, OpenWebText, etc.).

Instead of reading each shard in order (like Karpathy's GPT loader), this version
picks random text chunks. This helps avoid domain drift, where the model overfits 
to early shards and doesn't generalize well. 
Random sampling keeps training data more diverse and validation loss in line with
training loss. (This was actually a game changer in my experiments.)

It also uses mmap and a small in-memory cache to speed things up. More specifically,
we have an active buffer of N open shards (mmap objects) that we randomly sample from.
"""

import os
import random
import numpy as np
import torch
from fgpt.tokenizer import special_tokens

class BaseDataLoader:
    def __init__(self, B, T, split="train", buffer_size=4):
        self.B = B
        self.T = T
        data_root = "edu_fineweb100B"
        
        all_files = sorted(os.listdir(data_root))
        self.shards = [
            os.path.join(data_root, s) 
            for s in all_files 
            if s.startswith(f"edufineweb_{split}")
        ]

        # needed to reset later if we run out of shards (-> trained for full epoch)
        self.all_shards_master = self.shards[:]
        
        random.shuffle(self.shards)
        
        self.buffer_size = min(buffer_size, len(self.shards))
        self.open_shards = []
        
        # Fill the initial buffer with open mmap objects
        for _ in range(self.buffer_size):
            self._load_new_shard()

    def _load_new_shard(self):
        """Pops a shard from the list, mmaps it, and adds to open_shards."""
        if not self.shards:
            # If we ran out of shards, reset the list
            self.shards = self.all_shards_master[:] 
            random.shuffle(self.shards)
            
        shard_path = self.shards.pop() # get rid of used shard
        
        # load new shard with mmap
        data = np.load(shard_path, mmap_mode="r")
        
        self.open_shards.append({
            "data": data,
            "path": shard_path,
            "cursor": 0, # Keep track of where we are in this specific file
            "len": len(data)
        })

    def next_batch(self):
        B, T = self.B, self.T
        
        # randomly pick one of the currently open shards
        shard_idx = random.randint(0, len(self.open_shards) - 1)
        shard_obj = self.open_shards[shard_idx]
        
        # sequentiall< read from that shard
        start = shard_obj["cursor"]
        end = start + B * T + 1
        
        # check if shard has data left
        if end > shard_obj["len"]:
            
            # shard is done. Close it, load a new one, and try again
            self.open_shards.pop(shard_idx)
            self._load_new_shard()

            return self.next_batch() # Recursive retry
            
        # update cursor in the shard
        shard_obj["cursor"] = end
        
        # get data; first slice in numpy and then convert to tensor for efficiency
        chunk = shard_obj["data"][start:end]
        x = torch.from_numpy(chunk[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(chunk[1:].astype(np.int64)).view(B, T)
        
        return x, y, shard_idx


class InstructDataLoader:
    """
    DataLoader for instruction-following dataset. Different to the BaseDataLoader,
    We can save everything in memory since the dataset is small enough.
    """

    def __init__(self, data: dict, B=16, max_T=512, tokenizer=None):
        self.B = B
        self.max_T = max_T
        self.tokenizer = tokenizer

        self.encoded_texts = []
        for entry in data:
            # Apply phi-3 style formatting

            # input text = prompt + '<|assistant|>'
            tokens = self.format_encode_input(entry)
            # output text = response
            tokens += tokenizer.encode(entry["output"])

            # padding will be added during batch creation
            self.encoded_texts.append(tokens)

    def next_batch(self):
        """Returns the next batch of size B and sequence length max_T.

        Loops over the dataset in order, adds padding and ignore tokens.

        Returns:
            x_batch (tensor): inputs with padding; shape (B, max_T)
            y_batch (tensor): outputs with padding and ignore token; shape (B, max_T)
            None: placeholder for shard_index (not used here)
        """

        x_batch = torch.zeros((self.B, self.max_T), dtype=torch.long)
        y_batch = torch.zeros((self.B, self.max_T), dtype=torch.long)

        n = len(self.encoded_texts)
        if n == 0:
            return x_batch, y_batch, None

        idx = getattr(self, "_iter_idx", 0)
        batch_data = [
            enc[: self.max_T] for enc in self.encoded_texts[idx : idx + self.B]
        ]

        self._iter_idx = idx + self.B
        if self._iter_idx >= n:
            self._iter_idx = 0  # reset for next epoch

        # Add padding and ignore tokens
        x_batch, y_batch = self._create_padded_batch(
            batch_data,
            eos_token_id=50256,
            ignore_token_id=-100,
        )

        return x_batch, y_batch

    def _create_padded_batch(
        self,
        batch: list[list[int]],
        eos_token_id: int = 50256,
        ignore_token_id: int = -100,  # for xent loss, -100 is ignored by default
    ):
        """
        This function creates the batches for training.
        It takes the batch as an input, then adds a padding / eos token at the end, and
        fills the rest with the ignore_token_id.

        args:
            batch: list of lists, where each sublist is a sequence of token ids
            eos_token_id: the token id used for padding (usually the eos token)
            ignore_token_id: the token id used to mask out the
                in the loss calculation (usually -100 for xent loss)

        returns:
            inputs_tensor: tensor of shape (batch_size, max_seq_length) with input ids
            targets_tensor: tensor of shape (batch_size, max_seq_length) with target ids


        Example:
            >>> batch = [
            >>>     [0, 1, 2, 3, 4],
            >>>     [5, 6],
            >>>     [7, 8, 9]
            >>> ]
            >>> create_padded_batch(batch)
            tensor([[    0,     1,     2,     3,     4],
                    [    5,     6, 50256,  -100,  -100],
                    [    7,     8,     9, 50256,  -100]])
            tensor([[    1,     2,     3,     4, 50256],
                    [    6, 50256,  -100,  -100,  -100],
                    [    8,     9,  -100,  -100,  -100]])
        TODO: I am unsure why the longest batch should not have an extra eos token
            however, it is the same implementation that rasbt uses in his repo.
        """
        # Source: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/
        # 01_main-chapter-code/gpt_instruction_finetuning.py#L56

        # Find the longest sequence in the batch
        # and increase the max length by +1, which will add one extra
        # padding token below
        batch_max_length = max([len(item) + 1 for item in batch])

        # Pad and prepare inputs
        inputs_batched = []
        targets_batched = []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [eos_token_id]
            # Pad sequences to batch_max_length
            padded = new_item + [eos_token_id] * (batch_max_length - len(new_item))
            # Via padded[:-1], we remove the extra padded token
            # that has been added via the +1 setting in batch_max_length
            # (implementation of rasbt, sticking to it for simplicity)
            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])  # shifted one to the right

            # Replace all but the first padding tokens in targets by ignore_token_id
            mask = targets == eos_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = ignore_token_id

            inputs_batched.append(inputs)
            targets_batched.append(targets)

        inputs_tensor = torch.stack(inputs_batched).to("cuda")
        targets_tensor = torch.stack(targets_batched).to("cuda")

        return inputs_tensor, targets_tensor

    def format_encode_input(self, entry):
        """Turns the given text input into Phi style encoded inpuit."""

        tokens = []
        tokens += [special_tokens["<|user|>"]]
        tokens += self.tokenizer.encode(entry["instruction"])
        if entry.get("input", "") != "":
            tokens += self.tokenizer.encode("\n" + entry["input"])
        tokens += [special_tokens["<|assistant|>"]]
        return tokens

    def __len__(self):
        """allows to call len() on the dataset"""
        return len(self.encoded_texts)
