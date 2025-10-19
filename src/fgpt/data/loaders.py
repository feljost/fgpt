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
from tokenizer import tokenizer
from tokenizer import special_tokens


class BaseDataLoader:
    """Data Loader for large text datasets stored in .npy files as token ids."""

    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        data_root = "edu_fineweb100B"
        all_files = os.listdir(data_root)
        self.shards = [
            os.path.join(data_root, s)
            for s in all_files
            if s.startswith(f"edufineweb_{split}")
        ]
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
            self.loaded_shards[shard_path] = np.load(shard_path, mmap_mode="r")
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
        fills the rest with the ingore_token_id.

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
