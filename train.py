import time
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import json
from datetime import datetime

from model import GPT
from model import GPTConfig
from model import B, T
from data import DataLoader
from inference import model_inference

now_str = datetime.now().strftime("%Y%m%d_%H%M")


def train(
    num_steps,
    model,
    dataloader,
    optimizer,
    scheduler,
):

    print(f"Starting training for {num_steps} steps...")

    for i in range(num_steps):
        t0 = time.time()
        x, y, shard_index = dataloader.next_batch()
        x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()  # start with zero gradients
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)  # (B, T, vocab_size)
        loss.backward()  # compute gradients
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )  # clip gradients to avoid exploding gradients
        optimizer.step()  # update weights and biases
        scheduler.step()  # update learning rate

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = B * T / (t1 - t0)
        if i % 5 == 0:
            print(
                f"Step: {i} | Loss: {loss.item():.4f} | norm {norm:.4f} | t/s: {tokens_per_second:.2f} | time: {t1-t0:.3f} | lr {optimizer.param_groups[0]['lr']} | shard: {shard_index}"
            )
        if i % 10 == 0:
            # Save metrics to a JSON file
            metrics = {
                "step": i,
                "loss": float(loss.item()),
                "norm": float(norm),
                "tokens_per_second": float(tokens_per_second),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "shard_index": shard_index,
            }
            with open(f"fgpt/train_metrics_{now_str}.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
        if i % 200 == 0:
            generated_tokens, decoded_output = model_inference(model=model)
            print(f"Generated output: {decoded_output}")
            # Save generated output to a separate JSONL file
            sample = {
                "step": i,
                "decoded_output": decoded_output,
            }
            with open(f"fgpt/sample_outputs_{now_str}.jsonl", "a") as f:
                f.write(json.dumps(sample) + "\n")

    print("Training complete.")
    torch.save(model.state_dict(), f"fgpt/model_weights_{now_str}.pth")


if __name__ == "__main__":
  
    prev_model_weights = "fgpt/model_weights_20231030_1234.pth"
    load_weights = False
    if load_weights:
        print(f"Loading model weights from {prev_model_weights}")
        state_dict = torch.load(prev_model_weights, map_location="cuda")
        model = GPT(GPTConfig())
        model.load_state_dict(state_dict)
    else:
        model = GPT(GPTConfig())
    
    model.to("cuda")

    torch.set_float32_matmul_precision("medium")
    dataloader = DataLoader(B, T, split="train")  # create a dataloader

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8
    )  # optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100_000)
    try:
        train(
            num_steps=100_000,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), f"fgpt/model_weights_{now_str}_error.pth")
