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


model = GPT(GPTConfig())
model.eval()  # set to eval mode (would disable dropout, etc., but we don't have any in this model)
model.to("cuda")  # move model to GPU

torch.set_float32_matmul_precision("medium")
dataloader = DataLoader(B, T, process_rank=0, split="train")  # create a dataloader

model = torch.compile(model)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8
)  # optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100_000)
now_str = datetime.now().strftime("%Y%m%d_%H%M")


for i in range(100_000):
    t0 = time.time()
    x, y = dataloader.next_batch()
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
            f"Step: {i} | Loss: {loss.item():.4f} | norm {norm:.4f} | tokens/sec: {tokens_per_second:.2f} | time: {t1-t0:.3f} | lr {optimizer.param_groups[0]['lr']}"
        )
    if i % 10 == 0:
        # Save metrics to a JSON file
        metrics = {
            "step": i,
            "loss": float(loss.item()),
            "norm": float(norm),
            "tokens_per_second": float(tokens_per_second),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with open(f"train_metrics_{now_str}.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
    if i % 200 == 0:
        generated_tokens, decoded_output = model_inference(model=model)
        print(f"Generated output: {decoded_output}")
        # Save generated output to a separate JSONL file
        sample = {
            "step": i,
            "decoded_output": decoded_output,
        }
        with open(f"sample_outputs_{now_str}.jsonl", "a") as f:
            f.write(json.dumps(sample) + "\n")

print("Training complete.")
torch.save(model.state_dict(), f"model_weights_{now_str}.pth")