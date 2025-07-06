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
        if i % 500 == 0:
            generated_tokens, decoded_output = model_inference(model=model)
            print(f"Generated output: {decoded_output}")
            # Save generated output to a separate JSONL file
            sample = {
                "step": i,
                "decoded_output": decoded_output,
            }
            with open(f"fgpt/sample_outputs_{now_str}.jsonl", "a") as f:
                f.write(json.dumps(sample) + "\n")
        if i % 5000 == 0 and i > 0:
            # Save model weights every 1000 steps
            checkpoint = {
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': i,
                'loss': loss.item()
            }
            torch.save(checkpoint, f"fgpt/checkpoint_{now_str}_step_{i}.pth")
            print(f"Model weights saved at step {i}.")

    print("Training complete.")
    torch.save(model._orig_mod.state_dict(), f"fgpt/model_weights_{now_str}.pth")


if __name__ == "__main__":
  
    model = GPT(GPTConfig())
    model.to("cuda") 
    
    prev_model_weights = "fgpt/checkpoint_20250706_1423_step_54262.pth"
    load_weights = False
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= 1e-4,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150_000)
    
    if load_weights:
        checkpoint = torch.load(prev_model_weights, weights_only=False, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    torch.set_float32_matmul_precision("medium")
    dataloader = DataLoader(B, T, split="train", shard_index = 0)
    
    model = torch.compile(model)
   
    try:
        train(
            num_steps=150_000,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        with open(f"fgpt/error_{now_str}.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()} - {str(e)}\n")
