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
from hellaswag import iterate_examples
from hellaswag import render_example
from hellaswag import get_most_likely_row

now_str = datetime.now().strftime("%Y%m%d_%H%M")
# now_str = "20250901_1702"

def hellaswag_eval(model):
    """Evaluates a PyTorch model on the HellaSwag validation set 
    and logs accuracy metrics to a JSONL file. """
    # evaluate on hellaswag
    print("Running Hellaswag eval")
    model.eval()
    num_correct_norm = 0
    num_total = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to("cuda")
        mask = mask.to("cuda")
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    acc_norm = num_correct_norm / num_total
    model.train()
    print(f"HellaSwag accuracy: {acc_norm:.4f}")
    return acc_norm
    

def log_train_metrics(
    model, 
    step:int, 
    loss:float, 
    norm:float, 
    tokens_per_second:float, 
    lr:float, 
    shard_index:int, 
    dataloader_val, 
    now_str=now_str
    ):
    print(
        f"Step: {step} | Loss: {loss:.4f} | norm {norm:.4f} | t/s: {tokens_per_second:.2f} | lr {optimizer.param_groups[0]['lr']} | shard: {shard_index}"
    )
    
    metrics = {
        "step": step,
        "train_loss": loss,
        "val_loss": None,
        "hellaswag_acc": None,
        "norm": float(norm),
        "tokens_per_second": float(tokens_per_second),
        "lr": lr,
        "shard_index": shard_index,
    }
    if step % 100 == 0:
        metrics["val_loss"] = calculate_val_loss(model, dataloader_val)

    if step % 1_000 == 0 and step > 1:
        metrics["hellaswag_acc"] = hellaswag_eval(model)

    with open(f"fgpt/train_metrics_{now_str}.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")

def calculate_val_loss(model, dataloader_val, now_str=now_str):
    model.eval()
    with torch.no_grad():
        x_val, y_val, _ = dataloader_val.next_batch() # will loop through indefinitely
        x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, val_loss = model(x_val, y_val)
    print(f"Validation loss: {val_loss.item():.4f}")
    model.train()
    return val_loss.item()


def train(
    num_steps,
    model,
    dataloader_train,
    dataloader_val,
    optimizer,
    scheduler,
    current_step=0,
):

    print(f"Starting training for {num_steps} steps...")

    for i in range(current_step, num_steps):
        t0 = time.time()
        x, y, shard_index = dataloader_train.next_batch()
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
            log_train_metrics(
                model=model, 
                step=i,
                loss=float(loss.item()), 
                norm=norm, 
                tokens_per_second=tokens_per_second, 
                lr=float(optimizer.param_groups[0]["lr"]), 
                shard_index = shard_index, 
                dataloader_val=dataloader_val, 
                now_str=now_str
                )
        

        if i % 10_000 == 0:
            hellaswag_eval(model)

            
        if i % 5000 == 0 and i > 0:
            # Save model weights every 5000 steps
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': i,
                'loss': loss.item()
            }
            torch.save(checkpoint, f"fgpt/checkpoint_{now_str}_step_{i}.pth")
            print(f"Model weights saved at step {i}.")

    print("Training complete.")
    torch.save(model.state_dict(), f"fgpt/model_weights_{now_str}.pth")


if __name__ == "__main__":
  
    model = GPT(GPTConfig())
    model.to("cuda")
    current_step = 0
    max_steps=300_000
    start_lr = 1e-4
    min_lr = 0.05 * start_lr
    prev_model_weights = "fgpt/checkpoint_20250904_0642_step_140000.pth"
    load_weights = False
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= 1e-4,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_steps, 
        eta_min=min_lr
    )
    
    
    if load_weights:
        checkpoint = torch.load(prev_model_weights, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_step = checkpoint["step"]+1

    torch.set_float32_matmul_precision("medium")
    dataloader_train = DataLoader(B, T, split="train", shard_index = 0)
    dataloader_val = DataLoader(B, T, split="val", shard_index = 0)
    
    model = torch.compile(model)
   
    train(
        num_steps=max_steps,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        scheduler=scheduler,
        current_step=current_step,
    )
