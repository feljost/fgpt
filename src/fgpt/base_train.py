import time
import math
import torch
from torch import optim
import json
from datetime import datetime
from pathlib import Path

from model import FGPT
from model import FGPTConfig
from model import B, T
from fgpt.data.loaders import BaseDataLoader
from fgpt.inference import model_inference
from fgpt.eval.hellaswag import iterate_examples
from fgpt.eval.hellaswag import render_example
from fgpt.eval.hellaswag import get_most_likely_row

now_str = datetime.now().strftime("%Y%m%d_%H%M")
# now_str = "20251016_1458"

checkpoints_dir = Path(__file__).resolve().parents[2] / "checkpoints"
logs_dir = Path(__file__).resolve().parents[2] / "logs"


def hellaswag_eval(model):
    """Evaluates a PyTorch model on the HellaSwag validation set
    and logs accuracy metrics to a JSONL file."""
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
    step: int,
    loss: float,
    norm: float,
    tokens_per_second: float,
    seconds_per_step: float,
    adamw_lr: float,
    muon_lr: float,
    dataloader_val,
    val_batches,
    now_str=now_str,
):
    print(
        f"Step: {step} | Loss: {loss:.4f} | norm {norm:.2f} | "
        f"tok/s: {tokens_per_second:.0f} | adamw_lr {adamw_lr:.7f} | "
        f"muon_lr {muon_lr:.7f} | s/step: {seconds_per_step:.2f} |"
    )

    metrics = {
        "step": step,
        "train_loss": loss,
        "val_loss": None,
        "hellaswag_acc": None,
        "norm": float(norm),
        "tokens_per_second": float(tokens_per_second),
        "adamw_lr": adamw_lr,
        "muon_lr": muon_lr,
    }
    if step % 128 == 0:
        metrics["val_loss"] = calculate_val_loss(model, val_batches)

    if step % 10_000 == 0 and step > 1:
        metrics["hellaswag_acc"] = hellaswag_eval(model)

    with open(f"{logs_dir}/train_metrics_{now_str}.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_sample_output(model, step, now_str=now_str):
    generated_tokens, decoded_output = model_inference(
        model=model, prompt="Once upon a time"
    )
    print(f"Step: {step} | Generated output: {decoded_output}")
    # Save generated output to a separate JSONL file
    sample = {
        "step": step,
        "decoded_output": decoded_output,
    }
    with open(f"{logs_dir}/sample_outputs_{now_str}.jsonl", "a") as f:
        f.write(json.dumps(sample) + "\n")


def calculate_val_loss(model, val_batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_val, y_val in val_batches:
            x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, val_loss = model(x_val, y_val)
            losses.append(val_loss.item())
    model.train()
    return sum(losses) / len(losses)

def configure_optimizers(model, learning_rate, weight_decay):
    muon_params = []
    adamw_params = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            # Muon for 2D matrices, AdamW for everything else
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

    opt_muon = optim.Muon(
        muon_params,
        lr=0.02,
        momentum=0.95,
        ns_steps=5
    )
    
    opt_adamw = optim.AdamW(
        adamw_params, 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=True
    )
    
    return opt_muon, opt_adamw


def train(
    num_steps,
    model,
    dataloader_train,
    dataloader_val,
    val_batches,
    opt_muon,    # New Arg
    opt_adamw,   # New Arg
    sched_muon,  # New Arg
    sched_adamw, # New Arg
    current_step=0,
    accumulation_steps=5,
):
    print(f"Starting training for {num_steps} steps...")
    norm_val = 0

    for i in range(current_step, num_steps):
        t0 = time.time()
        x, y = dataloader_train.next_batch()
        x, y = x.to("cuda"), y.to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)  # (B, T, vocab_size)

        # keep a copy for logging (unscaled)
        loss_value = float(loss.item())

        # scale loss down so gradients are the average across the virtual batch
        (loss / accumulation_steps).backward()

        # perform optimizer step only at the end of an accumulation cycle
        if (i - current_step + 1) % accumulation_steps == 0:
            # clip grads before stepping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            norm_val = float(norm)
            
            # Step BOTH optimizers
            opt_muon.step()
            opt_adamw.step()
            
            # Step BOTH schedulers
            sched_muon.step()
            sched_adamw.step()
            
            # Zero grad for BOTH
            opt_muon.zero_grad()
            opt_adamw.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = B * T / (t1 - t0)
        second_per_step = t1 - t0

        if i % (accumulation_steps / 2) == 0:
            # We log the AdamW LR as the reference "base" learning rate
            adamw_lr = float(opt_adamw.param_groups[0]["lr"])
            muon_lr = float(opt_muon.param_groups[0]["lr"])
            
            log_train_metrics(
                model=model,
                step=i,
                loss=loss_value,
                norm=norm_val,
                tokens_per_second=tokens_per_second,
                seconds_per_step=second_per_step,
                adamw_lr=adamw_lr,
                muon_lr=muon_lr,
                dataloader_val=dataloader_val,
                val_batches=val_batches,
                now_str=now_str,
            )

        if i % 512 == 0:
            log_sample_output(model, step=i, now_str=now_str)

        if i % 5000 == 0 and i > 0:
            # Save model weights every 5000 steps
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "opt_muon_state_dict": opt_muon.state_dict(),   # Save Muon
                "opt_adamw_state_dict": opt_adamw.state_dict(), # Save AdamW
                "sched_muon_state_dict": sched_muon.state_dict(),
                "sched_adamw_state_dict": sched_adamw.state_dict(),
                "step": i,
                "loss": loss.item(),
            }
            torch.save(checkpoint, f"{checkpoints_dir}/checkpoint_{now_str}_step_{i}.pth")
            print(f"Model weights saved at step {i}.")
    
    print("Training complete.")
    torch.save(model.state_dict(), f"model_weights_{now_str}.pth")


if __name__ == "__main__":
    model = FGPT(FGPTConfig())
    model.to("cuda")
    accumulation_steps = 2
    current_step = 0
    max_steps = 200_000 + 1
    start_lr = 3e-4
    min_lr = 0.1 * start_lr
    
    # Optimizer Setup
    opt_muon, opt_adamw = configure_optimizers(model, learning_rate=start_lr, weight_decay=0.1)

    # Scheduler Logic (Applied to both)
    total_updates = math.ceil(max_steps / accumulation_steps)
    warmup_steps = max_steps * 0.02
    warmup_updates = math.ceil(warmup_steps / accumulation_steps)

    def lr_lambda(step: int):
        if step < warmup_updates:
            return float(step) / float(max(1, warmup_updates))
        progress = float(step - warmup_updates) / float(max(1, total_updates - warmup_updates))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = float(min_lr) / float(start_lr)
        return min_ratio + (1.0 - min_ratio) * cosine

    # Create two schedulers. 
    # LambdaLR scales the *initial* LR of the optimizer. 
    # So Muon (init 0.02) and AdamW (init 3e-4) will both scale correctly.
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon, lr_lambda)
    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw, lr_lambda)
    
    load_weights = False
    prev_model_weights = "/path/to/checkpoint.pth"

    if load_weights:
        print(f"Loading weights from {prev_model_weights}")
        checkpoint = torch.load(prev_model_weights, map_location="cuda")
        
        # Load Model
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(new_state_dict)
        
        opt_muon.load_state_dict(checkpoint["opt_muon_state_dict"])
        opt_adamw.load_state_dict(checkpoint["opt_adamw_state_dict"])
        sched_muon.load_state_dict(checkpoint["sched_muon_state_dict"])
        sched_adamw.load_state_dict(checkpoint["sched_adamw_state_dict"])
        current_step = checkpoint["step"] + 1

    torch.set_float32_matmul_precision("medium")
    dataloader_train = BaseDataLoader(B, T, split="train")
    dataloader_val = BaseDataLoader(B, T, split="val")

    # Preload fixed validation samples once
    val_batches = [dataloader_val.next_batch() for _ in range(64)]  # 64 mini-batches

    model = torch.compile(model)

    train(
        num_steps=max_steps,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        val_batches=val_batches,
        opt_muon=opt_muon,       # Pass Muon
        opt_adamw=opt_adamw,     # Pass AdamW
        sched_muon=sched_muon,   # Pass Muon Scheduler
        sched_adamw=sched_adamw, # Pass AdamW Scheduler
        current_step=current_step,
        accumulation_steps=accumulation_steps,
    )