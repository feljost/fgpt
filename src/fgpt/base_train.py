import time
import math
import torch
from torch import optim
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from functools import partial

from model import FGPT
from model import FGPTConfig
from model import B, T
from fgpt.data.loaders import BaseDataLoader
from fgpt.inference import model_inference
from fgpt.eval.hellaswag import iterate_examples
from fgpt.eval.hellaswag import render_example
from fgpt.eval.hellaswag import get_most_likely_row

# now_str = datetime.now().strftime("%Y%m%d_%H%M")
now_str = "20251220_1602"

checkpoints_dir = Path(__file__).resolve().parents[2] / "checkpoints"
logs_dir = Path(__file__).resolve().parents[2] / "logs"


def hellaswag_eval(model, pbar):
    """Evaluates a PyTorch model on the HellaSwag validation set
    and logs accuracy metrics to a JSONL file."""
    # evaluate on hellaswag
    pbar.write("Running Hellaswag eval")
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
    pbar.write(f"HellaSwag accuracy: {acc_norm:.4f}")
    return acc_norm


def log_train_metrics(
    model,
    pbar,
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
    pbar.write(
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
        "timestamp": datetime.now().isoformat(),
    }
    if step % 256 == 0:
        metrics["val_loss"] = calculate_val_loss(model, val_batches)

    if step % 10_000 == 0:
        metrics["hellaswag_acc"] = hellaswag_eval(model, pbar)

    with open(f"{logs_dir}/train_metrics_{now_str}.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_sample_output(model, pbar, step, now_str=now_str):
    generated_tokens, decoded_output = model_inference(
        model=model, prompt="Once upon a time"
    )
    pbar.write(f"Step: {step} | Generated output: {decoded_output}")
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


def configure_optimizers(
    model,
    adamw_lr: float,
    muon_lr: float,
):
    muon_params = []
    adamw_params = []

    # Set of parameter IDs that should be Muon
    muon_param_ids = set()

    # 1. Identify Muon Candidates (Linear Layers only, excluding Head)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            # Exclude the final lm_head if it shares weights or just generally
            # usually lm_head is better with AdamW for stability
            if "lm_head" in name:
                continue

            # Add the weight to Muon
            for p in m.parameters():
                # Linear weights are 2D, biases are 1D. Muon only wants weights.
                if p.ndim == 2 and p.requires_grad:
                    muon_params.append(p)
                    muon_param_ids.add(id(p))

    # 2. Everything else goes to AdamW (Embeddings, Norms, Biases, lm_head)
    for p in model.parameters():
        if p.requires_grad and id(p) not in muon_param_ids:
            adamw_params.append(p)

    # 3. Init Optimizers
    opt_muon = optim.Muon(muon_params, lr=muon_lr)  #  adjust_lr_fn="match_rms_adamw"

    opt_adamw = optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    return opt_muon, opt_adamw


def get_cosine_schedule_with_warmup_and_plateau(
    step: int,
    warmup_steps: int,
    plateau_steps: int,
    total_steps: int,
    min_ratio: float,
) -> float:
    """
    Calculates LR multiplier with 3 phases: Warmup -> Plateau -> Cosine Decay.
    """
    # 1. Linear Warmup Phase
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    # 2. Plateau Phase (Hold max LR)
    if step < (warmup_steps + plateau_steps):
        return 1.0

    # 3. Cosine Decay Phase
    decay_steps = total_steps - (warmup_steps + plateau_steps)
    step_in_decay = step - (warmup_steps + plateau_steps)

    # Calculate progress within the decay phase specifically
    progress = float(step_in_decay) / float(max(1, decay_steps))
    progress = min(1.0, max(0.0, progress))

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_ratio + (1.0 - min_ratio) * cosine_decay


def train(
    num_steps,
    model,
    dataloader_train,
    dataloader_val,
    val_batches,
    opt_muon,
    opt_adamw,
    sched_muon,
    sched_adamw,
    current_step=0,
    accumulation_steps=5,
):
    print(f"Starting training for {num_steps} steps...")
    norm_val = 0

    pbar = tqdm(
        range(current_step, num_steps),
        initial=current_step,
        total=num_steps,
        dynamic_ncols=True,
    )
    for i in pbar:
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
            # this schedule could be improved with a smoother transition
            norm_clip = 0.5 if current_step < 250_000 else 1.0

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip)
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

        adamw_lr = float(opt_adamw.param_groups[0]["lr"])
        muon_lr = float(opt_muon.param_groups[0]["lr"])

        log_train_metrics(
            model=model,
            pbar=pbar,
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

        if i % 2048 == 0:
            log_sample_output(model, pbar, step=i, now_str=now_str)

        if i % 10_000 == 0 and i > 0:
            # Save model weights every 5000 steps
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "opt_muon_state_dict": opt_muon.state_dict(),
                "opt_adamw_state_dict": opt_adamw.state_dict(),
                "sched_muon_state_dict": sched_muon.state_dict(),
                "sched_adamw_state_dict": sched_adamw.state_dict(),
                "step": i,
                "loss": loss.item(),
            }
            torch.save(
                checkpoint, f"{checkpoints_dir}/checkpoint_{now_str}_step_{i}.pth"
            )
            pbar.write(f"Model weights saved at step {i}.")

    print("Training complete.")
    torch.save(model.state_dict(), f"model_weights_{now_str}.pth")


if __name__ == "__main__":
    model = FGPT(FGPTConfig())
    model.to("cuda")
    accumulation_steps = 6 # -> effective batch size of roughly 0.5m tokens
    current_step = 0
    max_steps = 350_000 + 1
    start_lr_adamw = 3e-4
    start_lr_muon = 0.03
    min_lr_ratio = 0.1

    # Optimizer Setup
    opt_muon, opt_adamw = configure_optimizers(
        model, adamw_lr=start_lr_adamw, muon_lr=start_lr_muon
    )

    # Scheduler Logic (Applied to both)
    total_updates = math.ceil(max_steps / accumulation_steps)
    warmup_steps = total_updates * 0.03
    plateau_steps = 0

    # Create the lambdas for both optimizers
    scheduler_lambda = partial(
        get_cosine_schedule_with_warmup_and_plateau,
        warmup_steps=warmup_steps,
        plateau_steps=plateau_steps,
        total_steps=total_updates,
        min_ratio=min_lr_ratio,
    )

    sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw, scheduler_lambda)
    sched_muon = torch.optim.lr_scheduler.LambdaLR(opt_muon, scheduler_lambda)

    load_weights = False
    prev_model_weights = f"{checkpoints_dir}/checkpoint_{now_str}_step_250000.pth"

    if load_weights:
        print(f"Loading weights from {prev_model_weights}")
        checkpoint = torch.load(prev_model_weights, map_location="cuda")

        # Load Model
        new_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(new_state_dict)

        opt_muon.load_state_dict(checkpoint["opt_muon_state_dict"])
        opt_adamw.load_state_dict(checkpoint["opt_adamw_state_dict"])
        # sched_muon.load_state_dict(checkpoint["sched_muon_state_dict"])
        # sched_adamw.load_state_dict(checkpoint["sched_adamw_state_dict"])
        # current_step = checkpoint["step"] + 1

    # No warmup, just gentle continued training
    for param_group in opt_muon.param_groups:
        param_group['lr'] = 0.003

    sched_muon = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_muon,
        T_max=16666,
        eta_min=0.0005
    )

    for param_group in opt_adamw.param_groups:
        param_group['lr'] = 3e-5

    sched_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adamw,
        T_max=16666,
        eta_min=5e-6
    )

    torch.set_float32_matmul_precision("medium")
    dataloader_train = BaseDataLoader(B, T, split="train")
    dataloader_val = BaseDataLoader(B, T, split="val")

    # Preload fixed validation samples once
    val_batches = [dataloader_val.next_batch() for _ in range(128)]  # 64 mini-batches

    model = torch.compile(model)

    train(
        num_steps=max_steps,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        val_batches=val_batches,
        opt_muon=opt_muon,
        opt_adamw=opt_adamw,
        sched_muon=sched_muon,
        sched_adamw=sched_adamw,
        current_step=current_step,
        accumulation_steps=accumulation_steps,
    )
