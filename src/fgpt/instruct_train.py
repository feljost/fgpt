from time import time
from pathlib import Path
import json
import torch
from tokenizer import tokenizer
from fgpt.data.loaders import InstructDataLoader
from fgpt.inference import load_model, model_inference


log_dir = Path(__file__).resolve().parents[2] / "logs"
filename_rabst = "simple_instruction_data.json"
# filename_smoltalk = "smoltalk_instruction_response_pairs.json"
filename_simple = "simple_qa_only.json"
data_dir = Path(__file__).resolve().parents[2] / "instruction_data"

with open(data_dir / filename_simple, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(data_dir / filename_rabst, "r", encoding="utf-8") as f:
    data += json.load(f)

# with open(data_dir / filename_smoltalk, "r", encoding="utf-8") as f:
#     data += json.load(f)

print(f"Data loaded with {len(data)} entries")


train_portion = int(len(data) * 0.90)  # 90% for training
val_portion = len(data) - train_portion  # Remaining 10% for validation

train_data = data[:train_portion]
val_data = data[train_portion : train_portion + val_portion]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))


train_loader = InstructDataLoader(data=train_data, tokenizer=tokenizer, B=16)
val_loader = InstructDataLoader(data=val_data, tokenizer=tokenizer, B=16)
val_batches = [val_loader.next_batch() for _ in range(8)]  # prefetch a few validation

model_weights_path = "/home/ubuntu/fgpt/model_weights_20251129_1850.pth"
model = load_model(model_weights_path=model_weights_path, device="cuda")

print("Pre-training inference test:")

prompts = [
    "What is the capital of France?",
    "What is 2 + 2?",
]
for prompt in prompts:
    res = model_inference(model, prompt, generation_type="autocomplete", max_tokens=25)[
        1
    ]
    print(f"Prompt: {prompt}\nResponse: {res}\n")


batches_in_dataset = len(train_loader) // 16
epochs = 2
steps = batches_in_dataset * epochs
lr = 1e-6  # small LR for finetuning

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
)

print(
    f"Training Config\nLR: {lr}\nEpochs: {epochs}\nBatches per epoch: {batches_in_dataset}\nTotal steps: {steps}"
)
print("Starting training")
model.train()

accumulation_steps = 4  # simulate larger batch size
for i in range(steps):
    start_time = time()
    x, y = train_loader.next_batch()
    x, y = x.to("cuda"), y.to("cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)  # (B, T, vocab_size)

    # keep a copy for logging (unscaled)
    loss_value = float(loss.item())

    # scale loss down so gradients are the average across the virtual batch
    (loss / accumulation_steps).backward()

    # perform optimizer step only at the end of an accumulation cycle
    if (i + 1) % accumulation_steps == 0:
        # clip grads before stepping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # update weights and biases
        optimizer.zero_grad()

    torch.cuda.synchronize()
    end_time = time()

    if i % 32 == 0:
        # calculate validation loss every 16 steps
        model.eval()
        loss_vals = []
        for x_val, y_val in val_batches:
            x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss_val = model(x_val, y_val)
            loss_vals.append(float(loss_val.item()))

        val_loss_avg = sum(loss_vals) / len(loss_vals)
        model.train()

        # log entries every 4 steps
        log_entry = {
            "step": i,
            "train_loss": loss_value,
            "val_loss": val_loss_avg,
            "lr": lr,
        }
        with open(
            f"{log_dir}/instruct_training_metrics.jsonl", "a", encoding="utf-8"
        ) as jf:
            jf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"{val_loss_avg} validation loss at step {i}")

    print(
        f"Step {i:04d} | Train loss = {loss_value:.4f} | batch_size = {list(x.size())} | "
        f"t = {end_time - start_time:.2f}s"
    )


print("Training complete.")
torch.save(model.state_dict(), "model_weights_instruct.pth")


print("Post-training inference test:")

testing_prompts = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Say hello.",
    "Define 'cat'.",
    "What color is the sky?",
    "Translate 'hello' to Spanish.",
    "Where is the White House located?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the boiling point of water?",
    "Summarize the plot of '1984' by George Orwell.",
]

for i, prompt in enumerate(testing_prompts):
    res = model_inference(
        model, prompt, generation_type="conversational", max_tokens=100
    )[1]
    print(f"Example {i}\n PROMPT: {prompt}\n RESPONSE: {res}\n\n")

print("All done.")
