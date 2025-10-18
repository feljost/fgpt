from time import time

import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import tokenizer
from tokenizer import special_tokens
from data_loaders import InstructDataLoader
from inference import load_model, model_inference
from tokenizer import tokenizer


# load data  ( TODO: put this in a function / class )
with open("instruction_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Data loaded with {len(data)} entries")


train_portion = int(len(data) * 0.85)  # 90% for training
val_portion = len(data) - train_portion  # Remaining 10% for validation

train_data = data[:train_portion]
val_data = data[train_portion : train_portion + val_portion]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))


train_loader = InstructDataLoader(data=train_data, tokenizer=tokenizer, B=16)
val_loader = InstructDataLoader(data=val_data, tokenizer=tokenizer, B=16)
x, y = train_loader.next_batch()

model_weights_path = "/home/ubuntu/fgpt-base/model_weights_20251016_1458.pth"
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
lr = 1e-5

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
)

print(
    f"Training Config\nLR: {lr}\nEpochs: {epochs}\nBatches per epoch: {batches_in_dataset}\nTotal steps: {steps}"
)
print(f"Starting training")


start_time = time()
# put model into training mode
model.train()
end_time = time()
print(f"Time to switch to train mode: {end_time - start_time:.4f} seconds")

for i in range(steps):
    start_time = time()
    x, y = train_loader.next_batch()

    x, y = x.to("cuda"), y.to("cuda")

    print(f"Shape of training data: {x.shape}, {y.shape}")

    # clear previous gradients
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    # keep a copy for logging (unscaled)
    loss_value = float(loss.item())

    # compute gradients
    loss.backward()

    # update weights and biases
    optimizer.step()

    torch.cuda.synchronize()
    end_time = time()
    print(
        f"Step {i:04d} | Train loss = {loss_value:.4f} | batch_size = {x.shape} | "
        f"t = {end_time - start_time:.2f}s"
    )


print("Training complete.")
torch.save(model.state_dict(), f"model_weights_instruct.pth")


print("Post-training inference test:")
prompt = "Explain the theory of relativity in simple terms."

testing_prompts = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Say hello.",
    "Define 'cat'.",
    "What color is the sky?",
    "Translate 'hello' to Spanish.",
]

for i, prompt in enumerate(testing_prompts):
    res = model_inference(
        model, prompt, generation_type="conversational", max_tokens=25
    )[1]
    print(f"Example {i}\n PROMPT: {prompt}\n RESPONSE: {res}\n\n")
