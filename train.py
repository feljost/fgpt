import time
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


B = 16  # batch size
T = 1024  # sequence length / time

@dataclass
class GPTConfig:
    block_size: int = 1024  # context size, how many tokens we can look back
    vocab_size: int = 50304 # GPT-2's vocab size 50257 --> set to power of 2 for faster cuda
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    # using PyTorch's built-in multi-head attention for simplicity (not following karphaty's video)
    # this is a causal self-attention layer, meaning it only attends to previous tokens
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # this could be replaced with flash attention later
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        # Generate causal mask
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        y, _ = self.attn(x, x, x, attn_mask=attn_mask)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLP consists of two linear layers with a GELU activation in between

        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd
        )  # "context to feed-forward"
        # no need for approximate version as used by GPT-2 (it used to be slow, but now it's fast)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd
        )  # "context projection"

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # first we go through layer norm, that is fed into attention
        # then we go through layer norm again, that is fed into MLP
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # weight of token embeddings
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # weight of position embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # actual blocks -> transformer layers (h = hidden)
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # final layer norm
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # actual head that will output logits for each token in the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) -> batch of token indices
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T} > block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, n_emberd) position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd) token embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd) sum of token and position embeddings

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss 


class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("fgpt/tinyshakespeare.txt", "r") as f:
            text = f.read()
        tokens = tiktoken.get_encoding("gpt2").encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Total tokens loaded: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # labels
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        
        return x, y

def load_tokens(file_path):
    # Load tokens from a file and return as a tensor
    np_tokens = np.load(file_path)
    pptokens = torch.tensor(np_tokens, dtype=torch.long)
    return pptokens


class DataLoader:
    def __init__(self, B, T, process_rank = 0, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        assert split in ['train', 'val'], "split must be either 'train' or 'val'"

        data_root = "fgpt/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if s.startswith(f"edufineweb_{split}")]
        self.shards = shards
        
        self.current_shard_index = 0
        self.tokens = load_tokens(os.path.join(data_root, self.shards[self.current_shard_index]))
        self.current_position = self.B * self.T * self.process_rank  # start at a different position for each process
        self.current_position = 0 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # labels
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank  # reset to the start of the next shard
            self.current_shard_index += 1
            if self.current_shard_index >= len(self.shards):
                self.current_shard_index = 0
        
        return x, y


model = GPT(GPTConfig())
model.eval()  # set to eval mode (would disable dropout, etc., but we don't have any in this model)
model.to("cuda")  # move model to GPU


def model_inference(model = model, enc = tiktoken.get_encoding("gpt2")):
    # Inference
    max_length = 50
    tokens = enc.encode("Hello")  # encode a prompt
    # add batch dimension and move to GPU
    tokens = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    x = tokens.to("cuda")  # move to GPU


    # generate tokens

    for _ in range(max_length):
        logits, loss = model(x)  # (B, T, vocab_size)
        logits = logits[
            :, -1, :
        ]  # take the last token's logits (B, vocab_size) --> we only care about the next token
        probs = F.softmax(logits, dim=-1)  # convert to probabilities
        # skipped: temperature and top-k sampling
        next_token = torch.multinomial(
            probs, num_samples=1
        )  # sample from the distribution
        x = torch.cat((x, next_token), dim=1)  # append the new token to the sequence
    
    decoded_output = enc.decode(x[0].tolist())
    
    return x, " ".join(decoded_output.split())  # decode the output


torch.set_float32_matmul_precision("medium")
dataloader = DataLoader(B, T, process_rank = 0, split="train")  # create a dataloader


model = GPT(GPTConfig())
model.to("cuda")
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)  # optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

for i in range(10_001):
    t0 = time.time()
    x, y = dataloader.next_batch()
    x, y = x.to("cuda"), y.to("cuda") 
    optimizer.zero_grad()  # start with zero gradients
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)  # (B, T, vocab_size)
    loss.backward()  # compute gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients to avoid exploding gradients
    optimizer.step() # update weights and biases
    scheduler.step()  # update learning rate
    
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_second = B * T / (t1 - t0) 
    print(f"Step: {i} | Loss: {loss.item():.4f} | norm {norm:.4f} | tokens/sec: {tokens_per_second:.2f} | lr {optimizer.param_groups[0]['lr']}")
    if i % 200 == 0:
        generated_tokens, decoded_output = model_inference(model=model)
        print(f"Generated output: {decoded_output}")
        
