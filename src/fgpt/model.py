from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

B = 16  # batch size
T = 1024  # sequence length / time


@dataclass
class FGPTConfig:
    block_size: int = 1024  # context size, how many tokens we can look back
    vocab_size: int = (
        50304  # GPT-2's vocab size 50257 --> set to power of 2 for faster cuda
    )
    n_layer: int = 32
    n_head: int = 24
    n_embd: int = (
        1248  # embedding dimension -> number of features in each token embedding
    )


class CausalSelfAttention(nn.Module):
    """Flashattn version of Causal Self-Attention module."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # We combine Key, Query, and Value into a single linear layer for efficiency
        # This replaces the internal mechanics of nn.MultiheadAttention
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # rotary embedding object
        self.rotary_emb = RotaryEmbedding(dim = self.head_dim)

    def forward(self, x):
        B, T, C = x.size()

        # 1. Calculate Query, Key, Value
        # Result of c_attn is (B, T, 3 * C)
        qkv = self.c_attn(x)

        # Split into q, k, v -> Each is (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape for Multi-head attention
        # We need to transform (B, T, C) -> (B, n_head, T, head_dim)
        # The 'transpose' is physically moving memory, putting heads in the 2nd dimension
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # 3. Apply rotary embeddings to Q and K
        # rotary_embedding_torch expects shape (B, n_head, T, head_dim)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # PyTorch automatically selects the fastest kernel (FlashAttention V2, etc.)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # Transpose back: (B, nh, T, hs) -> (B, T, nh, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection
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


class FGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # weight of token embeddings
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # weight of position embeddings
                # wpe=nn.Embedding(config.block_size, config.n_embd),
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
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) -> batch of token indices
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T} > block size {self.config.block_size}"
        )

        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)
        # pos_emb = self.transformer.wpe(pos)  # (T, n_emberd) position embeddings
        # tok_emb = self.transformer.wte(idx)  # (B, T, n_embd) token embeddings
        # x = tok_emb + pos_emb  # (B, T, n_embd) sum of token and position embeddings

        x = self.transformer.wte(idx)  # (B, T, n_embd) token embeddings

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# Utility functions:
def load_model(
    model_weights_path: str | None = None,
    device: str = "cuda",
    matmul_precision: str | None = "high",
):
    """Loads the GPT model with given weights onto the specified device.
    Args:
        model_weights_path (str | None): Path to the model weights file. If None,
            initializes a new model.
        device (str): Device to load the model onto ('cuda' or 'cpu').
        matmul_precision (str | None): Precision for matrix multiplications.
    Returns:
        GPT: The loaded fgpt model.
    """
    model = FGPT(FGPTConfig())
    model.to(device)

    if model_weights_path is not None:
        print("Loading model weights from:", model_weights_path)
        checkpoint = torch.load(model_weights_path, map_location=device)
        if "model_state_dict" in checkpoint:
            new_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
        else:
            new_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
            }
        model.load_state_dict(new_state_dict)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    print("Compiling model for inference")
    model = torch.compile(model)
    return model


if __name__ == "__main__":
    config = FGPTConfig()
    model = FGPT(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(f"Config: {config}")
