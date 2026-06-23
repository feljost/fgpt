from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from rotary_embedding_torch import RotaryEmbedding

B = 24  # batch size (lowered from 40 to fit H100 80GB without OOM)
T = 1024  # sequence length / time


@dataclass
class FGPTConfig:
    block_size: int = 1024  # context size, how many tokens we can look back
    vocab_size: int = (
        50304  # GPT-2's vocab size 50257 --> set to power of 2 for faster cuda
    )
    n_layer: int = 32
    n_head: int = 8  # merged after exp-024: n_head=8 → 3.9780 (-0.062 vs 4.0400 baseline)
    n_embd: int = (
        1248  # embedding dimension -> number of features in each token embedding
    )
    # Recompute activations during backward instead of storing all 32 layers.
    # Reduces activation memory from ~63 GB to ~2 GB at ~33% compute cost.
    # Enable for autoresearch runs on 80 GB H100 (production used 96 GB GH200).
    gradient_checkpointing: bool = False
    # RoPE base frequency. Default 10000 matches GPT-NeoX / original RoPE paper.
    # LLaMA-3 uses 500000; common alternatives are 20000, 100000.
    # Merged after exp-031: rope_base=20000 → 3.9634 (-0.003 vs 3.9667 baseline).
    # Merged after exp-043: rope_base=50000 → 3.4391 (-0.011 vs 3.4504 baseline).
    rope_base: int = 50000
    # Grouped Query Attention: number of KV heads. Must divide n_head evenly.
    # Set equal to n_head for standard MHA. Set to 1 for MQA.
    # Merged after exp-023: GQA n_kv_heads=8 → 4.0400 (-0.08 vs 4.1243 baseline).
    # Merged after exp-024: n_kv_heads=4 (2:1 ratio with n_head=8) → 3.9780 (-0.062 vs 4.0400).
    # exp-044 pushed this to 1 (MQA) for -0.047 more, but that's an extreme ratio;
    # using 2 (4:1 ratio, same as Llama-3) keeps most of the win with a more
    # standard GQA setting.
    n_kv_heads: int = 2


class CausalSelfAttention(nn.Module):
    """Flashattn version of Causal Self-Attention module."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key parameters
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_head % config.n_kv_heads == 0
        self.n_groups = config.n_head // config.n_kv_heads

        kv_dim = config.n_kv_heads * self.head_dim
        # Q projects to n_embd; K+V each project to kv_dim (= n_embd when n_kv_heads == n_head)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * kv_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # rotary embedding object
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, theta=config.rope_base)

    def forward(self, x):
        B, T, C = x.size()

        # 1. Calculate Query, Key, Value
        kv_dim = self.n_kv_heads * self.head_dim
        qkv = self.c_attn(x)

        # Split into q (n_embd), k (kv_dim), v (kv_dim)
        q, k, v = qkv.split([self.n_embd, kv_dim, kv_dim], dim=2)

        # 2. Reshape for multi-head / grouped-query attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)

        # 3. Apply rotary embeddings to Q and K
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # 4. Expand K/V heads for GQA (no-op when n_groups == 1, i.e. standard MHA)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)  # (B, nh, T, hs)
            v = v.repeat_interleave(self.n_groups, dim=1)  # (B, nh, T, hs)

        # PyTorch automatically selects the fastest kernel (FlashAttention V2, etc.)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # Transpose back: (B, nh, T, hs) -> (B, T, nh, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """SwiGLU MLP with fused gate and up projection."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        # Fused gate and up projection
        self.w13 = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        w13_out = self.w13(x)
        w1_out, w3_out = w13_out.split(self.hidden_dim, dim=-1)
        return self.w2(F.silu(w1_out) * w3_out)


class Block(nn.Module):
    """PaLM-style parallel block: attn and MLP share one pre-norm, residuals summed together.
    Merged as permanent baseline after exp-007 (val_loss 5.1336, -0.39 vs sequential baseline).
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        normed = self.ln_1(x)
        x = x + self.attn(normed) + self.mlp(normed)
        return x


class FGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # weight of token embeddings
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # wpe not needed as we are using RoPE
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # final normalization
                ln_f=nn.RMSNorm(config.n_embd),
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
            if self.config.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
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
