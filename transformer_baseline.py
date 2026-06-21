"""
Transformer Baseline: LLaMA-style architecture for comparison with Multiscreen.

Features:
  - RoPE (Rotary Position Embedding)
  - RMSNorm (pre-norm)
  - Standard feed-forward (non-gated, SiLU)
  - Multi-head causal self-attention
  - Weight-tied embedding / output head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_rsqrt = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms_rsqrt).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def precompute_rope_params(dim: int, max_len: int, theta: float = 10000.0,
                           device: torch.device = None):
    """Precompute RoPE cos and sin tensors: (1, max_len, 1, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)     # (max_len, dim//2)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2) # (1, max_len, 1, dim//2)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2) # (1, max_len, 1, dim//2)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x.
    x: (B, T, H, head_dim)
    cos, sin: (1, max_len, 1, head_dim//2)
    """
    d2 = x.shape[-1] // 2
    x0 = x[..., :d2]
    x1 = x[..., d2:]
    c = cos[:, :x.shape[1], :, :]
    s = sin[:, :x.shape[1], :, :]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)


# ---------------------------------------------------------------------------
# Standard Feed-Forward
# ---------------------------------------------------------------------------
class StandardFF(nn.Module):
    """Standard feed-forward (non-gated, SiLU)."""

    def __init__(self, d_e: int, ff_dim: int = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_e
        self.w_up = nn.Linear(d_e, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, d_e, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_up(x)))


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention with RoPE
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_e: int, n_h: int, max_len: int = 2048):
        super().__init__()
        assert d_e % n_h == 0
        self.d_e = d_e
        self.n_h = n_h
        self.head_dim = d_e // n_h

        self.W_QKV = nn.Linear(d_e, 3 * d_e, bias=False)
        self.W_O = nn.Linear(d_e, d_e, bias=False)

        # Precompute RoPE angles
        cos, sin = precompute_rope_params(self.head_dim, max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.W_QKV(x).reshape(B, T, 3, self.n_h, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each: (B, T, H, head_dim)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Transpose to (B, H, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use FlashAttention / Memory-Efficient Attention if available
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).reshape(B, T, C)     # (B, T, C)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm LLaMA style)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_e: int, n_h: int, max_len: int = 2048):
        super().__init__()
        self.norm1 = RMSNorm(d_e)
        self.attn = CausalSelfAttention(d_e, n_h, max_len)
        self.norm2 = RMSNorm(d_e)
        self.ff = StandardFF(d_e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Language Model
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    """
    LLaMA-style Transformer baseline with weight-tied embeddings.
    """

    def __init__(self, vocab_size: int, d_e: int, n_l: int, n_h: int,
                 max_len: int = 2048):
        super().__init__()
        self.d_e = d_e
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_e)
        self.layers = nn.ModuleList([
            TransformerBlock(d_e, n_h, max_len) for _ in range(n_l)
        ])
        self.norm_f = RMSNorm(d_e)

        # Weight tying: output head shares embedding weight
        # (no separate lm_head parameter)

        self._init_weights(n_l, d_e)

    def _init_weights(self, n_l: int, d_e: int):
        """Pythia/GPT-NeoX style init."""
        small_std = math.sqrt(2.0 / (5.0 * d_e))
        wang_std = math.sqrt(2.0 / (n_l * d_e))

        nn.init.normal_(self.embedding.weight, std=small_std)

        for layer in self.layers:
            # Attention: small init for Q,K,V; Wang init for output proj
            # W_QKV is fused – init with small_std
            nn.init.normal_(layer.attn.W_QKV.weight, std=small_std)
            nn.init.normal_(layer.attn.W_O.weight, std=wang_std)

            # FF: small init for up; Wang init for down (residual proj)
            nn.init.normal_(layer.ff.w_up.weight, std=small_std)
            nn.init.normal_(layer.ff.w_down.weight, std=wang_std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token indices

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embedding(idx)               # (B, T, d_e)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        # Weight-tied head
        logits = x @ self.embedding.weight.T  # (B, T, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
