"""
Multiscreen: Language model architecture from "Screening Is Enough"
(arXiv:2604.01178v3, Nakanishi 2026)

Implements the screening mechanism with:
- Trim transform for content-based relevance
- Softmask for distance-aware relevance
- MiPE (Minimal Positional Encoding)
- TanhNorm for output norm bounding
- Gated screening tiles with SiLU-tanh gating
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: row-wise unit-length normalisation  (RSS in the paper diagrams)
# ---------------------------------------------------------------------------
def rss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise unit-length normalisation (divide each row by its L2 norm)."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


# ---------------------------------------------------------------------------
# TanhNorm  (Eq. 13)
# ---------------------------------------------------------------------------
def tanh_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """TanhNorm(x) = tanh(||x||) / ||x|| * x.  For small ||x|| → x."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return (torch.tanh(norm) / norm) * x


# ---------------------------------------------------------------------------
# MiPE  – Minimal Positional Encoding  (Eqs. 6-7, Fig. 3)
# ---------------------------------------------------------------------------
def apply_mipe(x: torch.Tensor, positions: torch.Tensor, w: torch.Tensor,
               w_th: float = 256.0) -> torch.Tensor:
    """
    Apply MiPE rotation to the first two dimensions of x.
    x: (..., T, d) or (B, H, T, d)
    positions: (T,)
    w: (H,) or broadcastable shape
    """
    gamma = torch.where(
        w < w_th,
        (torch.cos(math.pi * w / w_th) + 1.0) / 2.0,
        torch.zeros_like(w),
    )

    gamma = gamma.unsqueeze(1) # (H, 1)
    w_uns = w.unsqueeze(1).clamp(min=1e-8) # (H, 1)
    pos = positions.float().unsqueeze(0) # (1, T)
    
    angle = (math.pi * pos * gamma) / w_uns # (H, T)
    while angle.dim() < x.dim() - 1:
        angle = angle.unsqueeze(0)
    angle = angle.unsqueeze(-1) # (1, H, T, 1) if x is (B, H, T, d)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    x0, x1, x_rest = x[..., 0:1], x[..., 1:2], x[..., 2:]
    r0 = x0 * cos_a - x1 * sin_a
    r1 = x0 * sin_a + x1 * cos_a
    return torch.cat([r0, r1, x_rest], dim=-1)


# ---------------------------------------------------------------------------
# Multiscreen Layer  –  NH parallel gated screening tiles  (Eq. 2)
# ---------------------------------------------------------------------------
class MultiscreenLayer(nn.Module):
    def __init__(self, n_h: int, d_e: int, d_k: int, d_v: int,
                 w_th: float = 256.0, sw_values: list = None,
                 init_so: float = 0.0):
        super().__init__()
        self.n_h = n_h
        self.d_e = d_e
        self.d_k = d_k
        self.d_v = d_v
        self.w_th = w_th

        # Projections  (Eq. 15)
        self.W_Q = nn.Linear(d_e, n_h * d_k, bias=False)
        self.W_K = nn.Linear(d_e, n_h * d_k, bias=False)
        self.W_V = nn.Linear(d_e, n_h * d_v, bias=False)
        self.W_G = nn.Linear(d_e, n_h * d_v, bias=False)
        self.W_O = nn.Linear(n_h * d_v, d_e, bias=False)

        # Learned scalars
        if sw_values is None:
            sw_values = [0.0] * n_h
        self.s_w = nn.Parameter(torch.tensor(sw_values))
        self.s_r = nn.Parameter(torch.zeros(n_h))
        self.s_O = nn.Parameter(torch.full((n_h,), init_so))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_h

        # --- Eq. 15: projections ---
        q = self.W_Q(x).view(B, T, H, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, T, H, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, self.d_v).transpose(1, 2)
        g = self.W_G(x).view(B, T, H, self.d_v).transpose(1, 2)

        # --- Eq. 4: screening window & acceptance width ---
        w = torch.exp(self.s_w) + 1.0
        r = torch.sigmoid(self.s_r)

        # --- Eq. 5: unit-length normalisation ---
        q_bar = rss(q)
        k_bar = rss(k)
        v_bar = rss(v)

        # --- MiPE (Eqs. 6-7) ---
        q_tilde = apply_mipe(q_bar, positions, w, self.w_th)
        k_tilde = apply_mipe(k_bar, positions, w, self.w_th)

        # --- Eq. 8: bounded similarity  s_ij = q̃_i · k̃_j^T  ∈ [-1, 1] ---
        sim = q_tilde @ k_tilde.transpose(-2, -1)  # (B, H, T, T)

        # --- Eq. 9: Trim → distance-unaware relevance α_ij ---
        r_uns = r.view(1, H, 1, 1).clamp(min=1e-8)
        alpha = torch.clamp(1.0 - (1.0 - sim) / r_uns, min=0.0)
        alpha = alpha ** 2

        # --- Eq. 10: Softmask (causal, distance-aware) ---
        j_idx = torch.arange(T, device=x.device).float()
        dist = j_idx.unsqueeze(0) - j_idx.unsqueeze(1)   # (T, T)
        
        w_uns = w.view(H, 1, 1)
        dist_uns = dist.unsqueeze(0)  # (1, T, T)

        in_window = (dist_uns > -w_uns) & (dist_uns <= 0)
        cos_mask = (torch.cos(math.pi * dist_uns / w_uns.clamp(min=1e-8)) + 1.0) / 2.0
        softmask = torch.where(in_window, cos_mask, torch.zeros_like(cos_mask))
        softmask = softmask.unsqueeze(0)  # (1, H, T, T)

        # --- Eq. 11: distance-aware relevance ---
        alpha_d = alpha * softmask

        # --- Eq. 12: aggregate surviving values ---
        h = alpha_d @ v_bar

        # --- Eqs. 13-14: TanhNorm ---
        u = tanh_norm(h)

        # --- Eq. 17: gate = tanh(SiLU(g)) ---
        g_hat = torch.tanh(F.silu(g))

        # --- Eq. 18: Δx = (u ⊙ ĝ) · (e^{s_O} · W_O) ---
        out = u * g_hat
        
        # Scale by s_O before projection
        out = out * torch.exp(self.s_O).view(1, H, 1, 1)

        # Concatenate and project back to d_e
        out = out.transpose(1, 2).reshape(B, T, H * self.d_v)
        delta_x = self.W_O(out)
        
        return x + delta_x


# ---------------------------------------------------------------------------
# Multiscreen Model  (Section 3.1, Fig. 2a)
# ---------------------------------------------------------------------------
class Multiscreen(nn.Module):
    """
    Full Multiscreen language model.

    Architecture (from the paper):
      - Shared, row-wise unit-normalised embedding matrix W_E
      - N_L residual layers, each with N_H parallel gated screening tiles
      - Language-modelling head shares W_E (unit-normalised), scaled by e^{s_F}
    """

    def __init__(self, vocab_size: int, d_e: int, n_l: int, n_h: int,
                 d_k: int = 16, d_v: int = 64, w_th: float = 256.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.n_l = n_l
        self.n_h = n_h
        self.d_k = d_k
        self.d_v = d_v
        self.w_th = w_th

        # --- Embedding (Eq. 1) ---
        self.embedding = nn.Embedding(vocab_size, d_e)

        # Learned scalars for embedding / logit scaling
        self.s_E = nn.Parameter(torch.tensor(0.0))                   # init: 0
        self.s_F = nn.Parameter(torch.tensor(math.log(math.sqrt(d_e))))  # init: log √d_E

        # --- Layers ---
        # s_w linearly spaced from 0 to log(w_th) across heads in each layer
        sw_values = torch.linspace(0.0, math.log(w_th), n_h).tolist()
        # s_O init: log(1 / √(N_H · N_L))
        init_so = math.log(1.0 / math.sqrt(n_h * n_l))

        self.layers = nn.ModuleList([
            MultiscreenLayer(n_h, d_e, d_k, d_v, w_th=w_th,
                             sw_values=sw_values, init_so=init_so)
            for _ in range(n_l)
        ])

        # Initialise weights  (Table 4, Appendix D)
        self._init_weights()

    def _init_weights(self):
        """Initialise per Table 4 of the paper."""
        d_k, d_v, d_e = self.d_k, self.d_v, self.d_e

        # Embedding: N(0, 0.1/√d_E)
        nn.init.normal_(self.embedding.weight, std=0.1 / math.sqrt(d_e))

        for layer in self.layers:
            # W_Q, W_K: N(0, 0.1/√d_K)
            nn.init.normal_(layer.W_Q.weight, std=0.1 / math.sqrt(d_k))
            nn.init.normal_(layer.W_K.weight, std=0.1 / math.sqrt(d_k))
            # W_V: N(0, 0.1/√d_V)
            nn.init.normal_(layer.W_V.weight, std=0.1 / math.sqrt(d_v))
            # W_G: N(0, 0.1)
            nn.init.normal_(layer.W_G.weight, std=0.1)
            # W_O: N(0, 0.1/√d_E)
            nn.init.normal_(layer.W_O.weight, std=0.1 / math.sqrt(d_e))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token indices

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)

        # --- Eq. 1: x^(0) = e^{s_E} · ē_{t_i} ---
        emb = self.embedding(idx)                        # (B, T, d_e)
        x = torch.exp(self.s_E) * rss(emb)              # unit-normalised, scaled

        # --- Eq. 2: residual layers ---
        for layer in self.layers:
            x = layer(x, positions)

        # --- Eq. 3: logits using shared unit-normalised embeddings ---
        # z_ij = x^(N_L) · e^{s_F} · ē_j^T
        e_bar = rss(self.embedding.weight)               # (V, d_e) unit-normalised
        logits = torch.exp(self.s_F) * (x @ e_bar.T)    # (B, T, V)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

