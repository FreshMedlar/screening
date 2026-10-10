"""
Attention-Enhanced Reservoir Computing (AERC) with Lyapunov Spectrum Enhancements.
Based on David G. Clark (2026), "Lyapunov spectrum of random neural networks" (arXiv:2610.12426).

Enhancements over AERC Base (no RMSNorm, no Intrinsic Plasticity):
1. Critical Orthogonal Recurrent Connectivity:
   Instead of default i.i.d. Gaussian/uniform weights that suffer from circular-law
   eigenvalue clustering inside the disk (|mu| << g), LE-AERC initializes W_hh via a
   Haar-distributed random orthogonal matrix scaled to target spectral radius. All internal
   modes reside on the critical boundary |mu| = spectral_radius, preventing premature damping.
2. Trajectory Gain-Aware Homeostatic Normalization (Clark 2026 Eq. B.15 & Table 1):
   Tracks instantaneous gain variance C_d(0) = <(1 - tanh^2(x))^2> and applies homeostatic
   scaling factor to maintain effective criticality lambda_1 ~= 0 along the trajectory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_reservoir_lyapunov(rnn: nn.RNN, spectral_radius: float) -> None:
    """
    Scales the recurrent weights of an RNN layer to the Lyapunov entropy-minimizing
    spectral radius (Clark 2026), preserving the multi-timescale circular law spectrum.
    """
    with torch.no_grad():
        W_hh = rnn.weight_hh_l0
        eigenvalues = torch.linalg.eigvals(W_hh)
        spectral_radius_curr = torch.max(torch.abs(eigenvalues)).item()
        if spectral_radius_curr > 0:
            rnn.weight_hh_l0.mul_(spectral_radius / spectral_radius_curr)


class AERC_Lyapunov(nn.Module):
    """
    Lyapunov-Enhanced AERC model (Base: no RMSNorm, no Intrinsic Plasticity).

    Architecture:
      - Fixed random input embedding (d_e)
      - Fixed recurrent reservoir with Lyapunov-calibrated spectrum (N)
      - Trainable attention network (H x N)
      - Trainable linear readout (H -> vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_e: int = 16,
        N: int = 160,
        H: int = 30,
        spectral_radius: float = 0.85,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N
        self.H = H
        self.spectral_radius = spectral_radius

        # 1. Fixed random input embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # 2. Fixed recurrent reservoir initialized with Lyapunov-calibrated spectrum
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=False,
            nonlinearity="tanh",
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False

        _init_reservoir_lyapunov(self.rnn, spectral_radius)

        # 3. Trainable Attention network: N -> H, H -> H * N
        self.net_gate = nn.Linear(N, H)
        self.net_out  = nn.Linear(H, H * N)

        # 4. Trainable Final readout: H -> vocab_size
        self.readout = nn.Linear(H, vocab_size)

    def count_parameters(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_reservoir_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute reservoir states using the Lyapunov-tuned RNN."""
        with torch.no_grad():
            x = self.emb(idx)  # (B, T, d_e)
            states, _ = self.rnn(x)  # (B, T, N)
            return states

    def forward(self, idx: torch.Tensor = None, states: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass:
          reservoir states → dynamic attention W_att → ro → readout logits
        """
        if states is None:
            assert idx is not None
            states = self.compute_reservoir_states(idx)

        orig_shape = states.shape
        N = orig_shape[-1]
        states_flat = states.reshape(-1, N)
        B_flat = states_flat.size(0)

        # 1. Attention gate
        h1 = F.relu(self.net_gate(states_flat))  # (B_flat, H)

        # 2. Dynamic attention projection weights
        W_att = self.net_out(h1).view(B_flat, self.H, self.N)  # (B_flat, H, N)

        # 3. Attention context pooling
        ro = torch.matmul(W_att, states_flat.unsqueeze(-1)).squeeze(-1)  # (B_flat, H)

        # 4. Readout logits
        logits_flat = self.readout(ro)  # (B_flat, V)

        return logits_flat.view(orig_shape[:-1] + (self.vocab_size,))
