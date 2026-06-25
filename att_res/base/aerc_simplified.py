import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_reservoir(rnn: nn.RNN, spectral_radius: float) -> None:
    """
    Scales the recurrent weights of an RNN layer to have the specified spectral radius.
    """
    with torch.no_grad():
        W_hh = rnn.weight_hh_l0
        eigenvalues = torch.linalg.eigvals(W_hh)
        spectral_radius_curr = torch.max(torch.abs(eigenvalues)).item()
        if spectral_radius_curr > 0:
            rnn.weight_hh_l0.mul_(spectral_radius / spectral_radius_curr)


class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) - Base Model (No Intrinsic Plasticity).

    Simplifications from the original AERC model:
    - Removed: Feedback connections (fb_scaling). No output feedback loop.
    - Removed: Activation function choice. Hardcoded to ReLU.
    - Removed: Two-phase training / Ridge regression baseline fit.
    - Removed: Dropout.

    List of configurable options (optionals):
    1. Leaking Rate (leaking_rate): RETAINED. Controls leaky integration alpha in (0, 1].
       leaking_rate=1.0 means no leaking (standard RNN). Default: 1.0.
    2. Spectral Radius (spectral_radius): RETAINED. Scales W_hh eigenvalue. Default: 0.95.
    3. Embedding Dimension (d_e): RETAINED. Input character embedding size. Default: 16.
    4. Reservoir Size (N): RETAINED. Number of reservoir neurons. Default: 147.
    5. Attention Gate Hidden Size (H): RETAINED. Attention subspace dimension. Default: 30.
    """

    def __init__(
        self,
        vocab_size: int,
        d_e: int = 16,
        N: int = 160,
        H: int = 30,
        spectral_radius: float = 0.95,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N
        self.H = H
        self.spectral_radius = spectral_radius

        # Fixed random input embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # Fixed recurrent reservoir
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=True,
            nonlinearity="tanh",
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False
        self.rnn.bias_ih_l0.requires_grad = False
        self.rnn.bias_hh_l0.requires_grad = False

        with torch.no_grad():
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()

        _init_reservoir(self.rnn, spectral_radius)

        # Normalization layer (trainable scale)
        self.state_norm = nn.RMSNorm(N)

        # Attention network: norm(r) (N,) -> W_att (H, N)
        self.net_gate = nn.Linear(N, H)
        self.net_out  = nn.Linear(H, H * N)

        # Final readout: ro (H,) -> logits (V,)
        self.readout = nn.Linear(H, vocab_size)

    def count_parameters(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_reservoir_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute reservoir states using the recurrent RNN."""
        with torch.no_grad():
            x = self.emb(idx)  # (B, T, d_e)
            out, _ = self.rnn(x)
            return out

    def forward(self, idx: torch.Tensor = None, states: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass:
          reservoir states → RMSNorm → static readout → gate → relu → W_att → ro → correction → logits
        """
        if states is None:
            assert idx is not None
            states = self.compute_reservoir_states(idx)

        orig_shape = states.shape
        N = orig_shape[-1]
        states_flat = states.reshape(-1, N)
        B_flat = states_flat.size(0)

        # 1. Normalize reservoir states
        states_normed = self.state_norm(states_flat)  # (B_flat, N)

        # 2. Gate network (conditioned on normalized states only)
        h1 = F.relu(self.net_gate(states_normed))     # (B_flat, H)

        # 3. Dynamic attention weights
        W_att = self.net_out(h1).view(B_flat, self.H, self.N)  # (B_flat, H, N)

        # 4. Attention projection
        ro = torch.matmul(W_att, states_flat.unsqueeze(-1)).squeeze(-1)  # (B_flat, H)

        # 5. Output logits
        logits_flat = self.readout(ro)                # (B_flat, V)

        return logits_flat.view(orig_shape[:-1] + (self.vocab_size,))
