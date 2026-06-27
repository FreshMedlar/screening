import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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


@torch.compile(dynamic=True)
def _reservoir_scan_ip(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    ip_a: torch.Tensor,
    ip_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compiled reservoir scan, applying trainable node-wise gain (ip_a) and bias (ip_b).
    """
    h = h0
    states = []
    for t in range(x.shape[1]):
        pre_act = F.linear(x[:, t, :], weight_ih) + F.linear(h, weight_hh)
        h = torch.tanh(ip_a * pre_act + ip_b)
        states.append(h)
    return torch.stack(states, dim=1)


class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) with Intrinsic Plasticity (IP).

    Identical architecture to aerc_simplified.py, but supports Intrinsic Plasticity
    reservoir pre-training via pretrain_reservoir_ip() before the main training phase.

    List of configurable options (optionals):
    1. Leaking Rate (leaking_rate): RETAINED. Controls leaky integration alpha in (0, 1].
       leaking_rate=1.0 means no leaking (standard RNN). Default: 1.0.
    2. Spectral Radius (spectral_radius): RETAINED. Scales W_hh eigenvalue. Default: 0.95.
    3. Embedding Dimension (d_e): RETAINED. Input character embedding size. Default: 16.
    4. Reservoir Size (N): RETAINED. Number of reservoir neurons. Default: 147.
    5. Attention Gate Hidden Size (H): RETAINED. Attention subspace dimension. Default: 30.
    6. IP sigma (passed to pretrain_reservoir_ip): RETAINED. Target std of neuron outputs.
       Recommended: 0.5–0.6 (NOT 0.2 which over-compresses activations). Default: 0.5.
    7. IP mu (passed to pretrain_reservoir_ip): RETAINED. Target mean of neuron outputs.
       Typically 0.0. Default: 0.0.
    """

    def __init__(
        self,
        vocab_size: int,
        d_e: int = 16,
        N: int = 160,
        H: int = 30,
        spectral_radius: float = 0.95,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N
        self.H = H
        self.spectral_radius = spectral_radius
        self.use_rmsnorm = use_rmsnorm
        self.ip_a = nn.Parameter(torch.ones(1, N))
        self.ip_b = nn.Parameter(torch.zeros(1, N))

        # Fixed random input embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # Fixed recurrent reservoir
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=False,
            nonlinearity="tanh",
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False

        _init_reservoir(self.rnn, spectral_radius)

        # Normalization layer
        if use_rmsnorm:
            self.state_norm = nn.RMSNorm(N)
        else:
            self.state_norm = nn.Identity()

        # Attention network: norm(r) (N,) -> W_att (H, N)
        self.net_gate = nn.Linear(N, H)
        self.net_out  = nn.Linear(H, H * N)

        # Final readout: ro (H,) -> logits (V,)
        self.readout = nn.Linear(H, vocab_size)

    def count_parameters(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_reservoir_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute reservoir states, applying gain, bias, and leaky integration
        using the compiled _reservoir_scan_ip function.
        """
        with torch.no_grad():
            x = self.emb(idx)  # (B, T, d_e)

        B, T, _ = x.shape
        h0 = torch.zeros(B, self.N, dtype=x.dtype, device=x.device)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0

        return _reservoir_scan_ip(
            x, h0, weight_ih, weight_hh,
            self.ip_a, self.ip_b
        )

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


def pretrain_reservoir_ip(
    model: "AERC",
    dataset: torch.utils.data.Dataset,
    ip_chars: int,
    batch_size: int,
    eta: float = 1e-5,
    mu: float = 0.0,
    sigma: float = 0.4,
    nepochs: int = 5,
    device: str = "cuda",
) -> None:
    """
    Performs Intrinsic Plasticity (IP) pre-training on the reservoir.

    Each epoch uses a DIFFERENT sequential slice of `ip_chars` characters from the dataset,
    so that ip_chars=10000 with nepochs=15 covers 150k unique characters in total.

    IP adapts gain (ip_a) and bias (ip_b) per neuron so that the reservoir neuron outputs
    match N(mu, sigma^2). After training, ip_a and ip_b are folded back into the RNN weights.

    Recommended sigma: 0.5–0.6 (NOT 0.2 — that over-compresses raw states used by attention).

    Args:
        model:      AERC instance whose reservoir will be adapted.
        dataset:    Full training dataset (CharDataset).
        ip_chars:   Number of characters (samples) per epoch.
        batch_size: Batch size for data loading during IP.
        eta:        IP learning rate.
        mu:         Target output mean. Typically 0.0.
        sigma:      Target output std. Recommended 0.5–0.6.
        nepochs:    Number of sequential IP epochs.
        device:     Device string.
    """
    model.eval()
    rnn = model.rnn
    N = model.N

    # Directly update model.ip_a and model.ip_b in-place (under no_grad)
    ip_a = model.ip_a
    ip_b = model.ip_b

    W_in = rnn.weight_ih_l0   # (N, d_e)
    W_hh = rnn.weight_hh_l0   # (N, N)

    total_chars = ip_chars * nepochs
    available   = len(dataset)
    print(f"Starting IP pre-training: {nepochs} epochs × {ip_chars:,} chars = "
          f"{total_chars:,} chars total (dataset has {available:,} samples).")

    for epoch in range(nepochs):
        start = epoch * ip_chars
        end   = min(start + ip_chars, available)
        if start >= available:
            print(f"  Dataset exhausted after epoch {epoch}. Stopping IP early.")
            break

        ip_subset = torch.utils.data.Subset(dataset, range(start, end))
        ip_loader = DataLoader(ip_subset, batch_size=batch_size, shuffle=False, drop_last=True)

        old_a = ip_a.clone()
        old_b = ip_b.clone()

        for idxs, _ in ip_loader:
            idxs = idxs.to(device)
            with torch.no_grad():
                x = model.emb(idxs)

            B, T, _ = x.shape
            last_state = torch.zeros((B, N), dtype=torch.float32, device=device)

            for t in range(T):
                u = x[:, t, :]
                state_pre = F.linear(u, W_in) + F.linear(last_state, W_hh)  # (B, N)

                y = torch.tanh(ip_a * state_pre + ip_b)                      # (B, N)
                last_state = y

                delta_b = -eta * (
                    -(mu / (sigma**2))
                    + (y / (sigma**2)) * (2.0 * (sigma**2) + 1.0 - y**2 + mu * y)
                )
                delta_a = eta / ip_a + delta_b * state_pre

                with torch.no_grad():
                    ip_b.add_(delta_b.mean(dim=0, keepdim=True))
                    ip_a.add_(delta_a.mean(dim=0, keepdim=True))
                    ip_a.clamp_(min=1e-4)

        diff_a = torch.linalg.norm(old_a - ip_a).item()
        diff_b = torch.linalg.norm(old_b - ip_b).item()
        print(f"  Epoch {epoch+1:2d}/{nepochs} | chars [{start:,}–{end:,}] | "
              f"Δip_a: {diff_a:.6f} | Δip_b: {diff_b:.6f}")

    print("IP pre-training complete. Model gain (ip_a) and bias (ip_b) initialized.")
