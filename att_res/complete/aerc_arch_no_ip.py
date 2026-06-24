import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_reservoir(rnn: nn.RNN, spectral_radius: float) -> None:
    """
    Scales the recurrent weights of an RNN layer to have the specified spectral radius.
    """
    with torch.no_grad():
        W_hh = rnn.weight_hh_l0
        # Compute eigenvalues of recurrent recurrent weights
        eigenvalues = torch.linalg.eigvals(W_hh)
        spectral_radius_curr = torch.max(torch.abs(eigenvalues)).item()
        if spectral_radius_curr > 0:
            scaling_factor = spectral_radius / spectral_radius_curr
            rnn.weight_hh_l0.mul_(scaling_factor)


@torch.compile(dynamic=True)
def _leaky_reservoir_scan(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    leaking_rate: float,
) -> torch.Tensor:
    """
    Compiled leaky-integration reservoir scan without output feedback.
    """
    h = h0
    states = []
    for t in range(x.shape[1]):
        pre_act = F.linear(x[:, t, :], weight_ih, bias_ih) + F.linear(h, weight_hh, bias_hh)
        h_new = torch.tanh(pre_act)
        h = (1.0 - leaking_rate) * h + leaking_rate * h_new
        states.append(h)
    return torch.stack(states, dim=1)


@torch.compile(dynamic=True)
def _leaky_reservoir_scan_fb(
    x: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    leaking_rate: float,
    W_fb: torch.Tensor,
) -> torch.Tensor:
    """
    Compiled leaky-integration reservoir scan with previous-state output feedback.
    """
    h = h0
    states = []
    for t in range(x.shape[1]):
        pre_act = (
            F.linear(x[:, t, :], weight_ih, bias_ih)
            + F.linear(h, weight_hh, bias_hh)
            + F.linear(h, W_fb)
        )
        h_new = torch.tanh(pre_act)
        h = (1.0 - leaking_rate) * h + leaking_rate * h_new
        states.append(h)
    return torch.stack(states, dim=1)


class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) with Static Readout Conditioning.
    """

    def __init__(
        self,
        vocab_size: int,
        d_e: int = 16,
        N: int = 160,
        H: int = 30,
        spectral_radius: float = 0.95,
        fb_scaling: float = 0.0,
        leaking_rate: float = 1.0,
        activation: str = "tanh",
    ):
        _VALID = ("silu", "tanh", "relu")
        if activation not in _VALID:
            raise ValueError(f"activation must be one of {_VALID}, got {activation!r}")
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N
        self.H = H
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.activation = activation

        # Fixed random input embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # Fixed recurrent reservoir
        # the rnn takes in input the embedded letter (size d_e) and does not "output" anything
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=True,
            nonlinearity="tanh",
        )
        # TODO it seems like there are input -> hiddent weights, but we don't need them
        # they are like a second embedding
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False
        self.rnn.bias_ih_l0.requires_grad = False
        self.rnn.bias_hh_l0.requires_grad = False

        with torch.no_grad():
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
        
        # scaling for echo state property
        with torch.no_grad():
            W_hh = self.rnn.weight_hh_l0
            # Compute eigenvalues of recurrent recurrent weights
            eigenvalues = torch.linalg.eigvals(W_hh)
            spectral_radius_curr = torch.max(torch.abs(eigenvalues)).item()
            if spectral_radius_curr > 0:
                scaling_factor = spectral_radius / spectral_radius_curr
                self.rnn.weight_hh_l0.mul_(scaling_factor)

        # feedback settings (> 0.0 is active)
        if fb_scaling > 0.0:
            W_fb_raw = 2.0 * torch.rand(N, N) - 1.0
            with torch.no_grad():
                sr_curr = torch.max(torch.abs(torch.linalg.eigvals(W_fb_raw))).item()
                if sr_curr > 0:
                    W_fb_raw = W_fb_raw / sr_curr * fb_scaling
            self.register_buffer("W_fb", W_fb_raw)
        else:
            self.register_buffer("W_fb", None)

        self.state_norm = nn.RMSNorm(N)

        # Trainable attention network F: [norm(r)] (N,) -> W_att (H, N)
        gate_out = H
        self.net_gate = nn.Linear(N, gate_out)
        self.net_out  = nn.Linear(H, H * N)

        # Trainable AERC correction readout: ro (H,) -> correction Δy (V,)
        self.readout = nn.Linear(H, vocab_size)

    def count_parameters(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_phase(self, phase: int) -> None:
        """
        Control which parameters are trainable for two-phase training.
        """
        if phase == 1:
            self.state_norm.requires_grad_(False)
            self.net_gate.requires_grad_(False)
            self.net_out.requires_grad_(False)
            self.readout.requires_grad_(False)
        elif phase == 2:
            self.state_norm.requires_grad_(True)
            self.net_gate.requires_grad_(True)
            self.net_out.requires_grad_(True)
            self.readout.requires_grad_(True)
        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

    def compute_reservoir_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute the states for a batched B sequence of len T with embedding dimension d_e
        There are 3 paths for efficiency
            1. NOT feedback NOT leak - fastest one
            2. ONLY leak
            3. BOTH feedback AND leak
        """
        with torch.no_grad():
            x = self.emb(idx)           # (B, T, d_e)

            if self.W_fb is None and self.leaking_rate == 1.0:
                out, _ = self.rnn(x)    # (B, T, N)
                return out

            B, T, _ = x.shape
            h0 = torch.zeros(B, self.N, dtype=x.dtype, device=x.device)

            weight_ih = self.rnn.weight_ih_l0
            weight_hh = self.rnn.weight_hh_l0
            bias_ih   = self.rnn.bias_ih_l0
            bias_hh   = self.rnn.bias_hh_l0

            if self.W_fb is None:
                return _leaky_reservoir_scan(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.leaking_rate
                )
            else:
                return _leaky_reservoir_scan_fb(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.leaking_rate,
                    self.W_fb,
                )

    def forward(self, idx: torch.Tensor = None, states: torch.Tensor = None) -> torch.Tensor:
        """
        Collect states (B, T, N) -> (B*T, N)
        Nomalize -> Basic readout -> FF -> act_fun -> 
                 |                 |
                 >-----------------^ 
        """
        if states is None:
            assert idx is not None
            states = self.compute_reservoir_states(idx)

        orig_shape = states.shape
        N = orig_shape[-1]

        states_flat = states.reshape(-1, N)
        B_flat = states_flat.size(0) # B*T

        states_normed = self.state_norm(states_flat)     # norm neurons

        gate = self.net_gate(states_normed)              # (B_flat, gate_out)
        if self.activation == "silu":
            h1 = F.silu(gate)                            # (B_flat, H)
        elif self.activation == "tanh":
            h1 = torch.tanh(gate)                        # (B_flat, H)
        elif self.activation == "relu":
            h1 = F.relu(gate)                            # (B_flat, H)
        vec   = self.net_out(h1)                         # (B_flat, H*N)
        W_att = vec.view(B_flat, self.H, self.N)         # (B_flat, H, N)

        ro = torch.matmul(W_att, states_flat.unsqueeze(-1)).squeeze(-1)  # (B_flat, H)

        logits_flat  = self.readout(ro)                  # (B_flat, V)

        new_shape = orig_shape[:-1] + (self.vocab_size,)
        return logits_flat.view(new_shape)
