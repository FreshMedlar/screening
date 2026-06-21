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
        N: int = 130,
        H: int = 38,
        spectral_radius: float = 0.95,
        fb_scaling: float = 0.0,
        dropout: float = 0.0,
        leaking_rate: float = 1.0,
        activation: str = "silu",
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
        self.dropout = nn.Dropout(p=dropout)

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

        # Static ESN readout: norm(r) (N,) -> logits (vocab_size,)
        self.static_head = nn.Linear(N, vocab_size)

        # Trainable attention network F: [norm(r) | y_static] (N+V,) -> W_att (H, N)
        gate_out = H
        self.net_gate = nn.Linear(N + vocab_size, gate_out)
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
            self.static_head.requires_grad_(True)
        elif phase == 2:
            self.state_norm.requires_grad_(True)
            self.net_gate.requires_grad_(True)
            self.net_out.requires_grad_(True)
            self.readout.requires_grad_(True)
            self.static_head.requires_grad_(False)
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
        static_logits = self.static_head(states_normed)  # base static readout

        att_input = torch.cat([states_normed, static_logits], dim=-1)  # (B_flat, N+V)
        gate = self.net_gate(att_input)                  # (B_flat, gate_out)
        if self.activation == "silu":
            h1 = F.silu(gate)                            # (B_flat, H)
        elif self.activation == "tanh":
            h1 = torch.tanh(gate)                        # (B_flat, H)
        elif self.activation == "relu":
            h1 = F.relu(gate)                            # (B_flat, H)
        h1    = self.dropout(h1)                         # (B_flat, H)
        vec   = self.net_out(h1)                         # (B_flat, H*N)
        W_att = vec.view(B_flat, self.H, self.N)         # (B_flat, H, N)

        ro = torch.matmul(W_att, states_flat.unsqueeze(-1)).squeeze(-1)  # (B_flat, H)

        correction  = self.readout(ro)                   # (B_flat, V)  Δy
        logits_flat = static_logits + correction           # (B_flat, V)  residual

        new_shape = orig_shape[:-1] + (self.vocab_size,)
        return logits_flat.view(new_shape)


def pretrain_reservoir_ip(
    model: AERC,
    dataloader: torch.utils.data.DataLoader,
    eta: float = 1e-5,
    mu: float = 0.0,
    sigma: float = 0.2,
    nepochs: int = 5,
    device: str = "cuda",
):
    model.eval()
    rnn = model.rnn
    N = model.N

    ip_a = torch.ones((1, N), dtype=torch.float32, device=device)
    ip_b = torch.zeros((1, N), dtype=torch.float32, device=device)

    W_in = rnn.weight_ih_l0   # (N, d_e)
    W_hh = rnn.weight_hh_l0   # (N, N)

    print(f"Starting IP pre-training for {nepochs} epochs...")
    for epoch in range(nepochs):
        old_a = ip_a.clone()
        old_b = ip_b.clone()

        for idxs, _ in dataloader:
            idxs = idxs.to(device)
            with torch.no_grad():
                x = model.emb(idxs)

            B, T, d_e = x.shape
            last_state = torch.zeros((B, N), dtype=torch.float32, device=device)

            for t in range(T):
                u = x[:, t, :]  # (B, d_e)
                state_pre = F.linear(u, W_in) + F.linear(last_state, W_hh)  # (B, N)
                if model.W_fb is not None:
                    state_pre = state_pre + F.linear(last_state, model.W_fb)

                y_new = torch.tanh(ip_a * state_pre + ip_b)  # (B, N)
                y = (1.0 - model.leaking_rate) * last_state + model.leaking_rate * y_new
                last_state = y

                delta_b = -eta * (-(mu / (sigma**2)) + (y_new / (sigma**2)) * (2.0 * (sigma**2) + 1.0 - y_new**2 + mu * y_new))
                delta_a = eta / ip_a + delta_b * state_pre

                ip_b += delta_b.mean(dim=0, keepdim=True)
                ip_a += delta_a.mean(dim=0, keepdim=True)
                ip_a.clamp_(min=1e-4)

        diff_a = torch.linalg.norm(old_a - ip_a).item()
        diff_b = torch.linalg.norm(old_b - ip_b).item()
        print(f"  Epoch {epoch+1:2d}/{nepochs} | Change in ip_a: {diff_a:.6f} | Change in ip_b: {diff_b:.6f}")

    with torch.no_grad():
        rnn.weight_ih_l0.copy_(ip_a.T * rnn.weight_ih_l0)
        rnn.weight_hh_l0.copy_(ip_a.T * rnn.weight_hh_l0)
        rnn.bias_hh_l0.copy_(ip_b.squeeze(0))
        rnn.bias_ih_l0.zero_()
    print("IP pre-training completed and folded into the reservoir.")
