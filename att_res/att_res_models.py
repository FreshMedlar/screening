import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_reservoir(rnn):
    """Scale W_hh to spectral radius 0.95 (echo state property)."""
    with torch.no_grad():
        W_hh = rnn.weight_hh_l0
        spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_hh))).item()
        if spectral_radius > 0:
            rnn.weight_hh_l0.copy_(W_hh / spectral_radius * 0.95)


class ClassicReservoir(nn.Module):
    """
    Traditional Echo-State Reservoir with a trainable linear readout.
    Only W_out is trained; embedding and reservoir weights are fixed.

    Trainable params: N * vocab_size + vocab_size
    """

    def __init__(self, vocab_size, d_e=16, N=3076):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N

        # Fixed random embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # Fixed recurrent reservoir
        self.rnn = nn.RNN(
            input_size=d_e, hidden_size=N,
            batch_first=True, bias=False, nonlinearity="tanh"
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False
        _init_reservoir(self.rnn)

        # Trainable linear readout  W_out: N -> V
        self.readout = nn.Linear(N, vocab_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reservoir_states(self, idx):
        with torch.no_grad():
            x = self.emb(idx)           # (B, T, d_e)
            out, _ = self.rnn(x)        # (B, T, N)
        return out

    def forward(self, idx):
        states = self._reservoir_states(idx)   # (B, T, N)
        return self.readout(states)             # (B, T, V)


class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC).

    Architecture (1 reservoir, 1-layer attention network F):
      - Fixed random reservoir (echo-state RNN): r_t in R^N
      - Trainable F with 1 hidden layer (ReLU) that maps r_t -> W_att,t in R^{H x N}
          layer 1:  h1 = ReLU(W1 * r + b1)          N -> H
          output :  vec = W2 * h1 + b2               H -> H*N
          reshape:  W_att = reshape(vec, H, N)
      - Intermediate output: ro = W_att @ r           R^H
      - Final logits:        y  = W_out @ ro          R^V

    Trainable params ≈ 200k with default N=130, H=38, V=65:
      net_h1:  N*H + H  =   4,978
      net_out: H*H*N+HN = 192,660
      readout: H*V + V  =   2,535
      Total             = 200,173
    """

    def __init__(self, vocab_size, d_e=16, N=130, H=38):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.N = N
        self.H = H

        # Fixed random embedding
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # Fixed recurrent reservoir
        self.rnn = nn.RNN(
            input_size=d_e, hidden_size=N,
            batch_first=True, bias=False, nonlinearity="tanh"
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False
        _init_reservoir(self.rnn)

        # Trainable 1-layer attention network F: r (N,) -> W_att (H, N)
        self.net_h1  = nn.Linear(N, H)       # layer 1: N -> H
        self.net_out = nn.Linear(H, H * N)   # output:  H -> H*N  (will reshape to H x N)

        # Trainable final readout: ro (H,) -> logits (V,)
        self.readout = nn.Linear(H, vocab_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reservoir_states(self, idx):
        with torch.no_grad():
            x = self.emb(idx)           # (B, T, d_e)
            out, _ = self.rnn(x)        # (B, T, N)
        return out

    def forward(self, idx):
        states = self._reservoir_states(idx)    # (B, T, N)
        B, T, N = states.shape

        # --- 1-layer attention network F ---
        h1   = F.relu(self.net_h1(states))      # (B, T, H)
        vec  = self.net_out(h1)                 # (B, T, H*N)
        W_att = vec.view(B, T, self.H, self.N)  # (B, T, H, N)

        # --- Apply attention weights to reservoir state ---
        # ro = W_att @ r  =>  (B, T, H, N) x (B, T, N, 1) = (B, T, H, 1)
        ro = torch.matmul(W_att, states.unsqueeze(-1)).squeeze(-1)  # (B, T, H)

        # --- Final projection to logits ---
        return self.readout(ro)                  # (B, T, V)
