#!/usr/bin/env python3
"""
Replication training script for the 155k parameter Attention-Enhanced Reservoir Computing (AERC) model.
Matches the paper specifications exactly:
- No intrinsic plasticity, no RMSNorm, no SwiGLU (removed from codebase), no static head, no feedback connection.
- Reservoir size N = 160, Hidden size H = 30, embedding dimension d = 16.
- Total trainable parameters: exactly 155,459.
- Character-level next-character prediction task with lowercased Shakespeare (V = 59).
- Dynamic sequence training on batch sequence segments (no sharding, 90/10 train/val split).
- Uses ROCm AMD GPU environment settings.
"""

import os
# Configure ROCm environment variables for consumer AMD GPU compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

import sys
import time
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Suppress PyTorch and inductor warnings
import warnings
import logging
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, chars: list, seq_len: int):
        self.seq_len = seq_len
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_data(path: str, seq_len: int, val_frac: float = 0.1):
    """Loads the text file, lowercases it, and performs a 90/10 train/val split."""
    text = Path(path).read_text(encoding="utf-8").lower()
    chars = sorted(set(text))
    vocab_size = len(chars)
    split = int(len(text) * (1 - val_frac))
    train_ds = CharDataset(text[:split], chars, seq_len)
    val_ds = CharDataset(text[split:], chars, seq_len)
    return train_ds, val_ds, chars, vocab_size


# ---------------------------------------------------------------------------
# AERC Model Definition (Strict Paper Implementation)
# ---------------------------------------------------------------------------
class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) model matching the paper exactly.
    """
    def __init__(self, N: int = 160, H: int = 30, vocab_size: int = 59, d_e: int = 16, spectral_radius: float = 0.95):
        super().__init__()
        self.N = N
        self.H = H
        self.vocab_size = vocab_size
        self.d_e = d_e

        # 1. Fixed random input character embedding (requires_grad = False)
        self.emb = nn.Embedding(vocab_size, d_e)
        self.emb.weight.requires_grad = False

        # 2. Fixed recurrent reservoir (nn.RNN without bias and requires_grad = False)
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=False,
            nonlinearity="tanh",
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False

        # Initialize and scale recurrent weights for the echo state property
        with torch.no_grad():
            # W_in: input projection (normal distribution with std 0.1)
            nn.init.normal_(self.rnn.weight_ih_l0, mean=0.0, std=0.1)
            # W_res: recurrent connection (normal distribution with std 1/sqrt(N))
            nn.init.normal_(self.rnn.weight_hh_l0, mean=0.0, std=1.0 / math.sqrt(N))
            eigenvalues = torch.linalg.eigvals(self.rnn.weight_hh_l0)
            max_eig = torch.max(torch.abs(eigenvalues)).item()
            if max_eig > 0:
                self.rnn.weight_hh_l0.mul_(spectral_radius / max_eig)

        # 3. Trainable Attention Network F:
        # Consists of a single hidden layer equipped with a ReLU activation mapping r_t (N) to W_att (H * N)
        self.fc1 = nn.Linear(N, H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(H, H * N)

        # 4. Trainable Final Readout:
        # Maps intermediate projected vector r_o (H) to vocabulary logits (vocab_size)
        self.W_out = nn.Linear(H, vocab_size)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx shape: (Batch, SeqLen)
        with torch.no_grad():
            x = self.emb(idx)               # (Batch, SeqLen, d_e)
            r_l, _ = self.rnn(x)            # (Batch, SeqLen, N)

        B, L, N = r_l.shape
        r_l_flat = r_l.reshape(B * L, N)

        # 1. Attention Network (F)
        att_h = self.relu(self.fc1(r_l_flat)) # (B * L, H)
        W_att = self.fc2(att_h)               # (B * L, H * N)
        W_att = W_att.reshape(-1, self.H, self.N) # (B * L, H, N)

        # 2. Intermediate Projection: r_ol = W_att * r_l
        r_l_unsqueeze = r_l_flat.unsqueeze(-1) # (B * L, N, 1)
        r_ol = torch.bmm(W_att, r_l_unsqueeze).squeeze(-1) # (B * L, H)

        # 3. Final Output Mapping
        logits = self.W_out(r_ol)            # (B * L, vocab_size)
        return logits.reshape(B, L, self.vocab_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches: int = 50):
    """Evaluates cross-entropy loss over validation data."""
    model.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate(model, chars, seed_text: str, seq_len: int, max_new: int = 200,
             temperature: float = 0.8, device: str = "cuda"):
    """Autoregressively generates text using the trained model."""
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in seed_text.lower() if c in stoi]

    pad_id = stoi.get(' ', next(iter(stoi.values())))
    if len(tokens) < seq_len:
        tokens = [pad_id] * (seq_len - len(tokens)) + tokens

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new):
        x_cond = x[:, -seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
    model.train()
    return "".join(itos.get(t.item(), "?") for t in x[0, seq_len:])


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper AERC 155k Replication (No Sharding)")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/Shakespeare_Res/data/shakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length (default: 32 as in paper)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default: 1024 as in paper)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 as in paper)")
    parser.add_argument("--spectral_radius", type=float, default=0.95, help="Reservoir spectral radius")
    parser.add_argument("--log_interval", type=int, default=250, help="Logging and evaluation interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data}...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("\n" + "=" * 70)
    print("AERC 155k Parameter Paper Replication Model")
    print("=" * 70)
    
    # Initialize AERC model with paper specs
    model = AERC(
        N=160,
        H=30,
        vocab_size=vocab_size,
        d_e=16,
        spectral_radius=args.spectral_radius
    ).to(device)

    trainable_params = model.count_parameters()
    print(f"  Total Trainable Parameters: {trainable_params:,} (Target: 155,459)")
    assert trainable_params == 155459, f"Parameter mismatch! Expected 155,459, got {trainable_params:,}"
    print(f"  Reservoir: N=160, Hidden: H=30, embedding d=16, activation=ReLU")
    
    # Optimizer (Only trains Attention weights and Readout)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    print("\nStarting Dynamic Training...")
    print("=" * 70)

    train_losses = []
    val_losses = []
    step = 0
    start_time = time.time()

    model.train()
    while step < args.max_steps:
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            # Cross entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append((step, val_loss))
                ppl = math.exp(min(val_loss, 20))
                print(f"  step {step:5d}/{args.max_steps} | "
                      f"train {loss.item():.4f} | val {val_loss:.4f} | "
                      f"ppl {ppl:.2f} | {elapsed:.1f}s")

    # Generate sample text
    print("\n" + "=" * 70)
    print("Generated text (AERC Paper Replica):")
    print("-" * 40)
    seed_text = "ROMEO:\n"
    print(seed_text + generate(model, chars, seed_text, args.seq_len, device=device))
    print("=" * 70)

    # Plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        def smooth(vals, w=50):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        ax.plot(smooth(train_losses), label="Train Loss (smoothed)", alpha=0.7)
        if val_losses:
            steps, vloss = zip(*val_losses)
            ax.plot(steps, vloss, "o-", label="Validation Loss", color="C1")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("AERC 155k Parameter Paper Replication Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "replicate_155k_loss.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Training plot saved to {out_path}")
    except ImportError:
        print("\n⚠ matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
