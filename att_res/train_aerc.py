#!/usr/bin/env python3
"""
Training script for Attention-Enhanced Reservoir Computing (AERC).

Trains the AERC model matching the methodology and printing style of train_att_res.py:
- Standard contiguous 90/10 train/val split (no shards or passes).
- Dynamic sequence training on batch sequence segments (no memory-intensive precomputation).
- Clean logging showing steps, loss, validation loss, perplexity, and elapsed time.
- Optional Intrinsic Plasticity (IP) pre-training.
- Generates text autoregressively and saves a training loss plot.
"""

import os
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

# Import AERC architecture and IP helper
from aerc_arch import AERC, pretrain_reservoir_ip

# Suppress PyTorch warnings
import warnings
import logging
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    """Character-level language-modelling dataset."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: str, seq_len: int, val_frac: float = 0.1):
    """Loads the text file and performs a standard train/val split."""
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    split = int(len(text) * (1 - val_frac))
    train_ds = CharDataset(text[:split], chars, seq_len)
    val_ds = CharDataset(text[split:], chars, seq_len)
    return train_ds, val_ds, chars, vocab_size


@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches: int = 50, use_bf16: bool = False):
    """Evaluates cross-entropy loss over validation data in batches."""
    model.eval()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            logits = model(idx=x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def fit_static_head_ridge(
    model,
    dataloader: torch.utils.data.DataLoader,
    alpha: float,
    device: str,
    max_batches: int = 0,
) -> None:
    """
    Fits model.static_head analytically via ridge regression.
    """
    model.eval()
    N, V = model.N, model.vocab_size

    # Bias-augmented accumulators: (N+1, N+1) and (N+1, V)
    XtX = torch.zeros(N + 1, N + 1, device=device, dtype=torch.float64)
    XtY = torch.zeros(N + 1, V,     device=device, dtype=torch.float64)

    for i, (x, y) in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        states        = model.compute_reservoir_states(x)         # (B, T, N)
        states_flat   = states.reshape(-1, N)
        states_normed = model.state_norm(states_flat).double()    # (B*T, N), float64

        # Bias-augmented design matrix: (B*T, N+1)
        ones = torch.ones(states_normed.shape[0], 1, device=device, dtype=torch.float64)
        S    = torch.cat([states_normed, ones], dim=-1)

        # One-hot targets: (B*T, V)
        Y = F.one_hot(y.reshape(-1), num_classes=V).to(torch.float64)

        # Incremental accumulation
        XtX.addmm_(S.T, S)
        XtY.addmm_(S.T, Y)

    # Solve: (XtX + alpha*I) W = XtY
    XtX.diagonal().add_(alpha)
    W_aug = torch.linalg.solve(XtX, XtY).float()  # (N+1, V)

    with torch.no_grad():
        model.static_head.weight.copy_(W_aug[:N, :].T)  # (V, N)
        model.static_head.bias.copy_(W_aug[N, :])        # (V,)

    model.train()
    w_norm = model.static_head.weight.norm().item()
    print(f"  Ridge fit complete — alpha={alpha:.1e} | ||W||_F = {w_norm:.4f}")


@torch.no_grad()
def estimate_static_loss(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 50,
) -> float:
    """Evaluates only static_head (no AERC correction) to measure the phase-1 baseline."""
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        states        = model.compute_reservoir_states(x)    # (B, T, N)
        states_flat   = states.reshape(-1, model.N)
        states_normed = model.state_norm(states_flat)        # (B*T, N)
        static_logits = model.static_head(states_normed)     # (B*T, V)
        loss = F.cross_entropy(static_logits, y.reshape(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def generate(model, chars, seed_text: str, seq_len: int, max_new: int = 200,
             temperature: float = 0.8, device: str = "cuda"):
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in seed_text if c in stoi]

    pad_id = stoi.get(' ', next(iter(stoi.values())))
    if len(tokens) < seq_len:
        tokens = [pad_id] * (seq_len - len(tokens)) + tokens

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new):
        x_cond = x[:, -seq_len:]
        logits = model(idx=x_cond)
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
    parser = argparse.ArgumentParser(description="AERC Training")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps (default: 2000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for LR schedule")
    parser.add_argument("--cooldown_steps", type=int, default=200, help="Linear cooldown steps at end of training")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR ratio")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", help="Disable bfloat16")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Intrinsic Plasticity (IP)
    parser.add_argument("--no_ip", action="store_true", help="Disable Intrinsic Plasticity")
    parser.add_argument("--ip_epochs", type=int, default=11, help="IP epochs")
    parser.add_argument("--ip_lr", type=float, default=1e-5, help="IP learning rate")
    parser.add_argument("--ip_mu", type=float, default=0.0, help="IP target mean")
    parser.add_argument("--ip_sigma", type=float, default=0.2, help="IP target standard deviation")
    parser.add_argument("--ip_chars", type=int, default=10000, help="IP chars count")
    
    # ESN Reservoir & Architecture Hyperparameters
    parser.add_argument("--spectral_radius", type=float, default=0.95, help="ESN spectral radius")
    parser.add_argument("--fb_scaling", type=float, default=0.0, help="Feedback scaling")
    parser.add_argument("--dropout", type=float, default=0.0, help="Readout network dropout")
    parser.add_argument("--leaking_rate", type=float, default=1.0, help="Leaking rate")
    parser.add_argument("--d_e", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--N_aerc", type=int, default=55, help="Reservoir size N")
    parser.add_argument("--H_aerc", type=int, default=51, help="Attention dimension H")
    parser.add_argument("--activation", type=str, default="silu", choices=["silu", "tanh", "relu"])

    # Ridge regression (Phase 1)
    parser.add_argument("--ridge_alpha", type=float, default=1e-4, help="Ridge regularisation strength")
    parser.add_argument("--ridge_chars", type=int, default=20_000, help="Chars used for ridge regression")

    # Fast testing flag
    parser.add_argument("--test_only", action="store_true", help="Run a quick validation check and exit")

    args = parser.parse_args()
    device = args.device

    if args.test_only:
        print(">>> Fast verification test requested. Overriding configuration parameters.")
        args.max_steps = 200
        args.batch_size = 32
        args.ip_epochs = 1
        args.ip_chars = 1000
        args.log_interval = 50
        args.ridge_chars = 1000

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data}...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Build AERC model
    print("\n" + "=" * 70)
    print("AERC  (Attention-Enhanced Reservoir Computing)")
    print("=" * 70)
    model = AERC(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N_aerc,
        H=args.H_aerc,
        spectral_radius=args.spectral_radius,
        fb_scaling=args.fb_scaling,
        dropout=args.dropout,
        leaking_rate=args.leaking_rate,
        activation=args.activation,
    ).to(device)

    aerc_params = model.count_parameters()
    print(f"  Parameters: {aerc_params:,}")
    print(f"  Config: d_e={args.d_e}, N={args.N_aerc}, H={args.H_aerc}, SR={args.spectral_radius}, activation={args.activation}")

    # Intrinsic Plasticity (IP) Pre-training
    if not args.no_ip:
        print("\n" + "=" * 70)
        print("IP Pre-training AERC ...")
        print("=" * 70)
        
        ip_limit = min(args.ip_chars, len(train_ds))
        ip_subset = torch.utils.data.Subset(train_ds, range(ip_limit))
        ip_loader = DataLoader(ip_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        pretrain_reservoir_ip(
            model=model,
            dataloader=ip_loader,
            eta=args.ip_lr,
            mu=args.ip_mu,
            sigma=args.ip_sigma,
            nepochs=args.ip_epochs,
            device=device
        )

    # Phase 1 — Fit static_head via ridge regression
    chars_per_batch  = args.batch_size * args.seq_len
    ridge_max_batches = math.ceil(args.ridge_chars / chars_per_batch)
    print("\n" + "=" * 70)
    print(f"Phase 1  ─  Fitting static ESN readout (Ridge Regression, ~{args.ridge_chars:,} chars)")
    print("=" * 70)
    fit_static_head_ridge(
        model=model,
        dataloader=train_loader,
        alpha=args.ridge_alpha,
        device=device,
        max_batches=ridge_max_batches,
    )
    static_val_loss = estimate_static_loss(model, val_loader, device)
    static_ppl = math.exp(min(static_val_loss, 20))
    print(f"  Static baseline — val loss: {static_val_loss:.4f} | ppl: {static_ppl:.2f}")

    # Phase 2 — freeze static_head, train AERC correction
    model.set_phase(2)
    aerc_params = model.count_parameters()
    print(f"  Phase 2 trainable parameters: {aerc_params:,}  (static_head frozen)")

    # Optimizer & Scheduler (over phase-2 trainable params only)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    warmup_steps  = min(args.warmup_steps,  args.max_steps // 10)
    cooldown_steps = min(args.cooldown_steps, args.max_steps // 3)
    stable_steps  = max(args.max_steps - warmup_steps - cooldown_steps, 1)
    min_lr        = args.min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        elif step < warmup_steps + stable_steps:
            progress = (step - warmup_steps) / stable_steps
            return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))
        else:
            remaining = args.max_steps - step
            return min_lr * remaining / max(cooldown_steps, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Phase 2 Training Loop
    print("\n" + "=" * 70)
    print("Phase 2  ─  Training AERC attention correction ...")
    print("=" * 70)

    train_losses = []
    val_losses = []
    step = 0
    start_time = time.time()

    use_bf16 = args.bf16 and device.startswith("cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    model.train()
    while step < args.max_steps:
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                logits = model(idx=x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                val_loss = estimate_loss(model, val_loader, device, use_bf16=use_bf16)
                val_losses.append((step, val_loss))
                ppl = math.exp(min(val_loss, 20))
                print(f"  [AERC] step {step:5d}/{args.max_steps} | "
                      f"train {loss.item():.4f} | val {val_loss:.4f} | "
                      f"ppl {ppl:.2f} | {elapsed:.1f}s")

    # Generate sample
    print("\n" + "=" * 70)
    print("Generated text (AERC):")
    print("-" * 40)
    seed_text = "ROMEO:\n"
    print(seed_text + generate(model, chars, seed_text, args.seq_len, device=device))
    print("=" * 70)

    # Plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def smooth(vals, w=50):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        ax = axes[0]
        ax.plot(smooth(train_losses), label=f"AERC correction ({aerc_params:,} params)", alpha=0.9)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Training Loss (Phase 2)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if val_losses:
            steps, vloss = zip(*val_losses)
            ax.plot(steps, vloss, "s-", label=f"AERC ({aerc_params:,} params)", color="C1")
        ax.axhline(static_val_loss, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Static ESN baseline ({static_val_loss:.4f})")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("AERC Training Progress · Character-level tinyshakespeare", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "aerc_training_loss.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Training plot saved to {out_path}")
    except ImportError:
        print("\n⚠ matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
