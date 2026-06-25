#!/usr/bin/env python3
"""
Comparison training script: AERC-IP  vs  AERC-IP + Ridge Hybrid.

Model A: AERC-IP (same as base/, AdamW-only).
Model B: AERC-IP + Dual Ridge Hybrid.
    - AdamW trains the inner AERC-IP (identical to Model A).
    - After AdamW: fit Ridge-R (reservoir → logits) independently.
    - After Ridge-R: fit Ridge-M (concat(logits_A, logits_R) → logits) independently.

The two models are trained sequentially. Validation losses are plotted together.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

# ROCm environment variables (must be set before importing torch)
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Suppress noisy PyTorch warnings
import warnings
import logging
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Local imports (all in the same 2_way/ directory)
sys.path.insert(0, os.path.dirname(__file__))
from aerc_ip import AERC as AERC_IP, pretrain_reservoir_ip
from aerc_ip_hybrid import AERCHybrid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """Character-level language-modelling dataset."""

    def __init__(self, text: str, chars: list, seq_len: int):
        self.seq_len = seq_len
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor(
            [self.stoi[c] for c in text if c in self.stoi], dtype=torch.long
        )

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
    """Loads text file (preserving case) and splits into train/val datasets."""
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    split = int(len(text) * (1 - val_frac))
    train_ds = CharDataset(text[:split], chars, seq_len)
    val_ds   = CharDataset(text[split:], chars, seq_len)
    return train_ds, val_ds, chars, vocab_size


@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches: int = 50, use_bf16: bool = False):
    """Evaluates cross-entropy loss over validation data."""
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
def estimate_loss_hybrid(model, dataloader, device, max_batches: int = 50, use_bf16: bool = False):
    """
    Evaluates cross-entropy loss for the full hybrid model (requires fitted ridge readouts).
    Uses model.forward() which goes through Ridge-M.
    """
    model.eval()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            logits = model(idx=x)                            # full hybrid path
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate(model, chars, seed_text: str, seq_len: int, max_new: int = 200,
             temperature: float = 0.8, device: str = "cuda", use_adamw_only: bool = False):
    """Autoregressively generates text from a seed string."""
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
        if use_adamw_only:
            logits = model.forward_adamw(x_cond)
        else:
            logits = model(idx=x_cond)
        logits = logits[:, -1, :] / temperature
        probs  = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
    model.train()
    return "".join(itos.get(t.item(), "?") for t in x[0, seq_len:])


def _make_lr_lambda(warmup_steps, stable_steps, cooldown_steps, max_steps, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        elif step < warmup_steps + stable_steps:
            progress = (step - warmup_steps) / stable_steps
            return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))
        else:
            remaining = max_steps - step
            return min_lr * remaining / max(cooldown_steps, 1)
    return lr_lambda


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_aerc_ip(model, label, train_loader, val_loader, args, device, use_bf16):
    """
    Standard AdamW training loop for AERC-IP (Model A).
    Returns (train_losses, val_losses) where val_losses = list of (step, loss).
    """
    print(f"\n{'='*70}")
    print(f"Training {label}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    warmup_steps   = min(args.warmup_steps,   args.max_steps // 10)
    cooldown_steps = min(args.cooldown_steps,  args.max_steps // 3)
    stable_steps   = max(args.max_steps - warmup_steps - cooldown_steps, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _make_lr_lambda(warmup_steps, stable_steps, cooldown_steps, args.max_steps, args.min_lr_ratio),
    )

    train_losses, val_losses = [], []
    step = 0
    start_time = time.time()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    model.train()
    while step < args.max_steps:
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                logits = model(idx=x)
                loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                elapsed  = time.time() - start_time
                val_loss = estimate_loss(model, val_loader, device, use_bf16=use_bf16)
                val_losses.append((step, val_loss))
                ppl = math.exp(min(val_loss, 20))
                if hasattr(model, "ip_a"):
                    with torch.no_grad():
                        mean_a = model.ip_a.mean().item()
                        mean_b = model.ip_b.mean().item()
                        std_a  = model.ip_a.std().item()
                        std_b  = model.ip_b.std().item()
                    print(f"  step {step:5d}/{args.max_steps} | "
                          f"train {loss.item():.4f} | val {val_loss:.4f} | ppl {ppl:.2f} | "
                          f"ip_a {mean_a:.4f}±{std_a:.4f} | ip_b {mean_b:.4f}±{std_b:.4f} | "
                          f"{elapsed:.1f}s")
                else:
                    print(f"  step {step:5d}/{args.max_steps} | "
                          f"train {loss.item():.4f} | val {val_loss:.4f} | ppl {ppl:.2f} | "
                          f"{elapsed:.1f}s")

    return train_losses, val_losses


# train_hybrid removed, since we now share the model_ip training directly.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AERC-IP vs AERC-IP + Ridge Hybrid comparison"
    )
    # Data & architecture
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--spectral_radius", type=float, default=0.95)
    parser.add_argument("--d_e",          type=int,   default=16)
    parser.add_argument("--N",            type=int,   default=160)
    parser.add_argument("--H",            type=int,   default=30)

    # AdamW training
    parser.add_argument("--max_steps",      type=int,   default=2000)
    parser.add_argument("--lr",             type=float, default=5e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-3)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--warmup_steps",   type=int,   default=150)
    parser.add_argument("--cooldown_steps", type=int,   default=300)
    parser.add_argument("--min_lr_ratio",   type=float, default=0.1)
    parser.add_argument("--log_interval",   type=int,   default=100)

    # bfloat16
    parser.add_argument("--bf16",    action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")

    # IP pre-training
    parser.add_argument("--ip_epochs", type=int,   default=10)
    parser.add_argument("--ip_lr",     type=float, default=1e-5)
    parser.add_argument("--ip_mu",     type=float, default=-0.1)
    parser.add_argument("--ip_sigma",  type=float, default=0.3)
    parser.add_argument("--ip_chars",  type=int,   default=10000)

    # Ridge regression
    parser.add_argument("--ridge_alpha",       type=float, default=1.0,
                        help="Ridge regularisation strength (same for R and M)")
    parser.add_argument("--ridge_max_batches", type=int,   default=200,
                        help="Max training batches used for each ridge fit")

    # Misc
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--device",    type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_only", action="store_true",
                        help="Quick smoke-test with tiny settings; exits after verification")

    args = parser.parse_args()
    device = args.device

    if args.test_only:
        print(">>> Fast verification test. Overriding config.")
        args.max_steps          = 10
        args.batch_size         = 16
        args.seq_len            = 32
        args.log_interval       = 2
        args.ip_epochs          = 1
        args.ip_chars           = 500
        args.ridge_max_batches  = 5

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data}...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    use_bf16 = args.bf16 and device.startswith("cuda")

    # ------------------------------------------------------------------
    # Model A: AERC-IP  (baseline)
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_ip = AERC_IP(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
    ).to(device)

    print(f"\nModel A  –  AERC-IP")
    print(f"  Trainable params: {model_ip.count_parameters():,}")

    print("\nRunning IP pre-training for Model A...")
    pretrain_reservoir_ip(
        model=model_ip,
        dataset=train_ds,
        ip_chars=args.ip_chars,
        batch_size=args.batch_size,
        eta=args.ip_lr,
        mu=args.ip_mu,
        sigma=args.ip_sigma,
        nepochs=args.ip_epochs,
        device=device,
    )

    ip_train_losses, ip_val_losses = train_aerc_ip(
        model_ip, "AERC-IP (baseline)",
        train_loader, val_loader, args, device, use_bf16,
    )

    # ------------------------------------------------------------------
    # Model B: AERC-IP + Ridge Hybrid
    # ------------------------------------------------------------------
    model_hybrid = AERCHybrid(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
    ).to(device)

    print(f"\nModel B  –  AERC-IP + Ridge Hybrid")
    print(f"  AdamW-trainable params: {model_hybrid.count_adamw_parameters():,}")
    print(f"  Ridge-R params:         {model_hybrid.inner.N * vocab_size + vocab_size:,}  (not trained by AdamW)")
    print(f"  Ridge-M params:         {2 * vocab_size * vocab_size + vocab_size:,}  (not trained by AdamW)")

    print("\nCopying trained weights from Model A (AERC-IP) to Model B...")
    model_hybrid.inner.load_state_dict(model_ip.state_dict())

    # --- Phase 2: Fit Ridge-R ---
    print("\n[Phase 2] Fitting Ridge-R (reservoir → logits)...")
    model_hybrid.fit_ridge_R(
        train_loader, device,
        alpha=args.ridge_alpha,
        max_batches=args.ridge_max_batches,
    )

    # --- Phase 3: Fit Ridge-M ---
    print("\n[Phase 3] Fitting Ridge-M (meta-readout)...")
    model_hybrid.fit_ridge_M(
        train_loader, device,
        alpha=args.ridge_alpha,
        max_batches=args.ridge_max_batches,
        use_bf16=use_bf16,
    )

    # --- Final evaluation: full hybrid model ---
    print("\n[Eval] Computing final hybrid validation loss...")
    hybrid_final_val = estimate_loss_hybrid(model_hybrid, val_loader, device, use_bf16=use_bf16)
    hybrid_ppl = math.exp(min(hybrid_final_val, 20))
    print(f"  Final hybrid val loss: {hybrid_final_val:.4f}  |  ppl: {hybrid_ppl:.2f}")

    # ------------------------------------------------------------------
    # Generate text samples
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generated text samples:")
    print("-" * 70)
    seed_text = "ROMEO:\n"

    print("Model A — AERC-IP (baseline):")
    print(seed_text + generate(model_ip, chars, seed_text, args.seq_len, device=device))
    print("-" * 40)
    print("Model B — AERC-IP + Ridge Hybrid (AdamW head):")
    print(seed_text + generate(model_hybrid, chars, seed_text, args.seq_len,
                               device=device, use_adamw_only=True))
    print("-" * 40)
    print("Model B — AERC-IP + Ridge Hybrid (full hybrid):")
    print(seed_text + generate(model_hybrid, chars, seed_text, args.seq_len, device=device))
    print("=" * 70)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def smooth(vals, w=5):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # --- Left: training loss (AdamW phase) ---
        axes[0].plot(smooth(ip_train_losses),
                     label="AdamW train loss (shared)", color="C0", alpha=0.6)
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Right: validation loss ---
        if ip_val_losses:
            steps_ip, vloss_ip = zip(*ip_val_losses)
            axes[1].plot(steps_ip, vloss_ip, "o-",
                         label="AERC-IP val loss (shared)", color="C0", linewidth=2)

        # Mark final hybrid val loss as a horizontal dotted line
        axes[1].axhline(
            hybrid_final_val,
            color="C2", linestyle=":",
            linewidth=2.5,
            label=f"Hybrid (full, post-ridge) val = {hybrid_final_val:.4f}",
        )

        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Cross-entropy loss")
        axes[1].set_title("Validation Loss Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        n_params = model_ip.count_parameters()
        plt.suptitle(
            f"AERC-IP vs AERC-IP + Ridge Hybrid  ({n_params:,} AdamW params each)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        out_path = os.path.join(os.path.dirname(__file__), "aerc_hybrid_comparison.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Comparison plot saved to {out_path}")

    except ImportError:
        print("\n⚠  matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
