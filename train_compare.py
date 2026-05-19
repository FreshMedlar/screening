#!/usr/bin/env python3
"""
Train & compare Multiscreen vs. Transformer on character-level tinyshakespeare.

Both models are configured with CPU-friendly sizes and trained under identical
conditions (same data, optimizer, batch size, sequence length, number of steps).
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

from multiscreen import Multiscreen
from transformer_baseline import TransformerLM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    """Character-level language-modelling dataset."""

    def __init__(self, text: str, chars: list, seq_len: int):
        self.seq_len = seq_len
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

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
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    split = int(len(text) * (1 - val_frac))
    train_ds = CharDataset(text[:split], chars, seq_len)
    val_ds = CharDataset(text[split:], chars, seq_len)
    return train_ds, val_ds, chars, vocab_size


@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches: int = 50):
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
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in seed_text if c in stoi]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new):
        x_cond = x[:, -seq_len:]   # keep last seq_len tokens
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
    model.train()
    return "".join(itos.get(t.item(), "?") for t in x[0, len(tokens):])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, args, model_name="Model"):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Warmup-then-constant schedule (matching paper)
    warmup_steps = min(args.warmup_steps, args.max_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    val_losses = []
    step = 0
    epoch = 0
    start_time = time.time()

    model.train()
    while step < args.max_steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.max_steps:
                break

            x, y = x.to(device), y.to(device)
            logits = model(x)
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
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append((step, val_loss))
                ppl = math.exp(min(val_loss, 20))
                print(f"  [{model_name}] step {step:5d}/{args.max_steps} | "
                      f"train {loss.item():.4f} | val {val_loss:.4f} | "
                      f"ppl {ppl:.2f} | {elapsed:.1f}s")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multiscreen vs Transformer comparison")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/Tesi_EvoRes/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (used for both models unless --ms_lr set)")
    parser.add_argument("--ms_lr", type=float, default=0.0625,
                        help="Separate learning rate for Multiscreen (paper uses 0.0625)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # --- Model size (CPU-friendly defaults) ---
    parser.add_argument("--d_e", type=int, default=64,
                        help="Embedding dimension (both models)")
    parser.add_argument("--n_l", type=int, default=4,
                        help="Number of layers (both models)")
    parser.add_argument("--n_h", type=int, default=4,
                        help="Number of heads (both models)")
    parser.add_argument("--d_k", type=int, default=16,
                        help="Key dimension (Multiscreen)")
    parser.add_argument("--d_v", type=int, default=32,
                        help="Value dimension (Multiscreen)")

    args = parser.parse_args()
    device = "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data ---
    print(f"Loading data from {args.data} ...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=0)

    # --- Build models ---
    print("\n" + "=" * 70)
    print("MULTISCREEN  (Screening Is Enough)")
    print("=" * 70)
    ms_model = Multiscreen(
        vocab_size=vocab_size,
        d_e=args.d_e,
        n_l=args.n_l,
        n_h=args.n_h,
        d_k=args.d_k,
        d_v=args.d_v,
    ).to(device)
    ms_params = ms_model.count_parameters()
    print(f"  Parameters: {ms_params:,}")
    print(f"  Config: d_e={args.d_e}, n_l={args.n_l}, n_h={args.n_h}, "
          f"d_k={args.d_k}, d_v={args.d_v}")

    print("\n" + "=" * 70)
    print("TRANSFORMER  (LLaMA-style baseline)")
    print("=" * 70)
    tf_model = TransformerLM(
        vocab_size=vocab_size,
        d_e=args.d_e,
        n_l=args.n_l,
        n_h=args.n_h,
        max_len=args.seq_len + 64,
    ).to(device)
    tf_params = tf_model.count_parameters()
    print(f"  Parameters: {tf_params:,}")
    print(f"  Config: d_e={args.d_e}, n_l={args.n_l}, n_h={args.n_h}")

    # --- Train Multiscreen ---
    print("\n" + "=" * 70)
    print("Training MULTISCREEN ...")
    print("=" * 70)
    ms_lr = args.ms_lr if args.ms_lr is not None else args.lr
    ms_args = argparse.Namespace(**vars(args))
    ms_args.lr = ms_lr
    ms_args.weight_decay = 0.0      # Paper: Multiscreen omits weight decay
    ms_args.grad_clip = 0.0         # Paper: Multiscreen omits grad clipping
    ms_train, ms_val = train_model(ms_model, train_loader, val_loader, device,
                                    ms_args, "Multiscreen")

    # --- Train Transformer ---
    print("\n" + "=" * 70)
    print("Training TRANSFORMER ...")
    print("=" * 70)
    tf_args = argparse.Namespace(**vars(args))
    tf_train, tf_val = train_model(tf_model, train_loader, val_loader, device,
                                    tf_args, "Transformer")

    # --- Generate samples ---
    seed_text = "ROMEO:\n"
    print("\n" + "=" * 70)
    print("Generated text (Multiscreen):")
    print("-" * 40)
    print(generate(ms_model, chars, seed_text, args.seq_len, device=device))

    print("\n" + "-" * 40)
    print("Generated text (Transformer):")
    print("-" * 40)
    print(generate(tf_model, chars, seed_text, args.seq_len, device=device))

    # --- Plot comparison ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss (smoothed)
        def smooth(vals, w=50):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        ax = axes[0]
        ax.plot(smooth(ms_train), label=f"Multiscreen ({ms_params:,} params)", alpha=0.9)
        ax.plot(smooth(tf_train), label=f"Transformer ({tf_params:,} params)", alpha=0.9)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation loss
        ax = axes[1]
        if ms_val:
            ms_steps, ms_vloss = zip(*ms_val)
            ax.plot(ms_steps, ms_vloss, "o-",
                    label=f"Multiscreen ({ms_params:,} params)")
        if tf_val:
            tf_steps, tf_vloss = zip(*tf_val)
            ax.plot(tf_steps, tf_vloss, "s-",
                    label=f"Transformer ({tf_params:,} params)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation perplexity on secondary y-axis
        ax2 = ax.twinx()
        if ms_val:
            ms_ppl = [math.exp(min(v, 20)) for _, v in ms_val]
            ax2.plot(ms_steps, ms_ppl, "o--", alpha=0.4, color="C0")
        if tf_val:
            tf_ppl = [math.exp(min(v, 20)) for _, v in tf_val]
            ax2.plot(tf_steps, tf_ppl, "s--", alpha=0.4, color="C1")
        ax2.set_ylabel("Perplexity", alpha=0.5)

        plt.suptitle("Multiscreen vs. Transformer  ·  Character-level tinyshakespeare",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "comparison.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Comparison plot saved to {out_path}")

    except ImportError:
        print("\n⚠  matplotlib not installed — skipping plot.")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    ms_final = ms_val[-1][1] if ms_val else float("inf")
    tf_final = tf_val[-1][1] if tf_val else float("inf")
    print(f"  Multiscreen  — {ms_params:,} params  | final val loss: {ms_final:.4f} "
          f"| ppl: {math.exp(min(ms_final, 20)):.2f}")
    print(f"  Transformer  — {tf_params:,} params  | final val loss: {tf_final:.4f} "
          f"| ppl: {math.exp(min(tf_final, 20)):.2f}")
    if ms_final < tf_final:
        pct = (1 - ms_params / tf_params) * 100
        print(f"\n  → Multiscreen achieves lower validation loss"
              f" with {pct:+.1f}% parameter difference")
    else:
        print(f"\n  → Transformer achieves lower validation loss at this scale")


if __name__ == "__main__":
    main()
