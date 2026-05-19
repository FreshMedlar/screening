#!/usr/bin/env python3
"""
Train & compare Classic Reservoir vs. AERC on character-level tinyshakespeare.

This uses a training regime identical to train_compare.py for standardized comparison,
differing from the sharded methodology originally used in the AERC paper.
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

# Import models from att_res
from att_res_models import ClassicReservoir, AERC


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
    parser = argparse.ArgumentParser(description="Classic Reservoir vs AERC comparison")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/Tesi_EvoRes/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_classic", action="store_true", help="Skip training and comparing the Classic Reservoir")

    # --- Model size ---
    parser.add_argument("--d_e", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--N_classic", type=int, default=3076, help="Reservoir Size for Classic (~200k params)")
    parser.add_argument("--N_aerc", type=int, default=130, help="Reservoir Size for AERC (~200k params)")
    parser.add_argument("--H_aerc", type=int, default=38, help="Hidden Size for AERC (~200k params)")

    args = parser.parse_args()
    device = args.device

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
    if not args.skip_classic:
        print("\n" + "=" * 70)
        print("CLASSIC RESERVOIR")
        print("=" * 70)
        cr_model = ClassicReservoir(
            vocab_size=vocab_size,
            d_e=args.d_e,
            N=args.N_classic,
        ).to(device)
        cr_params = cr_model.count_parameters()
        print(f"  Parameters: {cr_params:,}")
        print(f"  Config: d_e={args.d_e}, N={args.N_classic}")
    else:
        cr_model = None
        cr_params = 0

    print("\n" + "=" * 70)
    print("AERC  (Attention-Enhanced Reservoir Computing)")
    print("=" * 70)
    aerc_model = AERC(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N_aerc,
        H=args.H_aerc,
    ).to(device)
    aerc_params = aerc_model.count_parameters()
    print(f"  Parameters: {aerc_params:,}")
    print(f"  Config: d_e={args.d_e}, N={args.N_aerc}, H={args.H_aerc}")

    # --- Train Classic Reservoir ---
    if not args.skip_classic:
        print("\n" + "=" * 70)
        print("Training CLASSIC RESERVOIR ...")
        print("=" * 70)
        cr_train, cr_val = train_model(cr_model, train_loader, val_loader, device,
                                        args, "Classic Res")
    else:
        cr_train, cr_val = [], []

    # --- Train AERC ---
    print("\n" + "=" * 70)
    print("Training AERC ...")
    print("=" * 70)
    aerc_train, aerc_val = train_model(aerc_model, train_loader, val_loader, device, args, "AERC")

    # --- Generate samples ---
    seed_text = "ROMEO:\n"
    if not args.skip_classic:
        print("\n" + "=" * 70)
        print("Generated text (Classic Reservoir):")
        print("-" * 40)
        print(generate(cr_model, chars, seed_text, args.seq_len, device=device))

    print("\n" + "-" * 40)
    print("Generated text (AERC):")
    print("-" * 40)
    print(generate(aerc_model, chars, seed_text, args.seq_len, device=device))

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
        if not args.skip_classic:
            ax.plot(smooth(cr_train), label=f"Classic Res ({cr_params:,} params)", alpha=0.9)
        ax.plot(smooth(aerc_train), label=f"AERC ({aerc_params:,} params)", alpha=0.9)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation loss
        ax = axes[1]
        if not args.skip_classic and cr_val:
            cr_steps, cr_vloss = zip(*cr_val)
            ax.plot(cr_steps, cr_vloss, "o-",
                    label=f"Classic Res ({cr_params:,} params)")
        if aerc_val:
            aerc_steps, aerc_vloss = zip(*aerc_val)
            ax.plot(aerc_steps, aerc_vloss, "s-",
                    label=f"AERC ({aerc_params:,} params)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation perplexity on secondary y-axis
        ax2 = ax.twinx()
        if not args.skip_classic and cr_val:
            cr_ppl = [math.exp(min(v, 20)) for _, v in cr_val]
            ax2.plot(cr_steps, cr_ppl, "o--", alpha=0.4, color="C0")
        if aerc_val:
            aerc_ppl = [math.exp(min(v, 20)) for _, v in aerc_val]
            ax2.plot(aerc_steps, aerc_ppl, "s--", alpha=0.4, color="C1")
        ax2.set_ylabel("Perplexity", alpha=0.5)

        plt.suptitle("Classic Reservoir vs. AERC  ·  Character-level tinyshakespeare",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "comparison_att_res.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Comparison plot saved to {out_path}")

    except ImportError:
        print("\n⚠  matplotlib not installed — skipping plot.")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if not args.skip_classic:
        cr_final = cr_val[-1][1] if cr_val else float("inf")
        print(f"  Classic Reservoir  — {cr_params:,} params  | final val loss: {cr_final:.4f} "
              f"| ppl: {math.exp(min(cr_final, 20)):.2f}")
              
    aerc_final = aerc_val[-1][1] if aerc_val else float("inf")
    print(f"  AERC               — {aerc_params:,} params  | final val loss: {aerc_final:.4f} "
          f"| ppl: {math.exp(min(aerc_final, 20)):.2f}")
          
    if not args.skip_classic:
        if aerc_final < cr_final:
            pct = (1 - aerc_params / cr_params) * 100
            print(f"\n  → AERC achieves lower validation loss"
                  f" with {pct:+.1f}% parameter difference compared to Classic Reservoir")
        else:
            print(f"\n  → Classic Reservoir achieves lower validation loss at this scale")


if __name__ == "__main__":
    main()
