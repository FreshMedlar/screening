#!/usr/bin/env python3
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

# Configure ROCm AMD GPU environment variables
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

# Append parent directory to sys.path to import base AERC model
sys.path.append(str(Path(__file__).parent.parent))
from base.aerc_simplified import AERC

# ---------------------------------------------------------------------------
# Dataset Definition
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
# Evaluation and Generation Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches: int = 50):
    """Evaluates cross-entropy loss over validation data in batches."""
    model.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(idx=x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)

@torch.no_grad()
def generate_sample(model, chars, seed_text, seq_len, max_new=200, temperature=0.8, device="cuda"):
    """Generates text autoregressively using AERC."""
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in seed_text.lower() if c in stoi]
    
    pad_id = stoi.get(' ', 0)
    if len(tokens) < seq_len:
        tokens = [pad_id] * (seq_len - len(tokens)) + tokens
        
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    for _ in range(max_new):
        x_cond = x[:, -seq_len:]
        logits = model(idx=x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
        
    return "".join(itos.get(t.item(), "?") for t in x[0, seq_len:])

# ---------------------------------------------------------------------------
# Main Training Run
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AERC 155k normal on-the-fly training.")
    parser.add_argument("--data", type=str, default="/home/medlar/Projects/Shakespeare_Res/data/shakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    print(f"  vocab_size={vocab_size} (Expected standard: 59) | train_size={len(train_ds):,} | val_size={len(val_ds):,}")
    assert vocab_size == 59, f"Expected vocabulary size 59, but got {vocab_size}"
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    print("\nInitializing AERC Model...")
    model = AERC(
        vocab_size=vocab_size,
        d_e=16,
        N=160,
        H=30,
        spectral_radius=args.spectral_radius,
        use_rmsnorm=False
    ).to(device)
    
    trainable_params = model.count_parameters()
    print(f"  Total Trainable Parameters: {trainable_params:,} (Expected: 155,459)")
    assert trainable_params == 155459, f"Parameter mismatch! Expected 155,459, got {trainable_params:,}"
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    print("\nStarting On-the-Fly Training (10,000 steps)...")
    print("=" * 75)
    
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
            logits = model(idx=x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            train_losses.append(loss.item())
            
            if step % args.log_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append((step, val_loss))
                ppl = math.exp(min(val_loss, 20))
                elapsed = time.time() - start_time
                print(f"Step {step:5d}/{args.max_steps} | train loss {loss.item():.4f} | "
                      f"val loss {val_loss:.4f} | ppl {ppl:.2f} | {elapsed:.1f}s")
                      
    print("\nTraining completed.")
    print("=" * 75)
    
    # Generate some text
    print("\nGenerated Shakespeare Sample:")
    print("-" * 50)
    seed = "romio:\n"
    generated_text = generate_sample(model, chars, seed, args.seq_len, max_new=200, device=device)
    print(seed + generated_text)
    print("=" * 75)
    
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
            
        ax.plot(smooth(train_losses), label="Train Loss (smoothed)", color="dodgerblue", alpha=0.7)
        if val_losses:
            steps, vloss = zip(*val_losses)
            ax.plot(steps, vloss, "o-", label="Validation Loss", color="darkorange")
            
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("AERC 155k Parameter Normal Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        out_dir = Path(__file__).parent
        out_path = out_dir / "aerc_ablation_loss.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Loss plot saved to {out_path}")
    except Exception as e:
        print(f"\n⚠ Failed to plot loss: {e}")

if __name__ == "__main__":
    main()
