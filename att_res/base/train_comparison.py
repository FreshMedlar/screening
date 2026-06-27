#!/usr/bin/env python3
"""
Training and comparison script for:
1. AERC Simplified (with RMSNorm)
2. AERC Paper (without RMSNorm)
3. AERC with Intrinsic Plasticity (IP)

Instantiates all three models with identical initial weights (except for model-specific elements),
runs IP pre-training on the IP model, trains all three end-to-end via backpropagation,
and plots a comparative performance graph.
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

# ROCm environment variables
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

# Import AERC architectures
from aerc_simplified import AERC as AERC_Simplified
from aerc_ip import AERC as AERC_IP, pretrain_reservoir_ip

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
    """Loads the text file (preserving case) and performs train/val split."""
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
def generate(model, chars, seed_text: str, seq_len: int, max_new: int = 200,
             temperature: float = 0.8, device: str = "cuda"):
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
        logits = model(idx=x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
    model.train()
    return "".join(itos.get(t.item(), "?") for t in x[0, seq_len:])


def train_model(model, label, train_loader, val_loader, args, device, use_bf16):
    """Utility function to train a single model."""
    print(f"\n--- Training {label} ---")
    
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

    train_losses = []
    val_losses = []
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
                if hasattr(model, "ip_a"):
                    with torch.no_grad():
                        mean_a = model.ip_a.mean().item()
                        mean_b = model.ip_b.mean().item()
                        std_a  = model.ip_a.std().item()
                        std_b  = model.ip_b.std().item()
                    print(f"  step {step:5d}/{args.max_steps} | "
                          f"train {loss.item():.4f} | val {val_loss:.4f} | "
                          f"ppl {ppl:.2f} | "
                          f"ip_a {mean_a:.4f}±{std_a:.4f} | ip_b {mean_b:.4f}±{std_b:.4f} | "
                          f"{elapsed:.1f}s")
                else:
                    print(f"  step {step:5d}/{args.max_steps} | "
                          f"train {loss.item():.4f} | val {val_loss:.4f} | "
                          f"ppl {ppl:.2f} | {elapsed:.1f}s")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AERC Three-Way Model Comparison")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Warmup steps")
    parser.add_argument("--cooldown_steps", type=int, default=300, help="Cooldown steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR ratio")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", help="Disable bfloat16")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # ESN Reservoir & Architecture Hyperparameters
    parser.add_argument("--spectral_radius", type=float, default=0.95, help="ESN spectral radius")
    parser.add_argument("--d_e", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--N", type=int, default=160, help="Reservoir size N")
    parser.add_argument("--H", type=int, default=30, help="Attention hidden dimension H")

    # Intrinsic Plasticity (IP) Hyperparameters
    parser.add_argument("--ip_epochs", type=int, default=10, help="IP pre-training epochs")
    parser.add_argument("--ip_lr", type=float, default=1e-5, help="IP learning rate")
    parser.add_argument("--ip_mu", type=float, default=-0.1, help="IP target mean")
    parser.add_argument("--ip_sigma", type=float, default=0.3, help="IP target std (recommended 0.4-0.6)")
    parser.add_argument("--ip_chars", type=int, default=10000, help="Chars per IP epoch (sequential, no repetition)")

    # Fast testing flag
    parser.add_argument("--test_only", action="store_true", help="Run a quick validation check and exit")

    args = parser.parse_args()
    device = args.device

    if args.test_only:
        print(">>> Fast verification test requested. Overriding configuration parameters.")
        args.max_steps = 10
        args.batch_size = 16
        args.seq_len = 32
        args.log_interval = 2
        args.ip_epochs = 1
        args.ip_chars = 1000

    print(f"Loading data from {args.data}...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size} (Expected standard: 65)")
    print(f"  train_size={len(train_ds):,}  val_size={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Initialize Model A: AERC Simplified (with RMSNorm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_simplified = AERC_Simplified(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        use_rmsnorm=True,
    ).to(device)

    # Initialize Model B: AERC Paper (without RMSNorm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_paper = AERC_Simplified(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        use_rmsnorm=False,
    ).to(device)

    # Initialize Model C: AERC IP (with Intrinsic Plasticity)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_ip = AERC_IP(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        use_rmsnorm=True,
    ).to(device)

    # Initialize Model D: AERC IP (with Intrinsic Plasticity, without RMSNorm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_ip_no_rmsnorm = AERC_IP(
        vocab_size=vocab_size,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        use_rmsnorm=False,
    ).to(device)

    # Copy identical initialization weights for all common layers
    with torch.no_grad():
        for target_model in [model_paper, model_ip, model_ip_no_rmsnorm]:
            target_model.emb.weight.copy_(model_simplified.emb.weight)
            target_model.rnn.weight_ih_l0.copy_(model_simplified.rnn.weight_ih_l0)
            target_model.rnn.weight_hh_l0.copy_(model_simplified.rnn.weight_hh_l0)
            target_model.net_gate.weight.copy_(model_simplified.net_gate.weight)
            target_model.net_gate.bias.copy_(model_simplified.net_gate.bias)
            target_model.net_out.weight.copy_(model_simplified.net_out.weight)
            target_model.net_out.bias.copy_(model_simplified.net_out.bias)
            target_model.readout.weight.copy_(model_simplified.readout.weight)
            target_model.readout.bias.copy_(model_simplified.readout.bias)

    params_simp = model_simplified.count_parameters()
    params_paper = model_paper.count_parameters()
    params_ip = model_ip.count_parameters()
    params_ip_no_rmsnorm = model_ip_no_rmsnorm.count_parameters()
    
    print("\n" + "=" * 70)
    print("AERC Four-Way Model Comparison Setup:")
    print(f"  Model A (Base, with RMSNorm) Trainable Parameters:    {params_simp:,}")
    print(f"  Model B (Base, no RMSNorm) Trainable Parameters:      {params_paper:,}")
    print(f"  Model C (IP, with RMSNorm) Trainable Parameters:      {params_ip:,}")
    print(f"  Model D (IP, no RMSNorm) Trainable Parameters:        {params_ip_no_rmsnorm:,}")
    print(f"  Config: d_e={args.d_e}, N={args.N}, H={args.H}, SR={args.spectral_radius}")
    print("=" * 70)

    # Pre-train Model C (IP)
    print("\nStarting Intrinsic Plasticity (IP) Pre-training for Model C (with RMSNorm)...")
    print("-" * 70)
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

    # Pre-train Model D (IP without RMSNorm)
    print("\nStarting Intrinsic Plasticity (IP) Pre-training for Model D (no RMSNorm)...")
    print("-" * 70)
    pretrain_reservoir_ip(
        model=model_ip_no_rmsnorm,
        dataset=train_ds,
        ip_chars=args.ip_chars,
        batch_size=args.batch_size,
        eta=args.ip_lr,
        mu=args.ip_mu,
        sigma=args.ip_sigma,
        nepochs=args.ip_epochs,
        device=device,
    )

    use_bf16 = args.bf16 and device.startswith("cuda")

    # Train Model A (Simplified)
    simp_train_losses, simp_val_losses = train_model(
        model_simplified, "AERC Simplified (With RMSNorm)", train_loader, val_loader, args, device, use_bf16
    )

    # Train Model B (Paper)
    paper_train_losses, paper_val_losses = train_model(
        model_paper, "AERC Paper (No RMSNorm)", train_loader, val_loader, args, device, use_bf16
    )

    # Train Model C (IP)
    ip_train_losses, ip_val_losses = train_model(
        model_ip, "AERC with Intrinsic Plasticity (IP, With RMSNorm)", train_loader, val_loader, args, device, use_bf16
    )

    # Train Model D (IP, No RMSNorm)
    ip_no_norm_train_losses, ip_no_norm_val_losses = train_model(
        model_ip_no_rmsnorm, "AERC with Intrinsic Plasticity (IP, No RMSNorm)", train_loader, val_loader, args, device, use_bf16
    )

    # Generate samples from all four
    print("\n" + "=" * 70)
    print("Generated text comparisons:")
    print("-" * 70)
    seed_text = "ROMEO:\n"
    print("AERC Simplified (With RMSNorm):")
    print(seed_text + generate(model_simplified, chars, seed_text, args.seq_len, device=device))
    print("-" * 40)
    print("AERC Paper (No RMSNorm):")
    print(seed_text + generate(model_paper, chars, seed_text, args.seq_len, device=device))
    print("-" * 40)
    print("AERC with Intrinsic Plasticity (IP, With RMSNorm):")
    print(seed_text + generate(model_ip, chars, seed_text, args.seq_len, device=device))
    print("-" * 40)
    print("AERC with Intrinsic Plasticity (IP, No RMSNorm):")
    print(seed_text + generate(model_ip_no_rmsnorm, chars, seed_text, args.seq_len, device=device))
    print("=" * 70)

    # Plotting comparisons
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        def smooth(vals, w=5):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        # Plot training losses
        axes[0].plot(smooth(simp_train_losses), label="Base (with RMSNorm)", color="C0", alpha=0.6)
        axes[0].plot(smooth(paper_train_losses), label="Base (no RMSNorm)", color="C1", alpha=0.6)
        axes[0].plot(smooth(ip_train_losses), label="IP (with RMSNorm)", color="C2", alpha=0.6)
        axes[0].plot(smooth(ip_no_norm_train_losses), label="IP (no RMSNorm)", color="C3", alpha=0.6)
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].set_title("Training Loss Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot validation/test losses
        if simp_val_losses:
            steps, simp_vloss = zip(*simp_val_losses)
            axes[1].plot(steps, simp_vloss, "o-", label="Base (with RMSNorm) Test", color="C0", linewidth=2)
        if paper_val_losses:
            steps, paper_vloss = zip(*paper_val_losses)
            axes[1].plot(steps, paper_vloss, "s-", label="Base (no RMSNorm) Test", color="C1", linewidth=2)
        if ip_val_losses:
            steps, ip_vloss = zip(*ip_val_losses)
            axes[1].plot(steps, ip_vloss, "d-", label="IP (with RMSNorm) Test", color="C2", linewidth=2)
        if ip_no_norm_val_losses:
            steps, ip_no_norm_vloss = zip(*ip_no_norm_val_losses)
            axes[1].plot(steps, ip_no_norm_vloss, "x-", label="IP (no RMSNorm) Test", color="C3", linewidth=2)
        
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Cross-entropy loss")
        axes[1].set_title("Test (Validation) Loss Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("AERC Comparison: Base vs Intrinsic Plasticity (IP) with & without RMSNorm", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        out_path = os.path.join(os.path.dirname(__file__), "aerc_four_way_comparison.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Comparison plot saved directly to {out_path}")

        # -------------------------------------------------------------
        # Generate Comparison Table Image
        # -------------------------------------------------------------
        fig_tbl, ax_tbl = plt.subplots(figsize=(10, 4))
        ax_tbl.axis('off')
        
        # Calculate final validation losses
        vloss_simp = simp_val_losses[-1][1] if simp_val_losses else 0.0
        vloss_paper = paper_val_losses[-1][1] if paper_val_losses else 0.0
        vloss_ip = ip_val_losses[-1][1] if ip_val_losses else 0.0
        vloss_ip_no_norm = ip_no_norm_val_losses[-1][1] if ip_no_norm_val_losses else 0.0
        
        # Calculate final perplexities
        ppl_simp = math.exp(min(vloss_simp, 20))
        ppl_paper = math.exp(min(vloss_paper, 20))
        ppl_ip = math.exp(min(vloss_ip, 20))
        ppl_ip_no_norm = math.exp(min(vloss_ip_no_norm, 20))
        
        models = [
            "Base (with RMSNorm)",
            "Base (no RMSNorm) [Baseline]",
            "IP (with RMSNorm)",
            "IP (no RMSNorm)"
        ]
        
        losses = [vloss_simp, vloss_paper, vloss_ip, vloss_ip_no_norm]
        ppls = [ppl_simp, ppl_paper, ppl_ip, ppl_ip_no_norm]
        
        table_data = [["Model", "Val Loss", "Val PPL", "Abs Diff (Loss)", "% Diff (Loss)"]]
        
        for name, l, p in zip(models, losses, ppls):
            abs_diff = l - vloss_paper
            pct_diff = (abs_diff / vloss_paper * 100) if vloss_paper != 0 else 0.0
            
            abs_str = f"{abs_diff:+.4f}" if "Baseline" not in name else "-"
            pct_str = f"{pct_diff:+.2f}%" if "Baseline" not in name else "-"
            
            table_data.append([
                name,
                f"{l:.4f}",
                f"{p:.2f}",
                abs_str,
                pct_str
            ])
            
        tbl = ax_tbl.table(
            cellText=table_data, 
            loc='center', 
            cellLoc='center',
            colWidths=[0.35, 0.15, 0.15, 0.18, 0.17]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.8)
        
        # Bold the header cells
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                
        plt.title("AERC Model Performance Comparison Table", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        
        tbl_path = os.path.join(os.path.dirname(__file__), "aerc_comparison_table.png")
        plt.savefig(tbl_path, dpi=150)
        print(f"✓ Comparison table saved directly to {tbl_path}")

    except ImportError:
        print("\n⚠ matplotlib not installed — skipping plot and table generation.")


if __name__ == "__main__":
    main()
