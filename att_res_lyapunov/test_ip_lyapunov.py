#!/usr/bin/env python3
"""
Empirical test: Does combining subcritical Lyapunov initialization (SR=0.85)
with Intrinsic Plasticity (IP pretrain + BPTT) improve upon IP BPTT only (SR=0.95)?
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

# Add screening and att_res paths
sys.path.insert(0, "/home/medlar/Projects/screening/att_res/base")
from aerc_ip import AERC as AERC_IP, pretrain_reservoir_ip
from train_five_way import CharDataset, load_data, estimate_loss, build_lr_lambda

def train_one(sr, label, args, train_ds, val_ds, chars, device):
    torch.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = AERC_IP(
        vocab_size=len(chars),
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=sr,
        use_rmsnorm=False,
    ).to(device)

    # IP pretraining
    ip_kwargs = dict(
        dataset=train_ds,
        ip_chars=args.ip_chars,
        batch_size=args.batch_size,
        eta=args.ip_lr,
        mu=args.ip_mu,
        sigma=args.ip_sigma,
        nepochs=args.ip_epochs,
        device=device,
    )
    pretrain_reservoir_ip(model=model, **ip_kwargs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    warmup   = min(args.warmup_steps,   args.max_steps // 10)
    cooldown = min(args.cooldown_steps, args.max_steps // 3)
    stable   = max(args.max_steps - warmup - cooldown, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(warmup, stable, args.max_steps, args.min_lr_ratio)
    )
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    print(f"\nTraining {label} (SR={sr}, params={sum(p.numel() for p in trainable):,})")
    step, t0 = 0, time.time()
    val_losses = []

    model.train()
    while step < args.max_steps:
        for x, y in train_loader:
            if step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            with ctx:
                logits = model(idx=x)
                loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_interval == 0:
                elapsed  = time.time() - t0
                val_loss = estimate_loss(model, val_loader, device, use_bf16=True)
                val_losses.append([step, val_loss])
                ma = model.ip_a.mean().item()
                sa = model.ip_a.std().item()
                print(f"  step {step:5d}/{args.max_steps} | val {val_loss:.4f} | ppl {math.exp(val_loss):.2f} | ip_a {ma:.4f}±{sa:.4f} | {elapsed:.1f}s")

    return val_losses[-1][1], val_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--cooldown_steps", type=int, default=300)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_e", type=int, default=16)
    parser.add_argument("--N", type=int, default=160)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--ip_epochs", type=int, default=10)
    parser.add_argument("--ip_lr", type=float, default=1e-5)
    parser.add_argument("--ip_mu", type=float, default=-0.1)
    parser.add_argument("--ip_sigma", type=float, default=0.3)
    parser.add_argument("--ip_chars", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_ds, val_ds, chars, _ = load_data(args.data, args.seq_len)

    # Run Lyapunov (SR=0.85) + IP BPTT
    final_val, history = train_one(0.85, "AERC Lyapunov + IP BPTT (SR=0.85)", args, train_ds, val_ds, chars, args.device)
    print(f"\nFinal Val Loss (Lyapunov SR=0.85 + IP BPTT): {final_val:.4f} (PPL {math.exp(final_val):.2f})")

if __name__ == "__main__":
    main()
