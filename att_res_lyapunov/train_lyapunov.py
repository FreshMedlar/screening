#!/usr/bin/env python3
"""
Training script for Lyapunov-Enhanced AERC model.
Trains aerc_lyapunov under the exact same pipeline and hyperparameters
as train_five_way.py from att_res/base/.

Output JSON format matches att_res/base/results_10k.json.
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ROCm / AMD GPU environment
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

# Add paths for imports
base_dir = Path(__file__).resolve().parent.parent / "att_res" / "base"
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aerc import AERC as AERC_Base
from aerc_lyapunov import AERC_Lyapunov

import warnings, logging
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, text, chars, seq_len):
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


def load_data(path, seq_len, val_frac=0.1):
    text  = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    split = int(len(text) * (1 - val_frac))
    return CharDataset(text[:split], chars, seq_len), \
           CharDataset(text[split:], chars, seq_len), \
           chars, len(chars)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50, use_bf16=False):
    model.eval()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            logits = model(idx=x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        count += 1
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def generate(model, chars, seed_text, seq_len, max_new=200, temperature=0.8, device="cuda"):
    stoi   = {c: i for i, c in enumerate(chars)}
    itos   = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in seed_text if c in stoi]
    pad    = stoi.get(" ", next(iter(stoi.values())))
    if len(tokens) < seq_len:
        tokens = [pad] * (seq_len - len(tokens)) + tokens
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new):
        logits = model(idx=x[:, -seq_len:])
        probs  = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return "".join(itos.get(t.item(), "?") for t in x[0, seq_len:])


def build_lr_lambda(warmup, stable, total, min_lr_ratio):
    cooldown = total - warmup - stable
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        elif step < warmup + stable:
            p = (step - warmup) / stable
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * p))
        else:
            return min_lr_ratio * (total - step) / max(cooldown, 1)
    return lr_lambda


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, label, train_loader, val_loader, args, device, use_bf16):
    print(f"\n{'─'*70}\nTraining: {label}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_params:,}\n{'─'*70}")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    warmup   = min(args.warmup_steps,   args.max_steps // 10)
    cooldown = min(args.cooldown_steps, args.max_steps // 3)
    stable   = max(args.max_steps - warmup - cooldown, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(warmup, stable, args.max_steps, args.min_lr_ratio)
    )
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    train_losses, val_losses = [], []
    step, t0 = 0, time.time()

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
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                elapsed  = time.time() - t0
                val_loss = estimate_loss(model, val_loader, device, use_bf16=use_bf16)
                ppl      = math.exp(min(val_loss, 20))
                val_losses.append([step, val_loss])
                print(f"  step {step:5d}/{args.max_steps} | train {loss.item():.4f} | "
                      f"val {val_loss:.4f} | ppl {ppl:.2f} | {elapsed:.1f}s")

    result = {
        "label": label,
        "n_params": n_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Lyapunov-Enhanced AERC Training")
    # data
    parser.add_argument("--data",    default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len", type=int,   default=64)
    parser.add_argument("--val_frac",type=float, default=0.1)
    # training
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--max_steps",      type=int,   default=10000)
    parser.add_argument("--lr",             type=float, default=5e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-3)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--warmup_steps",   type=int,   default=150)
    parser.add_argument("--cooldown_steps", type=int,   default=300)
    parser.add_argument("--min_lr_ratio",   type=float, default=0.1)
    parser.add_argument("--log_interval",   type=int,   default=400)
    parser.add_argument("--seed",           type=int,   default=42)
    # bf16
    parser.add_argument("--bf16",    action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    # architecture
    parser.add_argument("--spectral_radius", type=float, default=0.85)
    parser.add_argument("--d_e", type=int, default=16)
    parser.add_argument("--N",   type=int, default=160)
    parser.add_argument("--H",   type=int, default=30)
    # output
    parser.add_argument("--out", default=None,
                        help="JSON output path (default: results_10k.json in att_res_lyapunov)")
    # device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # quick test
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        print(">>> Fast verification mode – overriding config.")
        args.max_steps=10; args.batch_size=8; args.seq_len=32
        args.log_interval=5; args.warmup_steps=2; args.cooldown_steps=3

    if args.out is None:
        args.out = str(Path(__file__).parent / "results_10k.json")

    device   = args.device
    use_bf16 = args.bf16 and device.startswith("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Loading data from {args.data} …")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len, args.val_frac)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Instantiate baseline to ensure identical initialization of attention & readout weights
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    m_base = AERC_Base(vocab_size=vocab_size, d_e=args.d_e, N=args.N, H=args.H,
                       spectral_radius=0.95, use_rmsnorm=False).to(device)

    # Instantiate Lyapunov-Enhanced AERC (SR=0.85)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    m_lyap = AERC_Lyapunov(vocab_size=vocab_size, d_e=args.d_e, N=args.N, H=args.H,
                           spectral_radius=args.spectral_radius).to(device)

    # Copy identical initial attention & readout weights from base model (matching train_five_way.py)
    with torch.no_grad():
        m_lyap.emb.weight.copy_(m_base.emb.weight)
        m_lyap.rnn.weight_ih_l0.copy_(m_base.rnn.weight_ih_l0)
        m_lyap.net_gate.weight.copy_(m_base.net_gate.weight)
        m_lyap.net_gate.bias.copy_(m_base.net_gate.bias)
        m_lyap.net_out.weight.copy_(m_base.net_out.weight)
        m_lyap.net_out.bias.copy_(m_base.net_out.bias)
        m_lyap.readout.weight.copy_(m_base.readout.weight)
        m_lyap.readout.bias.copy_(m_base.readout.bias)

    print("\n" + "=" * 70)
    print("AERC Lyapunov Training Setup")
    print(f"  Config: d_e={args.d_e}, N={args.N}, H={args.H}, SR={args.spectral_radius}")
    print(f"  Trainable params: {m_lyap.count_parameters():,}")
    print("=" * 70)

    # Train only the new aerc improved with lyapunov
    label = "AERC + Lyapunov (Entropy-Minimizing Calibration, SR=0.85, no RMSNorm, no IP)"
    model_res = train_model(m_lyap, label, train_loader, val_loader, args, device, use_bf16)

    # Generated text sample
    seed_text = "ROMEO:\n"
    sample = generate(m_lyap, chars, seed_text, args.seq_len, max_new=200, device=device)
    print(f"\nSample generation:\n{seed_text}{sample}\n")

    # Build results dictionary matching train_five_way.py / results_10k.json schema
    out_dict = {
        "args": {
            "data": args.data,
            "seq_len": args.seq_len,
            "val_frac": args.val_frac,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "cooldown_steps": args.cooldown_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "log_interval": args.log_interval,
            "seed": args.seed,
            "spectral_radius": args.spectral_radius,
            "d_e": args.d_e,
            "N": args.N,
            "H": args.H,
        },
        "args_extra": {
            "use_bf16": use_bf16,
            "device": device,
        },
        "vocab_size": vocab_size,
        "models": {
            "aerc_lyapunov": model_res,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n[Saved Results JSON] -> {args.out}")


if __name__ == "__main__":
    main()
