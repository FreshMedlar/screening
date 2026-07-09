#!/usr/bin/env python3
"""
Five-way comparison script for AERC model variants.

Models compared:
  1. aerc_base          — AERC (no RMSNorm, no IP)
  2. aerc_rmsnorm       — AERC + RMSNorm (no IP)
  3. aerc_ip_frozen     — AERC + IP pretrain, ip_a/ip_b FROZEN during BPTT
  4. aerc_ip_bptt       — AERC + IP pretrain, ip_a/ip_b GRADIENT-TRAINED (no RMSNorm)
  5. aerc_ip_bptt_norm  — AERC + IP pretrain, ip_a/ip_b GRADIENT-TRAINED + RMSNorm

All models share identical reservoir / attention / readout initialization.
IP models share identical IP pretrain data slices.

Results are saved to a JSON file with the following schema:

  {
    "args":       { ... },          # all hyperparameters
    "vocab_size": int,
    "models": {
      "<key>": {
        "label":         str,
        "n_params":      int,
        "train_losses":  [float, ...],  # one value per training step
        "val_losses":    [[step, val_loss], ...],
        "ip_a_history":  [[step, mean, std], ...],  # IP models only
        "ip_b_history":  [[step, mean, std], ...]   # IP models only
      },
      ...
    }
  }
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

# ── ROCm / AMD GPU environment ─────────────────────────────────────────────
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

from aerc import AERC as AERC_Base
from aerc_ip import AERC as AERC_IP, pretrain_reservoir_ip

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

    train_losses, val_losses, ip_a_history, ip_b_history = [], [], [], []
    is_ip = hasattr(model, "ip_a")
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
                if is_ip:
                    with torch.no_grad():
                        ma = model.ip_a.mean().item(); sa = model.ip_a.std().item()
                        mb = model.ip_b.mean().item(); sb = model.ip_b.std().item()
                    ip_a_history.append([step, ma, sa])
                    ip_b_history.append([step, mb, sb])
                    print(f"  step {step:5d}/{args.max_steps} | train {loss.item():.4f} | "
                          f"val {val_loss:.4f} | ppl {ppl:.2f} | "
                          f"ip_a {ma:.4f}±{sa:.4f} | ip_b {mb:.4f}±{sb:.4f} | {elapsed:.1f}s")
                else:
                    print(f"  step {step:5d}/{args.max_steps} | train {loss.item():.4f} | "
                          f"val {val_loss:.4f} | ppl {ppl:.2f} | {elapsed:.1f}s")

    result = {"label": label, "n_params": n_params,
               "train_losses": train_losses, "val_losses": val_losses}
    if is_ip:
        result["ip_a_history"] = ip_a_history
        result["ip_b_history"] = ip_b_history
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AERC Five-Way Comparison")
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
    parser.add_argument("--spectral_radius", type=float, default=0.95)
    parser.add_argument("--d_e", type=int, default=16)
    parser.add_argument("--N",   type=int, default=160)
    parser.add_argument("--H",   type=int, default=30)
    # IP
    parser.add_argument("--ip_epochs", type=int,   default=10)
    parser.add_argument("--ip_lr",     type=float, default=1e-5)
    parser.add_argument("--ip_mu",     type=float, default=-0.1)
    parser.add_argument("--ip_sigma",  type=float, default=0.3)
    parser.add_argument("--ip_chars",  type=int,   default=10000)
    # output
    parser.add_argument("--out", default=None,
                        help="JSON output path (default: five_way_results.json next to this script)")
    # device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # quick test
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        print(">>> Fast verification mode – overriding config.")
        args.max_steps=10; args.batch_size=8; args.seq_len=32
        args.log_interval=5; args.ip_epochs=1; args.ip_chars=500
        args.warmup_steps=2; args.cooldown_steps=3

    if args.out is None:
        args.out = str(Path(__file__).parent / "five_way_results.json")

    device   = args.device
    use_bf16 = args.bf16 and device.startswith("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} …")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len, args.val_frac)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── helpers to build models ───────────────────────────────────────────────
    def make_base(rmsnorm=False):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        return AERC_Base(vocab_size=vocab_size, d_e=args.d_e, N=args.N, H=args.H,
                         spectral_radius=args.spectral_radius, use_rmsnorm=rmsnorm).to(device)

    def make_ip(rmsnorm=False):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        return AERC_IP(vocab_size=vocab_size, d_e=args.d_e, N=args.N, H=args.H,
                       spectral_radius=args.spectral_radius, use_rmsnorm=rmsnorm).to(device)

    def copy_shared(src, dst):
        """Copy reservoir + attention + readout weights; skip RMSNorm (model-specific)."""
        with torch.no_grad():
            for attr in ("emb.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0",
                         "net_gate.weight", "net_gate.bias",
                         "net_out.weight",  "net_out.bias",
                         "readout.weight",  "readout.bias"):
                parts = attr.split(".")
                s = src; d = dst
                for p in parts[:-1]: s = getattr(s, p); d = getattr(d, p)
                getattr(d, parts[-1]).copy_(getattr(s, parts[-1]))

    # ── instantiate models ────────────────────────────────────────────────────
    m1_base      = make_base(rmsnorm=False)   # 1. base
    m2_norm      = make_base(rmsnorm=True);   copy_shared(m1_base, m2_norm)   # 2. +RMSNorm
    m3_frozen    = make_ip(rmsnorm=False);    copy_shared(m1_base, m3_frozen) # 3. IP frozen
    m4_bptt      = make_ip(rmsnorm=False);    copy_shared(m1_base, m4_bptt)   # 4. IP+BPTT
    m5_bptt_norm = make_ip(rmsnorm=True);     copy_shared(m1_base, m5_bptt_norm)  # 5. IP+BPTT+Norm

    print("\n" + "=" * 70)
    print("Five-Way Comparison Setup")
    print(f"  Config: d_e={args.d_e}, N={args.N}, H={args.H}, SR={args.spectral_radius}")
    for tag, m in [
        ("1. Base (no RMSNorm)",              m1_base),
        ("2. Base + RMSNorm",                 m2_norm),
        ("3. IP pretrain, frozen",            m3_frozen),
        ("4. IP pretrain + gradient IP",      m4_bptt),
        ("5. IP pretrain + gradient IP+Norm", m5_bptt_norm),
    ]:
        print(f"  {tag:<42} {m.count_parameters():>8,} trainable params")
    print("=" * 70)

    # ── IP pre-training ───────────────────────────────────────────────────────
    ip_kwargs = dict(dataset=train_ds, ip_chars=args.ip_chars, batch_size=args.batch_size,
                     eta=args.ip_lr, mu=args.ip_mu, sigma=args.ip_sigma,
                     nepochs=args.ip_epochs, device=device)

    for tag, m in [("Model 3 (frozen)", m3_frozen),
                    ("Model 4 (BPTT, no norm)", m4_bptt),
                    ("Model 5 (BPTT + norm)",   m5_bptt_norm)]:
        with torch.no_grad():          # reset to defaults before each independent run
            m.ip_a.fill_(1.0); m.ip_b.fill_(0.0)
        print(f"\nIP pre-training: {tag}\n" + "-" * 70)
        pretrain_reservoir_ip(model=m, **ip_kwargs)

    # ── freeze IP params for model 3 only ────────────────────────────────────
    print("\nFreezing ip_a / ip_b for Model 3 …")
    m3_frozen.ip_a.requires_grad = False
    m3_frozen.ip_b.requires_grad = False
    print(f"  Model 3 trainable params after freeze: {m3_frozen.count_parameters():,}")

    # ── train ─────────────────────────────────────────────────────────────────
    models_to_train = [
        ("aerc_base",         m1_base,      "AERC base (no RMSNorm)"),
        ("aerc_rmsnorm",      m2_norm,      "AERC + RMSNorm"),
        ("aerc_ip_frozen",    m3_frozen,    "AERC + IP pretrain (frozen during BPTT)"),
        ("aerc_ip_bptt",      m4_bptt,      "AERC + IP pretrain + gradient IP (no RMSNorm)"),
        ("aerc_ip_bptt_norm", m5_bptt_norm, "AERC + IP pretrain + gradient IP + RMSNorm"),
    ]

    results = {}
    for key, model, label in models_to_train:
        results[key] = train_model(model, label, train_loader, val_loader, args, device, use_bf16)

    # ── generated text ────────────────────────────────────────────────────────
    seed_text = "ROMEO:\n"
    print("\n" + "=" * 70 + "\nGenerated text (seed: 'ROMEO:\\n'):")
    for key, model, label in models_to_train:
        print(f"\n  [{label}]")
        print(seed_text + generate(model, chars, seed_text, args.seq_len, device=device))

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "args": {k: getattr(args, k) for k in (
            "data","seq_len","val_frac","batch_size","max_steps","lr",
            "weight_decay","grad_clip","warmup_steps","cooldown_steps",
            "min_lr_ratio","log_interval","seed","spectral_radius",
            "d_e","N","H","ip_epochs","ip_lr","ip_mu","ip_sigma","ip_chars",
        )},
        "args_extra": {"use_bf16": use_bf16, "device": device},
        "vocab_size":  vocab_size,
        "models":      results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Results saved  →  {out_path}\n")
    print(f"{'Model':<48} {'Params':>8}  {'Val loss':>9}  {'PPL':>7}")
    print("-" * 78)
    for key, _, label in models_to_train:
        r  = results[key]
        vl = r["val_losses"][-1][1] if r["val_losses"] else float("nan")
        ppl = math.exp(min(vl, 20))
        print(f"  {label:<46} {r['n_params']:>8,}  {vl:>9.4f}  {ppl:>7.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
