#!/usr/bin/env python3
"""
Grid search over ip_sigma × leaking_rate [× lr] for both AERC Base and AERC IP.

Each combination trains two models (identical seed, identical config) for
MAX_STEPS steps and records the final validation loss for each.

Results are printed as a table and saved as heatmap plots and a JSON file.

Usage:
    # 2D grid (ip_sigma × leaking_rate), fixed lr:
    python grid_search.py --max_steps 1500 --lr 1e-3

    # 3D grid (adds lr as a third axis):
    python grid_search.py --max_steps 1500 --grid_lr
"""

import os
import sys
import json
import time
import math
import itertools
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# ROCm environment variables for AMD GPU
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
if "ROCM_PATH" not in os.environ:
    os.environ["ROCM_PATH"] = "/opt/rocm"

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from aerc_ip import AERC as AERC_IP, pretrain_reservoir_ip
from train import CharDataset, load_data, estimate_loss


# ---------------------------------------------------------------------------
# LR Schedule (cosine warmup + linear cooldown, same as train.py)
# ---------------------------------------------------------------------------
def get_lr(step: int, max_steps: int, lr: float, warmup: int, cooldown: int, min_ratio: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    cooldown_start = max_steps - cooldown
    if step >= cooldown_start:
        frac = (step - cooldown_start) / max(cooldown, 1)
        return lr * max(min_ratio, 1.0 - frac * (1.0 - min_ratio))
    progress = (step - warmup) / max(max_steps - warmup - cooldown, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_ratio + (1.0 - min_ratio) * cosine)


# ---------------------------------------------------------------------------
# Single training run — returns final validation loss
# ---------------------------------------------------------------------------
def run_one(
    model,
    train_ds,
    val_ds,
    lr: float,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    weight_decay: float,
    warmup: int,
    cooldown: int,
    grad_clip: float,
    device: str,
    use_bf16: bool,
) -> float:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    model.train()
    step = 0
    data_iter = iter(train_loader)

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        current_lr = get_lr(step, max_steps, lr, warmup, cooldown, 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(idx=x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        step += 1

    return estimate_loss(model, val_loader, device, max_batches=80, use_bf16=use_bf16)


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
def grid_search(args):
    device = args.device
    use_bf16 = args.bf16 and device.startswith("cuda")

    print(f"Loading data from {args.data}...")
    train_ds, val_ds, chars, vocab_size = load_data(args.data, args.seq_len)
    print(f"  vocab_size={vocab_size}  train={len(train_ds):,}  val={len(val_ds):,}")
    print(f"  Device: {device}  bf16: {use_bf16}\n")

    # ── Grid definition ────────────────────────────────────────────────────
    grid_ip_sigma        = args.ip_sigmas if args.ip_sigmas is not None else [args.ip_sigma]
    grid_ip_mu           = args.ip_mus if args.ip_mus is not None else [args.ip_mu]
    grid_spectral_radius = args.spectral_radii if args.spectral_radii is not None else [args.spectral_radius]
    grid_lr              = args.lrs if args.grid_lr else [args.lr]

    total_ip     = len(grid_lr) * len(grid_ip_sigma) * len(grid_ip_mu) * len(grid_spectral_radius)

    print("=" * 70)
    print(f"Grid search: {total_ip} IP runs")
    print(f"  lr values              : {grid_lr}")
    print(f"  ip_sigma values        : {grid_ip_sigma}")
    print(f"  ip_mu values           : {grid_ip_mu}")
    print(f"  spectral_radius values : {grid_spectral_radius}")
    print(f"  max_steps              : {args.max_steps}")
    print("=" * 70 + "\n")

    results = []

    outer_idx = 0
    for lr in grid_lr:
        outer_idx += 1
        print(f"{'═'*60}")
        print(f"[outer {outer_idx}/{len(grid_lr)}] lr={lr:.0e}")

        # ── IP models — sweep sigmas, mus, and spectral radii ────────────
        for sigma in grid_ip_sigma:
            for mu in grid_ip_mu:
                for sr in grid_spectral_radius:
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    model_ip = AERC_IP(
                        vocab_size=vocab_size,
                        d_e=args.d_e, N=args.N, H=args.H,
                        spectral_radius=sr,
                    ).to(device)

                    pretrain_reservoir_ip(
                        model=model_ip,
                        dataset=train_ds,
                        ip_chars=args.ip_chars,
                        batch_size=args.batch_size,
                        eta=args.ip_lr,
                        mu=mu,
                        sigma=sigma,
                        nepochs=args.ip_epochs,
                        device=device,
                    )

                    t0 = time.time()
                    val_ip = run_one(
                        model_ip, train_ds, val_ds,
                        lr=lr, max_steps=args.max_steps,
                        batch_size=args.batch_size, seq_len=args.seq_len,
                        weight_decay=args.weight_decay,
                        warmup=args.warmup_steps, cooldown=args.cooldown_steps,
                        grad_clip=args.grad_clip, device=device, use_bf16=use_bf16,
                    )
                    del model_ip
                    t_ip = time.time() - t0
                    print(f"  sigma={sigma} | mu={mu} | sr={sr} | IP val={val_ip:.4f}  ({t_ip:.0f}s)")

                    results.append({
                        "lr": lr,
                        "ip_sigma": sigma,
                        "ip_mu": mu,
                        "spectral_radius": sr,
                        "val_ip": val_ip,
                    })

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'lr':>8}  {'sigma':>7}  {'mu':>6}  {'sr':>5}  {'ip':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["val_ip"]):
        print(f"  {r['lr']:6.0e}  {r['ip_sigma']:7.2f}  {r['ip_mu']:6.2f}  {r['spectral_radius']:5.2f}  {r['val_ip']:8.4f}")

    best_ip   = min(results, key=lambda x: x["val_ip"])
    print(f"\n  Best IP   config : lr={best_ip['lr']:.0e}  sigma={best_ip['ip_sigma']}  mu={best_ip['ip_mu']}  sr={best_ip['spectral_radius']}  → val={best_ip['val_ip']:.4f}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_json}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Determine active sweep dimensions
        unique_sigmas = sorted(set(r["ip_sigma"] for r in results))
        unique_mus    = sorted(set(r["ip_mu"]    for r in results))
        unique_lrs    = sorted(set(r["lr"]       for r in results))
        unique_srs    = sorted(set(r["spectral_radius"] for r in results))

        active_dims = []
        if len(unique_sigmas) > 1: active_dims.append(("ip_sigma", unique_sigmas))
        if len(unique_mus) > 1:    active_dims.append(("ip_mu", unique_mus))
        if len(unique_lrs) > 1:    active_dims.append(("lr", unique_lrs))
        if len(unique_srs) > 1:    active_dims.append(("spectral_radius", unique_srs))

        if len(active_dims) == 2:
            dim1_name, dim1_vals = active_dims[0]
            dim2_name, dim2_vals = active_dims[1]

            # Build 2D array: dim2 (rows/y-axis) x dim1 (cols/x-axis)
            grid = np.zeros((len(dim2_vals), len(dim1_vals)))
            for r in results:
                i1 = dim1_vals.index(r[dim1_name])
                i2 = dim2_vals.index(r[dim2_name])
                grid[i2, i1] = r["val_ip"]

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.suptitle(
                f"Grid Search (IP model only) — max_steps={args.max_steps}\n"
                f"Val loss ↓ is better",
                fontsize=12, fontweight="bold",
            )

            im = ax.imshow(grid, cmap="Blues_r", aspect="auto")
            ax.set_xticks(range(len(dim1_vals)));  ax.set_xticklabels([f"{v:.2g}" if isinstance(v, float) else f"{v}" for v in dim1_vals])
            ax.set_yticks(range(len(dim2_vals)));  ax.set_yticklabels([f"{v:.2g}" if isinstance(v, float) else f"{v}" for v in dim2_vals])
            ax.set_xlabel(dim1_name)
            ax.set_ylabel(dim2_name)
            ax.set_title(f"IP model val loss ({dim2_name} vs {dim1_name})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for (row, col), val in np.ndenumerate(grid):
                ax.text(col, row, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 * grid.max() else "black")

            out_plot = args.image_name
            if not out_plot.endswith(".png"):
                out_plot += ".png"
            if not os.path.isabs(out_plot) and os.path.sep not in out_plot:
                out_plot = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_plot)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Heatmap saved to {out_plot}")
        elif len(active_dims) == 1:
            dim_name, dim_vals = active_dims[0]
            # sort by dim_vals
            sorted_results = sorted(results, key=lambda x: x[dim_name])
            xs = [r[dim_name] for r in sorted_results]
            ys = [r["val_ip"] for r in sorted_results]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(xs, ys, "o-", color="C0", linewidth=2)
            ax.set_xlabel(dim_name)
            ax.set_ylabel("val_ip (loss)")
            ax.set_title(f"Sweep over {dim_name}")
            ax.grid(True, alpha=0.3)

            out_plot = args.image_name
            if not out_plot.endswith(".png"):
                out_plot += ".png"
            if not os.path.isabs(out_plot) and os.path.sep not in out_plot:
                out_plot = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_plot)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Sweep plot saved to {out_plot}")
        else:
            print("  (Plotting skipped: Plotting supports exactly 1 or 2 sweep dimensions)")

    except ImportError:
        print("  (matplotlib not available — skipping plots)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AERC Grid Search")
    parser.add_argument("--data", type=str,
                        default="/home/medlar/Projects/screening/att_res/tinyshakespeare.txt")
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_steps",  type=int,   default=1500)
    parser.add_argument("--lr",         type=float, default=5e-3,
                        help="Fixed LR when --grid_lr is not set")
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_steps",  type=int, default=150)
    parser.add_argument("--cooldown_steps", type=int, default=300)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--bf16",       action="store_true", default=True)
    parser.add_argument("--no_bf16",    dest="bf16", action="store_false")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture defaults (keep ~155k params)
    parser.add_argument("--d_e",            type=int,   default=16)
    parser.add_argument("--N",              type=int,   default=160)
    parser.add_argument("--H",              type=int,   default=30)
    parser.add_argument("--spectral_radius",type=float, default=0.95)

    # IP fixed hyperparameters
    parser.add_argument("--ip_epochs",  type=int,   default=5)
    parser.add_argument("--ip_chars",   type=int,   default=10000)
    parser.add_argument("--ip_lr",      type=float, default=1e-5)
    parser.add_argument("--ip_mu",      type=float, default=-0.1)
    parser.add_argument("--ip_sigma",   type=float, default=0.3)

    # ── Grid axes ────────────────────────────────────────────────────────
    parser.add_argument("--image_name", type=str, required=True,
                        help="Name of the output heatmap image file (e.g. grid_heatmap.png)")
    parser.add_argument("--ip_sigmas", type=float, nargs="+",
                        help="ip_sigma values to sweep (overrides --ip_sigma if specified)")
    parser.add_argument("--ip_mus", type=float, nargs="+",
                        help="ip_mu values to sweep (overrides --ip_mu if specified)")
    parser.add_argument("--spectral_radii", type=float, nargs="+",
                        help="spectral_radius values to sweep (overrides --spectral_radius if specified)")
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[3e-4, 1e-3, 3e-3],
                        help="LR values to sweep (only active with --grid_lr)")
    parser.add_argument("--grid_lr", action="store_true",
                        help="Include lr as a grid axis")

    args = parser.parse_args()
    grid_search(args)


if __name__ == "__main__":
    main()
