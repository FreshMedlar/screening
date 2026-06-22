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

from aerc_simplified import AERC as AERC_Base
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
    grid_ip_sigma     = args.ip_sigmas
    grid_leaking_rate = args.leaking_rates
    grid_lr           = args.lrs if args.grid_lr else [args.lr]

    # Base model only depends on (lr, leaking_rate) — NOT on ip_sigma.
    # Train it once per (lr, leaking_rate) pair, then reuse for all ip_sigma values.
    outer_combos = list(itertools.product(grid_lr, grid_leaking_rate))
    total_base   = len(outer_combos)
    total_ip     = len(outer_combos) * len(grid_ip_sigma)

    print("=" * 70)
    print(f"Grid search: {total_base} base runs + {total_ip} IP runs = {total_base + total_ip} total")
    print(f"  lr values       : {grid_lr}")
    print(f"  leaking_rate    : {grid_leaking_rate}")
    print(f"  ip_sigma values : {grid_ip_sigma}")
    print(f"  max_steps       : {args.max_steps}")
    print("=" * 70 + "\n")

    results = []
    base_cache: dict = {}  # (lr, leaking_rate) -> val_base

    outer_idx = 0
    for lr, alpha in outer_combos:
        outer_idx += 1
        print(f"{'═'*60}")
        print(f"[outer {outer_idx}/{total_base}] lr={lr:.0e}  leak={alpha}")

        # ── Base model — trained ONCE per (lr, leaking_rate) ─────────────
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model_base = AERC_Base(
            vocab_size=vocab_size,
            d_e=args.d_e, N=args.N, H=args.H,
            spectral_radius=args.spectral_radius,
            leaking_rate=alpha,
        ).to(device)

        t0 = time.time()
        val_base = run_one(
            model_base, train_ds, val_ds,
            lr=lr, max_steps=args.max_steps,
            batch_size=args.batch_size, seq_len=args.seq_len,
            weight_decay=args.weight_decay,
            warmup=args.warmup_steps, cooldown=args.cooldown_steps,
            grad_clip=args.grad_clip, device=device, use_bf16=use_bf16,
        )
        del model_base  # free VRAM before IP runs
        t_base = time.time() - t0
        base_cache[(lr, alpha)] = val_base
        print(f"  Base  : val_loss={val_base:.4f}  ({t_base:.0f}s)")

        # ── IP models — one per ip_sigma ─────────────────────────────────
        for sigma in grid_ip_sigma:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model_ip = AERC_IP(
                vocab_size=vocab_size,
                d_e=args.d_e, N=args.N, H=args.H,
                spectral_radius=args.spectral_radius,
                leaking_rate=alpha,
            ).to(device)

            pretrain_reservoir_ip(
                model=model_ip,
                dataset=train_ds,
                ip_chars=args.ip_chars,
                batch_size=args.batch_size,
                eta=args.ip_lr,
                mu=args.ip_mu,
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
            diff = val_base - val_ip  # positive = IP is better
            winner = "IP ✓" if diff > 0 else "Base"
            print(f"  sigma={sigma} | IP val={val_ip:.4f}  ({t_ip:.0f}s)  diff={diff:+.4f}  {winner}")

            results.append({
                "lr": lr, "ip_sigma": sigma, "leaking_rate": alpha,
                "val_base": val_base, "val_ip": val_ip, "diff": diff,
            })

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'lr':>8}  {'sigma':>7}  {'leak':>6}  {'base':>8}  {'ip':>8}  {'Δ(base-ip)':>12}  winner"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["val_ip"]):
        winner = "IP ✓" if r["diff"] > 0 else "Base"
        print(f"  {r['lr']:6.0e}  {r['ip_sigma']:7.2f}  {r['leaking_rate']:6.2f}"
              f"  {r['val_base']:8.4f}  {r['val_ip']:8.4f}  {r['diff']:+12.4f}  {winner}")

    best_ip   = min(results, key=lambda x: x["val_ip"])
    best_base = min(results, key=lambda x: x["val_base"])
    print(f"\n  Best IP   config : lr={best_ip['lr']:.0e}  sigma={best_ip['ip_sigma']}  leak={best_ip['leaking_rate']}  → val={best_ip['val_ip']:.4f}")
    print(f"  Best Base config : lr={best_base['lr']:.0e}  sigma=—  leak={best_base['leaking_rate']}  → val={best_base['val_base']:.4f}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_json}")

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        for lr_val in grid_lr:
            subset = [r for r in results if r["lr"] == lr_val]
            sigmas  = sorted(set(r["ip_sigma"]     for r in subset))
            alphas  = sorted(set(r["leaking_rate"] for r in subset))

            # Build 2D arrays
            base_grid = np.zeros((len(alphas), len(sigmas)))
            ip_grid   = np.zeros((len(alphas), len(sigmas)))
            diff_grid = np.zeros((len(alphas), len(sigmas)))
            for r in subset:
                ai = alphas.index(r["leaking_rate"])
                si = sigmas.index(r["ip_sigma"])
                base_grid[ai, si] = r["val_base"]
                ip_grid[ai, si]   = r["val_ip"]
                diff_grid[ai, si] = r["diff"]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(
                f"Grid Search — lr={lr_val:.0e}, max_steps={args.max_steps}\n"
                f"Val loss ↓ is better  |  Δ = base − ip (green = IP wins)",
                fontsize=12, fontweight="bold",
            )

            def _heatmap(ax, data, title, cmap, fmt=".4f"):
                im = ax.imshow(data, cmap=cmap, aspect="auto")
                ax.set_xticks(range(len(sigmas)));  ax.set_xticklabels([f"{s}" for s in sigmas])
                ax.set_yticks(range(len(alphas)));  ax.set_yticklabels([f"{a}" for a in alphas])
                ax.set_xlabel("ip_sigma");  ax.set_ylabel("leaking_rate")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                for (row, col), val in np.ndenumerate(data):
                    ax.text(col, row, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=8, color="white" if abs(val) > 0.5 * data.max() else "black")

            _heatmap(axes[0], base_grid, "Base model val loss", "Blues_r")
            _heatmap(axes[1], ip_grid,   "IP model val loss",   "Blues_r")
            # Diverging colormap: green = IP better, red = base better
            vmax = np.abs(diff_grid).max() + 1e-6
            _heatmap(axes[2], diff_grid, "Δ val loss (base − ip)\nGreen = IP wins",
                     mcolors.LinearSegmentedColormap.from_list("rg", ["#d73027", "#ffffbf", "#1a9850"]),
                     fmt="+.4f")
            axes[2].images[0].set_clim(-vmax, vmax)

            plt.tight_layout()
            suffix = f"_lr{lr_val:.0e}".replace("-", "m")
            out_plot = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"grid_heatmap{suffix}.png")
            plt.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Heatmap saved to {out_plot}")

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
    parser.add_argument("--lr",         type=float, default=1e-3,
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
    parser.add_argument("--d_e",            type=int,   default=64)
    parser.add_argument("--N",              type=int,   default=150)
    parser.add_argument("--H",              type=int,   default=31)
    parser.add_argument("--spectral_radius",type=float, default=0.99)

    # IP fixed hyperparameters
    parser.add_argument("--ip_epochs",  type=int,   default=8)
    parser.add_argument("--ip_chars",   type=int,   default=10000)
    parser.add_argument("--ip_lr",      type=float, default=1e-5)
    parser.add_argument("--ip_mu",      type=float, default=0.0)

    # ── Grid axes ────────────────────────────────────────────────────────
    parser.add_argument("--ip_sigmas", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7],
                        help="ip_sigma values to sweep")
    parser.add_argument("--leaking_rates", type=float, nargs="+",
                        default=[0.5, 0.7, 0.9],
                        help="leaking_rate values to sweep")
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[3e-4, 1e-3, 3e-3],
                        help="LR values to sweep (only active with --grid_lr)")
    parser.add_argument("--grid_lr", action="store_true",
                        help="Include lr as a third grid axis")

    args = parser.parse_args()
    grid_search(args)


if __name__ == "__main__":
    main()
