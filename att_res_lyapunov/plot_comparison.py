#!/usr/bin/env python3
"""
Comparison Plotter: AERC Lyapunov vs AERC IP BPTT (from results_10k.json).
Plots validation loss, training curves, and perplexity.
"""

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.9):
    """Exponential moving average for smoother loss visualization."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", default="att_res/base/results_10k.json")
    parser.add_argument("--lyap-json", default="att_res_lyapunov/results_10k.json")
    parser.add_argument("--out-plot", default="att_res_lyapunov/comparison_lyapunov_vs_ip.png")
    args = parser.parse_args()

    with open(args.base_json) as f:
        data_base = json.load(f)
    with open(args.lyap_json) as f:
        data_lyap = json.load(f)

    # Models to compare
    ip_bptt = data_base["models"]["aerc_ip_bptt"]
    base = data_base["models"].get("aerc_base", None)
    lyap = data_lyap["models"]["aerc_lyapunov"]

    # Extract losses
    ip_val = np.array(ip_bptt["val_losses"])       # [[step, val_loss], ...]
    lyap_val = np.array(lyap["val_losses"])
    base_val = np.array(base["val_losses"]) if base else None

    ip_train = ip_bptt["train_losses"]
    lyap_train = lyap["train_losses"]
    base_train = base["train_losses"] if base else None

    # Print summary table
    print("\n" + "=" * 78)
    print("  EVALUATION SUMMARY: 10,000 STEPS AERC TRAINING COMPARISON")
    print("=" * 78)
    print(f"  {'Model':<44} | {'Val Loss':<9} | {'Val PPL':<8} | {'Delta vs Base':<12}")
    print("  " + "-" * 74)

    base_final_val = base_val[-1, 1] if base_val is not None else float("nan")
    for name, v_arr in [
        ("AERC Base (No RMSNorm, No IP)", base_val),
        ("AERC IP BPTT (from att_res/base/results_10k)", ip_val),
        ("AERC Lyapunov (Ours, No RMSNorm, No IP)", lyap_val),
    ]:
        if v_arr is not None:
            final_loss = v_arr[-1, 1]
            ppl = math.exp(min(final_loss, 20))
            delta = final_loss - base_final_val
            delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "baseline"
            print(f"  {name:<44} | {final_loss:8.4f}  | {ppl:7.2f}  | {delta_str:<12}")
    print("=" * 78 + "\n")

    # Plot
    fig = plt.figure(figsize=(15, 10), dpi=160)
    fig.patch.set_facecolor("#0e1117")

    plt.rcParams["text.color"] = "#e0e6ed"
    plt.rcParams["axes.labelcolor"] = "#c9d1d9"
    plt.rcParams["xtick.color"] = "#8b949e"
    plt.rcParams["ytick.color"] = "#8b949e"

    # Colors
    c_base = "#8b949e"
    c_ip = "#ffa657"
    c_lyap = "#2ea043"

    # 1. Validation Loss
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#161b22")
    if base_val is not None:
        ax1.plot(base_val[:, 0], base_val[:, 1], color=c_base, lw=1.8, marker="o", ms=4,
                 label=f"AERC Base ({base_val[-1, 1]:.4f})", alpha=0.7)
    ax1.plot(ip_val[:, 0], ip_val[:, 1], color=c_ip, lw=2.2, marker="s", ms=4,
             label=f"AERC IP BPTT ({ip_val[-1, 1]:.4f})")
    ax1.plot(lyap_val[:, 0], lyap_val[:, 1], color=c_lyap, lw=2.6, marker="^", ms=5,
             label=f"AERC Lyapunov (Ours) ({lyap_val[-1, 1]:.4f})")

    ax1.set_title("Validation Cross-Entropy Loss (Lower is Better)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Validation Loss")
    ax1.grid(True, color="#30363d", alpha=0.6)
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9.5)

    # 2. Validation Perplexity
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#161b22")
    if base_val is not None:
        ppl_base = [math.exp(v) for v in base_val[:, 1]]
        ax2.plot(base_val[:, 0], ppl_base, color=c_base, lw=1.8, marker="o", ms=4,
                 label=f"AERC Base ({ppl_base[-1]:.2f})", alpha=0.7)
    ppl_ip = [math.exp(v) for v in ip_val[:, 1]]
    ppl_lyap = [math.exp(v) for v in lyap_val[:, 1]]
    ax2.plot(ip_val[:, 0], ppl_ip, color=c_ip, lw=2.2, marker="s", ms=4,
             label=f"AERC IP BPTT ({ppl_ip[-1]:.2f})")
    ax2.plot(lyap_val[:, 0], ppl_lyap, color=c_lyap, lw=2.6, marker="^", ms=5,
             label=f"AERC Lyapunov (Ours) ({ppl_lyap[-1]:.2f})")

    ax2.set_title("Validation Perplexity (Lower is Better)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, color="#30363d", alpha=0.6)
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9.5)

    # 3. Training Loss Curves (Smoothed)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#161b22")
    if base_train is not None:
        ax3.plot(smooth(base_train, 0.95), color=c_base, lw=1.5, alpha=0.5, label="AERC Base")
    ax3.plot(smooth(ip_train, 0.95), color=c_ip, lw=1.8, alpha=0.8, label="AERC IP BPTT")
    ax3.plot(smooth(lyap_train, 0.95), color=c_lyap, lw=2.0, alpha=0.9, label="AERC Lyapunov")
    ax3.set_title("Training Loss Progression (Smoothed EMA)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Cross-Entropy Loss")
    ax3.grid(True, color="#30363d", alpha=0.6)
    ax3.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9.5)

    # 4. Architectural Summary Box
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#161b22")
    ax4.axis("off")

    summary_card = (
        "THEORETICAL MECHANISM COMPARISON:\n\n"
        "1. AERC Base (baseline):\n"
        "   - Random i.i.d. Gaussian/uniform recurrent weights.\n"
        "   - Eigenvalues spread inside disk (|mu| < g) via circular law.\n"
        "   - Suffers gain compression: C_d(0) < 1 pushes lambda_1 << 0.\n\n"
        "2. AERC IP BPTT (att_res baseline):\n"
        "   - Requires 10 epochs of unsupervised IP pre-training.\n"
        "   - Requires BPTT gradient updates on ip_a and ip_b.\n"
        "   - Adds parameter tuning overhead during training.\n\n"
        "3. AERC Lyapunov (Clark 2026, Ours):\n"
        "   - Critical Orthogonal Spectrum: all eigenvalues on |mu| = g.\n"
        "   - Zero damped modes -> eliminates representation dissipation.\n"
        "   - Gain-Aware Homeostatic Normalization: compensates for\n"
        "     C_d(0) = <(1 - tanh^2(x))^2> to maintain lambda_1 ~= 0.\n"
        "   - NO RMSNorm, NO IP pre-training, NO extra parameters (155,645)."
    )

    ax4.text(
        0.05, 0.95, summary_card,
        transform=ax4.transAxes,
        fontsize=10.0,
        verticalalignment="top",
        family="monospace",
        color="#7ee787",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d")
    )

    plt.tight_layout()
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_plot, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Plot Saved] -> {args.out_plot}\n")


if __name__ == "__main__":
    main()
