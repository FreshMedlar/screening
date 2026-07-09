#!/usr/bin/env python3
"""
Plotting script for AERC Five-Way Comparison results.

Reads JSON results (e.g. results_10k.json or five_way_results.json)
and generates:
1. Training loss comparison (smooth curves).
2. Validation loss comparison (over training steps).
3. IP parameters (ip_a and ip_b) evolution history for the IP models.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def smooth(vals, w=50):
    if len(vals) < w:
        return vals
    return np.convolve(vals, np.ones(w) / w, mode="valid")

def main():
    parser = argparse.ArgumentParser(description="Plot AERC 5-Way Comparison Results")
    parser.add_argument("--json", default="five_way_results.json",
                        help="Path to the JSON results file.")
    parser.add_argument("--out", default="aerc_5way_comparison.png",
                        help="Output image path for the plot.")
    parser.add_argument("--smooth_window", type=int, default=50,
                        help="Window size for smoothing training loss.")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        sys.exit(1)

    with open(args.json, "r") as f:
        data = json.load(f)

    models_data = data["models"]
    
    # Setup plotting grid (2 rows: Row 1 has training/validation loss, Row 2 has IP parameters)
    has_ip_history = any("ip_a_history" in m for m in models_data.values())
    
    if has_ip_history:
        fig = plt.figure(figsize=(12, 10), dpi=150)
        ax_train = fig.add_subplot(2, 2, 1)
        ax_val = fig.add_subplot(2, 2, 2)
        ax_ip = fig.add_subplot(2, 1, 2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
        ax_train, ax_val = axes[0], axes[1]
        
    color_map = {
        "aerc_base": "C0",
        "aerc_rmsnorm": "C1",
        "aerc_ip_frozen": "C2",
        "aerc_ip_bptt": "C3",
        "aerc_ip_bptt_norm": "C4"
    }

    # 1. Plot Training Loss
    for key, m_info in models_data.items():
        label = m_info.get("label", key)
        losses = m_info["train_losses"]
        
        # Dynamically shrink smoothing window if losses list is too small (e.g. in test runs)
        w = min(args.smooth_window, len(losses))
        w = max(1, w)  # Ensure at least 1
        
        smoothed_losses = smooth(losses, w=w)
        steps = np.arange(w - 1, len(losses))
        ax_train.plot(steps, smoothed_losses, label=label, color=color_map.get(key), alpha=0.85)
    
    ax_train.set_xlabel("Steps")
    ax_train.set_ylabel("Cross-Entropy Loss")
    ax_train.set_title("Training Loss Comparison")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(fontsize=8, loc="upper right")

    # 2. Plot Validation Loss
    for key, m_info in models_data.items():
        label = m_info.get("label", key)
        val_losses = m_info["val_losses"] # List of [step, loss]
        if val_losses:
            steps, losses = zip(*val_losses)
            ax_val.plot(steps, losses, "o-", label=label, color=color_map.get(key), linewidth=2, markersize=4)
            
    ax_val.set_xlabel("Steps")
    ax_val.set_ylabel("Validation Cross-Entropy Loss")
    ax_val.set_title("Validation Loss Comparison")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=8, loc="upper right")

    # 3. Plot IP Parameter Evolution (if available)
    if has_ip_history:
        for key, m_info in models_data.items():
            if "ip_a_history" in m_info and m_info["ip_a_history"]:
                label = m_info.get("label", key)
                color = color_map.get(key)
                
                # ip_a_history = [step, mean, std]
                steps, a_means, a_stds = zip(*m_info["ip_a_history"])
                steps, b_means, b_stds = zip(*m_info["ip_b_history"])
                
                # Plot ip_a (Gain) as solid line
                ax_ip.plot(steps, a_means, "-", label=f"{label} (Gain $a$)", color=color, linewidth=2)
                # Plot ip_b (Bias) as dashed line
                ax_ip.plot(steps, b_means, "--", label=f"{label} (Bias $b$)", color=color, linewidth=1.5)
                
        ax_ip.set_xlabel("Steps")
        ax_ip.set_ylabel("Parameter Value (Mean)")
        ax_ip.set_title("IP Parameter Evolution (Gain $a$ vs Bias $b$)")
        ax_ip.grid(True, alpha=0.3)
        ax_ip.legend(fontsize=8, loc="best")

    plt.suptitle("AERC Five-Way Comparison: Impact of Intrinsic Plasticity & RMSNorm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"✓ Matplotlib visualization successfully generated and saved to: {args.out}")

if __name__ == "__main__":
    main()
