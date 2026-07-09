#!/usr/bin/env python3
"""
Plotting script for AERC Three-Way Comparison results.

Reads JSON results (e.g. results_10k.json)
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
    parser = argparse.ArgumentParser(description="Plot AERC 3-Way Comparison Results")
    parser.add_argument("--json", default="results_10k.json",
                        help="Path to the JSON results file.")
    parser.add_argument("--out", default="../thesis/images/aerc_three_way_comparison.pdf",
                        help="Output image path for the plot.")
    parser.add_argument("--smooth_window", type=int, default=50,
                        help="Window size for smoothing training loss.")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        sys.exit(1)

    with open(args.json, "r") as f:
        data = json.load(f)

    # Filter to three models
    selected_keys = ["aerc_base", "aerc_ip_frozen", "aerc_ip_bptt"]
    models_data = {k: data["models"][k] for k in selected_keys if k in data["models"]}
    
    # Custom labels to match thesis text terminology
    label_map = {
        "aerc_base": "Base AERC",
        "aerc_ip_frozen": "IP-frozen",
        "aerc_ip_bptt": "IP-gradient"
    }
    
    # Setup 2x2 plotting grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    
    color_map = {
        "aerc_base": "C0",
        "aerc_ip_frozen": "C2",
        "aerc_ip_bptt": "C3"
    }

    # 1. Plot Training Loss (Top-Left)
    ax_train = axes[0, 0]
    for key, m_info in models_data.items():
        label = label_map.get(key, key)
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
    ax_train.legend(fontsize=9, loc="upper right")

    # 2. Plot Validation Loss (Top-Right)
    ax_val = axes[0, 1]
    for key, m_info in models_data.items():
        label = label_map.get(key, key)
        val_losses = m_info["val_losses"] # List of [step, loss]
        if val_losses:
            steps, losses = zip(*val_losses)
            ax_val.plot(steps, losses, "o-", label=label, color=color_map.get(key), linewidth=2, markersize=4)
            
    ax_val.set_xlabel("Steps")
    ax_val.set_ylabel("Validation Cross-Entropy Loss")
    ax_val.set_title("Validation Loss Comparison")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=9, loc="upper right")

    # 3. Plot IP Gain parameter evolution (Bottom-Left)
    ax_gain = axes[1, 0]
    # Add horizontal baseline for Base AERC where a = 1
    ax_gain.axhline(1.0, color=color_map["aerc_base"], linestyle=":", linewidth=2, label="Base AERC (a = 1.0)")
    
    for key, m_info in models_data.items():
        if "ip_a_history" in m_info and m_info["ip_a_history"]:
            label = label_map.get(key, key)
            color = color_map.get(key)
            steps, a_means, a_stds = zip(*m_info["ip_a_history"])
            
            steps = np.array(steps)
            a_means = np.array(a_means)
            a_stds = np.array(a_stds)
            
            # Plot mean line
            ax_gain.plot(steps, a_means, "-", label=label, color=color, linewidth=2)
            # Plot standard deviation band
            ax_gain.fill_between(steps, a_means - a_stds, a_means + a_stds, color=color, alpha=0.15)
            
    ax_gain.set_xlabel("Steps")
    ax_gain.set_ylabel("Gain Parameter $a$")
    ax_gain.set_title("IP Gain $a$ Parameter Evolution")
    ax_gain.grid(True, alpha=0.3)
    ax_gain.legend(fontsize=9, loc="best")

    # 4. Plot IP Bias parameter evolution (Bottom-Right)
    ax_bias = axes[1, 1]
    # Add horizontal baseline for Base AERC where b = 0
    ax_bias.axhline(0.0, color=color_map["aerc_base"], linestyle=":", linewidth=2, label="Base AERC (b = 0.0)")
    
    for key, m_info in models_data.items():
        if "ip_b_history" in m_info and m_info["ip_b_history"]:
            label = label_map.get(key, key)
            color = color_map.get(key)
            steps, b_means, b_stds = zip(*m_info["ip_b_history"])
            
            steps = np.array(steps)
            b_means = np.array(b_means)
            b_stds = np.array(b_stds)
            
            # Plot mean line
            ax_bias.plot(steps, b_means, "-", label=label, color=color, linewidth=2)
            # Plot standard deviation band
            ax_bias.fill_between(steps, b_means - b_stds, b_means + b_stds, color=color, alpha=0.15)
            
    ax_bias.set_xlabel("Steps")
    ax_bias.set_ylabel("Bias Parameter $b$")
    ax_bias.set_title("IP Bias $b$ Parameter Evolution")
    ax_bias.grid(True, alpha=0.3)
    ax_bias.legend(fontsize=9, loc="best")

    plt.suptitle("AERC Three-Way Comparison: Impact of Intrinsic Plasticity", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Ensure directory of output path exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(args.out)
    print(f"✓ Matplotlib visualization successfully generated and saved to: {args.out}")

if __name__ == "__main__":
    main()
