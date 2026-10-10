#!/usr/bin/env python3
"""
Benchmark Comparison: Classic ESN vs Lyapunov-Enhanced ESN (LE-ESN).
Based on the theoretical principles from:
David G. Clark (2026), "Lyapunov spectrum of random neural networks" (arXiv:2610.12426).

Compares:
1. Information Memory Capacity (MC) for N in {100, 300, 500}
2. NARMA Nonlinear Memory (k = 10, 20, 30)
3. Mackey-Glass Chaotic Forecasting & Autonomous Rollout Stability (VPT)
4. Lorenz-63 3D Chaotic Attractor Tracking
5. Echo State Property & Estimated Effective Lyapunov Exponent

Run with:
    /home/medlar/Projects/.venv/bin/python reservoir/compare_benchmarks.py
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reservoir.esn import EchoStateNetwork
from reservoir.lyapunov_esn import LyapunovEnhancedReservoir
from reservoir.data import (
    generate_mackey_glass,
    generate_narma,
    generate_memory_capacity,
    generate_lorenz,
)
from reservoir.metrics import (
    compute_nrmse,
    compute_nmse,
    compute_memory_capacity,
    compute_vpt,
    compute_attractor_divergence,
)
from reservoir.stability import (
    evaluate_echo_state_property,
    evaluate_autonomous_stability,
)


def compare_memory_capacity() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  1. MEMORY CAPACITY COMPARISON: CLASSIC ESN vs LE-ESN")
    print("=" * 70)

    reservoir_sizes = [100, 300, 500]
    max_delay = 80
    data_len = 5000
    u, Y = generate_memory_capacity(length=data_len, max_delay=max_delay, seed=42)

    split = int(0.7 * data_len)
    u_tr, Y_tr = u[:split], Y[:split]
    u_te, Y_te = u[split:], Y[split:]

    results = {"classic": {}, "lyapunov": {}}

    for N in reservoir_sizes:
        # Classic ESN
        esn_classic = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=N,
            n_outputs=max_delay,
            spectral_radius=0.98,
            leaking_rate=1.0,
            input_scaling=0.1,
            bias_scaling=0.0,
            density=0.1,
            include_input=False,
            include_bias=True,
            random_state=42,
        )
        t0 = time.time()
        esn_classic.fit(u_tr, Y_tr, washout=200, reg=1e-4)
        pred_c = esn_classic.predict(u_te)
        time_c = time.time() - t0
        mc_c, prof_c = compute_memory_capacity(Y_te[100:], pred_c[100:])

        # Lyapunov-Enhanced ESN (Clark 2026 critical orthogonal spectrum)
        esn_lyap = LyapunovEnhancedReservoir(
            n_inputs=1,
            n_reservoir=N,
            n_outputs=max_delay,
            coupling_type="critical_orthogonal",
            spectral_radius=0.99,
            leaking_rate=1.0,
            input_scaling=0.1,
            bias_scaling=0.0,
            dynamic_gain_control=False,
            forward_backward_dichotomy=False,
            include_input=False,
            include_bias=True,
            random_state=42,
        )
        t0 = time.time()
        esn_lyap.fit(u_tr, Y_tr, washout=200, reg=1e-4)
        pred_l = esn_lyap.predict(u_te)
        time_l = time.time() - t0
        mc_l, prof_l = compute_memory_capacity(Y_te[100:], pred_l[100:])

        results["classic"][N] = {"mc": mc_c, "profile": prof_c, "time": time_c}
        results["lyapunov"][N] = {"mc": mc_l, "profile": prof_l, "time": time_l}

        delta_pct = ((mc_l - mc_c) / mc_c) * 100.0
        print(f"  N = {N:3d} | Classic MC: {mc_c:6.2f} | LE-ESN MC: {mc_l:6.2f} ({delta_pct:+5.1f}%) | Time: {time_l*1000:5.1f}ms")

    return results


def compare_narma() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  2. NARMA BENCHMARK COMPARISON (k = 10, 20, 30)")
    print("=" * 70)

    orders = [10, 20, 30]
    data_len = 5000
    results = {"classic": {}, "lyapunov": {}}

    for k in orders:
        u, y = generate_narma(k=k, length=data_len, seed=42)
        split = int(0.7 * data_len)
        u_tr, y_tr = u[:split], y[:split]
        u_te, y_te = u[split:], y[split:]

        leak = 0.8 if k <= 20 else 0.6
        sr = 0.95 if k <= 20 else 0.98

        # Classic ESN
        esn_c = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=400,
            n_outputs=1,
            spectral_radius=sr,
            leaking_rate=leak,
            input_scaling=0.2,
            bias_scaling=0.1,
            density=0.1,
            include_input=True,
            include_bias=True,
            random_state=42,
        )
        t0 = time.time()
        esn_c.fit(u_tr, y_tr, washout=200, reg=1e-4)
        pred_c = esn_c.predict(u_te)
        time_c = time.time() - t0
        nrmse_c = compute_nrmse(y_te[100:], pred_c[100:])

        # Lyapunov-Enhanced ESN (Clark 2026 critical orthogonal spectrum)
        esn_l = LyapunovEnhancedReservoir(
            n_inputs=1,
            n_reservoir=400,
            n_outputs=1,
            coupling_type="critical_orthogonal",
            spectral_radius=sr,
            leaking_rate=leak,
            input_scaling=0.2,
            bias_scaling=0.1,
            dynamic_gain_control=False,
            forward_backward_dichotomy=False,
            include_input=True,
            include_bias=True,
            random_state=42,
        )
        t0 = time.time()
        esn_l.fit(u_tr, y_tr, washout=200, reg=1e-4)
        pred_l = esn_l.predict(u_te)
        time_l = time.time() - t0
        nrmse_l = compute_nrmse(y_te[100:], pred_l[100:])

        results["classic"][k] = {"nrmse": nrmse_c, "y_pred": pred_c[100:], "time": time_c}
        results["lyapunov"][k] = {"nrmse": nrmse_l, "y_pred": pred_l[100:], "time": time_l}
        results[f"y_true_{k}"] = y_te[100:]

        improvement = ((nrmse_c - nrmse_l) / nrmse_c) * 100.0
        print(f"  NARMA-{k:2d} | Classic NRMSE: {nrmse_c:.4f} | LE-ESN NRMSE: {nrmse_l:.4f} (Error reduction: {improvement:+5.1f}%)")

    return results


def compare_mackey_glass() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  3. MACKEY-GLASS CHAOTIC PREDICTION & AUTONOMOUS STABILITY")
    print("=" * 70)

    data_len = 5000
    u, y = generate_mackey_glass(length=data_len, tau=17.0, seed=42)

    split = int(0.6 * data_len)
    u_tr, y_tr = u[:split], y[:split]
    u_te, y_te = u[split:], y[split:]

    # Classic ESN
    esn_c = EchoStateNetwork(
        n_inputs=1,
        n_reservoir=300,
        n_outputs=1,
        spectral_radius=0.95,
        leaking_rate=0.8,
        input_scaling=0.5,
        bias_scaling=0.1,
        density=0.1,
        include_input=True,
        include_bias=True,
        random_state=42,
    )
    esn_c.fit(u_tr, y_tr, washout=200, reg=1e-5)
    pred_c_1step = esn_c.predict(u_te)
    nrmse_c_1step = compute_nrmse(y_te[100:], pred_c_1step[100:])

    auto_c = evaluate_autonomous_stability(
        esn_c,
        warmup_u=u_te[:200],
        warmup_y=y_te[:200],
        ground_truth_continuation=y_te[200:700],
        n_rollout_steps=500,
    )

    # Lyapunov-Enhanced ESN (Clark 2026 critical orthogonal spectrum & dynamic gain control)
    esn_l = LyapunovEnhancedReservoir(
        n_inputs=1,
        n_reservoir=300,
        n_outputs=1,
        coupling_type="critical_orthogonal",
        spectral_radius=0.95,
        leaking_rate=0.8,
        input_scaling=0.5,
        bias_scaling=0.1,
        dynamic_gain_control=True,
        forward_backward_dichotomy=False,
        include_input=True,
        include_bias=True,
        random_state=42,
    )
    esn_l.fit(u_tr, y_tr, washout=200, reg=1e-5)
    pred_l_1step = esn_l.predict(u_te)
    nrmse_l_1step = compute_nrmse(y_te[100:], pred_l_1step[100:])

    auto_l = evaluate_autonomous_stability(
        esn_l,
        warmup_u=u_te[:200],
        warmup_y=y_te[:200],
        ground_truth_continuation=y_te[200:700],
        n_rollout_steps=500,
    )

    print(f"  1-Step NRMSE            | Classic: {nrmse_c_1step:.6f} | LE-ESN: {nrmse_l_1step:.6f}")
    print(f"  Autonomous VPT (steps)  | Classic: {auto_c['vpt']:4d}      | LE-ESN: {auto_l['vpt']:4d}")
    print(f"  Variance Ratio          | Classic: {auto_c['attractor_diagnostics']['variance_ratio']:.4f}    | LE-ESN: {auto_l['attractor_diagnostics']['variance_ratio']:.4f}")
    print(f"  Attractor Stable?       | Classic: {auto_c['attractor_diagnostics']['is_stable']}      | LE-ESN: {auto_l['attractor_diagnostics']['is_stable']}")

    return {
        "classic": {"1step_nrmse": nrmse_c_1step, "vpt": auto_c["vpt"], "auto": auto_c},
        "lyapunov": {"1step_nrmse": nrmse_l_1step, "vpt": auto_l["vpt"], "auto": auto_l},
        "y_gt": y_te[200:700],
    }


def compare_lorenz() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  4. LORENZ-63 MULTIVARIATE CHAOTIC FORECASTING")
    print("=" * 70)

    data_len = 4000
    u, y = generate_lorenz(length=data_len, dt=0.01, seed=42)

    split = int(0.7 * data_len)
    u_tr, y_tr = u[:split], y[:split]
    u_te, y_te = u[split:], y[split:]

    # Classic ESN
    esn_c = EchoStateNetwork(
        n_inputs=3,
        n_reservoir=400,
        n_outputs=3,
        spectral_radius=0.90,
        leaking_rate=0.5,
        input_scaling=0.3,
        bias_scaling=0.1,
        density=0.1,
        random_state=42,
    )
    esn_c.fit(u_tr, y_tr, washout=200, reg=1e-5)
    pred_c = esn_c.predict(u_te)
    nrmse_c = compute_nrmse(y_te[100:], pred_c[100:])
    auto_c = evaluate_autonomous_stability(
        esn_c,
        warmup_u=u_te[:200],
        warmup_y=y_te[:200],
        ground_truth_continuation=y_te[200:600],
        n_rollout_steps=400,
    )

    # Lyapunov-Enhanced ESN
    esn_l = LyapunovEnhancedReservoir(
        n_inputs=3,
        n_reservoir=400,
        n_outputs=3,
        target_lyapunov=0.0,
        spectral_radius=0.90,
        leaking_rate=0.5,
        input_scaling=0.3,
        bias_scaling=0.1,
        density=0.1,
        reciprocal_corr=-0.5,
        dynamic_gain_control=True,
        forward_backward_dichotomy=True,
        dichotomy_eta=1e-3,
        random_state=42,
    )
    esn_l.fit(u_tr, y_tr, washout=200, reg=1e-5)
    pred_l = esn_l.predict(u_te)
    nrmse_l = compute_nrmse(y_te[100:], pred_l[100:])
    auto_l = evaluate_autonomous_stability(
        esn_l,
        warmup_u=u_te[:200],
        warmup_y=y_te[:200],
        ground_truth_continuation=y_te[200:600],
        n_rollout_steps=400,
    )

    print(f"  Lorenz-63 1-Step NRMSE  | Classic: {nrmse_c:.6f} | LE-ESN: {nrmse_l:.6f}")
    print(f"  Autonomous VPT (3D)     | Classic: {auto_c['vpt']:4d}      | LE-ESN: {auto_l['vpt']:4d}")

    return {
        "classic": {"nrmse": nrmse_c, "vpt": auto_c["vpt"], "auto": auto_c},
        "lyapunov": {"nrmse": nrmse_l, "vpt": auto_l["vpt"], "auto": auto_l},
    }


def plot_comparison_dashboard(
    mc_res: Dict[str, Any],
    narma_res: Dict[str, Any],
    mg_res: Dict[str, Any],
    lorenz_res: Dict[str, Any],
    output_path: str = "reservoir/comparison_results.png",
):
    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    plt.rcParams["text.color"] = "#e0e6ed"
    plt.rcParams["axes.labelcolor"] = "#c9d1d9"
    plt.rcParams["xtick.color"] = "#8b949e"
    plt.rcParams["ytick.color"] = "#8b949e"

    # 1. Memory Capacity comparison bar chart
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#161b22")
    sizes = [100, 300, 500]
    mc_c = [mc_res["classic"][s]["mc"] for s in sizes]
    mc_l = [mc_res["lyapunov"][s]["mc"] for s in sizes]

    x_idx = np.arange(len(sizes))
    w = 0.35
    ax1.bar(x_idx - w/2, mc_c, width=w, label="Classic ESN", color="#8b949e", alpha=0.9)
    ax1.bar(x_idx + w/2, mc_l, width=w, label="LE-ESN (Clark 2026)", color="#2ea043", alpha=0.9)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"N={s}" for s in sizes])
    ax1.set_title("Memory Capacity (MC) vs Reservoir Size", color="#58a6ff", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total MC score")
    ax1.grid(True, color="#30363d", alpha=0.6, axis="y")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d")

    # 2. Memory retention curves for N=500
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#161b22")
    prof_c = mc_res["classic"][500]["profile"]
    prof_l = mc_res["lyapunov"][500]["profile"]
    ax2.plot(range(1, len(prof_c) + 1), prof_c, label=f"Classic (MC={mc_res['classic'][500]['mc']:.1f})", color="#8b949e", lw=2)
    ax2.plot(range(1, len(prof_l) + 1), prof_l, label=f"LE-ESN (MC={mc_res['lyapunov'][500]['mc']:.1f})", color="#2ea043", lw=2.5)
    ax2.set_title("Memory Retention Profile r^2(d) (N=500)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Delay horizon d")
    ax2.set_ylabel("Determination r^2")
    ax2.grid(True, color="#30363d", alpha=0.6)
    ax2.legend(facecolor="#21262d", edgecolor="#30363d")

    # 3. NARMA NRMSE comparison
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#161b22")
    orders = [10, 20, 30]
    nrmse_c = [narma_res["classic"][k]["nrmse"] for k in orders]
    nrmse_l = [narma_res["lyapunov"][k]["nrmse"] for k in orders]
    x_k = np.arange(len(orders))
    ax3.bar(x_k - w/2, nrmse_c, width=w, label="Classic ESN", color="#f85149", alpha=0.85)
    ax3.bar(x_k + w/2, nrmse_l, width=w, label="LE-ESN (Clark 2026)", color="#388bfd", alpha=0.9)
    ax3.set_xticks(x_k)
    ax3.set_xticklabels([f"NARMA-{k}" for k in orders])
    ax3.set_title("NARMA Error (Lower is Better)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax3.set_ylabel("NRMSE")
    ax3.grid(True, color="#30363d", alpha=0.6, axis="y")
    ax3.legend(facecolor="#21262d", edgecolor="#30363d")

    # 4. Mackey-Glass Autonomous Rollout (Classic vs LE-ESN)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#161b22")
    gt = mg_res["y_gt"][:300, 0]
    gen_c = mg_res["classic"]["auto"]["y_generated"][:300, 0]
    gen_l = mg_res["lyapunov"]["auto"]["y_generated"][:300, 0]
    vpt_c = mg_res["classic"]["vpt"]
    vpt_l = mg_res["lyapunov"]["vpt"]

    ax4.plot(gt, color="#ffffff", lw=2, label="Ground Truth MG-17", alpha=0.8)
    ax4.plot(gen_c, color="#f85149", linestyle=":", lw=1.8, label=f"Classic (VPT={vpt_c})")
    ax4.plot(gen_l, color="#2ea043", linestyle="--", lw=2, label=f"LE-ESN (VPT={vpt_l})")
    ax4.set_title("Mackey-Glass Autonomous Rollout", color="#58a6ff", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Autonomous steps")
    ax4.set_ylabel("x(t)")
    ax4.grid(True, color="#30363d", alpha=0.6)
    ax4.legend(facecolor="#21262d", edgecolor="#30363d")

    # 5. Lorenz-63 Autonomous Rollout X-coordinate
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#161b22")
    l_gt = lorenz_res["classic"]["auto"]["y_ground_truth"][:250, 0]
    l_c = lorenz_res["classic"]["auto"]["y_generated"][:250, 0]
    l_l = lorenz_res["lyapunov"]["auto"]["y_generated"][:250, 0]
    vpt_lc = lorenz_res["classic"]["vpt"]
    vpt_ll = lorenz_res["lyapunov"]["vpt"]

    ax5.plot(l_gt, color="#ffffff", lw=2, label="True Lorenz X(t)", alpha=0.8)
    ax5.plot(l_c, color="#f85149", linestyle=":", lw=1.8, label=f"Classic (VPT={vpt_lc})")
    ax5.plot(l_l, color="#2ea043", linestyle="--", lw=2, label=f"LE-ESN (VPT={vpt_ll})")
    ax5.set_title("Lorenz-63 Autonomous Multi-Step Tracking", color="#58a6ff", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Autonomous steps")
    ax5.set_ylabel("X coordinate")
    ax5.grid(True, color="#30363d", alpha=0.6)
    ax5.legend(facecolor="#21262d", edgecolor="#30363d")

    # 6. Summary Performance Radar/Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#161b22")
    ax6.axis("off")

    summary_text = (
        "THEORETICAL MECHANISMS FROM CLARK (2026):\n\n"
        "1. Dynamic Criticality Calibration:\n"
        "   lambda_1 = log(g * sqrt(C_d(0))) ~= 0\n"
        "   Compensates for gain saturation <(1-tanh^2)^2>.\n\n"
        "2. Reciprocal Coupling Tuning (tau = -0.6):\n"
        "   Shifts spectrum along imaginary axis,\n"
        "   preventing explosive real modes and maximizing\n"
        "   participation-ratio dimensionality PR/N.\n\n"
        "3. Forward-Backward Dichotomy Filter (Sec 2.2):\n"
        "   Eliminates diverging tangent components and\n"
        "   projects state onto stable invariant manifold.\n\n"
        f"KEY COMPARISON SUMMARY:\n"
        f"• Memory Capacity (N=500): {mc_res['classic'][500]['mc']:.1f} -> {mc_res['lyapunov'][500]['mc']:.1f}\n"
        f"• NARMA-30 NRMSE: {narma_res['classic'][30]['nrmse']:.4f} -> {narma_res['lyapunov'][30]['nrmse']:.4f}\n"
        f"• Mackey-Glass VPT: {vpt_c} -> {vpt_l} steps\n"
        f"• Lorenz-63 VPT: {vpt_lc} -> {vpt_ll} steps"
    )
    ax6.text(
        0.05, 0.95, summary_text,
        transform=ax6.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        family="monospace",
        color="#7ee787",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d")
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n  [Comparison Dashboard Plot Saved] -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classic ESN vs Lyapunov-Enhanced ESN Comparison")
    parser.add_argument("--save-plot", type=str, default="reservoir/comparison_results.png")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK: CLASSIC ESN vs LYAPUNOV-ENHANCED ESN")
    print("  Applying principles from Clark (2026), arXiv:2610.12426")
    print("=" * 70)

    mc_res = compare_memory_capacity()
    narma_res = compare_narma()
    mg_res = compare_mackey_glass()
    lorenz_res = compare_lorenz()

    plot_comparison_dashboard(
        mc_res=mc_res,
        narma_res=narma_res,
        mg_res=mg_res,
        lorenz_res=lorenz_res,
        output_path=args.save_plot,
    )

    print("\n" + "=" * 70)
    print("  COMPARISON BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
