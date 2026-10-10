#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Echo State Networks (ESN):
Evaluating Memory and Stability on:
1. Linear Memory Capacity (Jaeger 2002) across reservoir sizes
2. NARMA-10, NARMA-20, NARMA-30 (memory & nonlinearity)
3. Mackey-Glass chaotic time series (1-step prediction & autonomous rollout stability)
4. Echo State Property (ESP) convergence & contraction rate
5. Spectral radius sweep (Edge-of-Chaos phase transition)
6. Lorenz-63 3D chaotic attractor forecasting

Run with:
    /home/medlar/Projects/.venv/bin/python -m reservoir.benchmark
or
    /home/medlar/Projects/.venv/bin/python reservoir/benchmark.py
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reservoir.esn import EchoStateNetwork
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
    evaluate_spectral_radius_sweep,
)


def run_memory_capacity_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run Memory Capacity benchmarks across reservoir dimensions N."""
    if verbose:
        print("\n" + "=" * 65)
        print("  1. INFORMATION MEMORY CAPACITY (MC) BENCHMARK")
        print("=" * 65)

    reservoir_sizes = [100, 300, 500]
    max_delay = 80
    data_len = 5000
    u, Y = generate_memory_capacity(length=data_len, max_delay=max_delay, seed=42)

    split = int(0.7 * data_len)
    u_tr, Y_tr = u[:split], Y[:split]
    u_te, Y_te = u[split:], Y[split:]

    results = {}
    for N in reservoir_sizes:
        esn = EchoStateNetwork(
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
        esn.fit(u_tr, Y_tr, washout=200, reg=1e-4)
        Y_pred = esn.predict(u_te)
        fit_time = time.time() - t0

        mc_score, mc_profile = compute_memory_capacity(Y_te[100:], Y_pred[100:])
        results[N] = {
            "mc_score": mc_score,
            "mc_profile": mc_profile,
            "theoretical_max": float(N),
            "ratio": mc_score / N,
            "fit_time": fit_time,
        }

        if verbose:
            print(f"  N = {N:3d} | Total MC: {mc_score:6.2f} / {N:3d} ({mc_score/N*100:5.1f}%) | Time: {fit_time*1000:6.1f}ms")

    return results


def run_narma_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run NARMA-10, NARMA-20, and NARMA-30 benchmarks."""
    if verbose:
        print("\n" + "=" * 65)
        print("  2. NARMA NONLINEAR MEMORY BENCHMARK (k = 10, 20, 30)")
        print("=" * 65)

    orders = [10, 20, 30]
    data_len = 5000
    results = {}

    for k in orders:
        u, y = generate_narma(k=k, length=data_len, seed=42)
        split = int(0.7 * data_len)
        u_tr, y_tr = u[:split], y[:split]
        u_te, y_te = u[split:], y[split:]

        # Appropriate leaking rate and spectral radius for memory scaling
        leaking_rate = 0.8 if k <= 20 else 0.6
        spectral_radius = 0.95 if k <= 20 else 0.98

        esn = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=400,
            n_outputs=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            input_scaling=0.2,
            bias_scaling=0.1,
            density=0.1,
            include_input=True,
            include_bias=True,
            random_state=42,
        )
        t0 = time.time()
        esn.fit(u_tr, y_tr, washout=200, reg=1e-4)
        y_pred = esn.predict(u_te)
        fit_time = time.time() - t0

        test_wash = 100
        nrmse = compute_nrmse(y_te[test_wash:], y_pred[test_wash:])
        nmse = compute_nmse(y_te[test_wash:], y_pred[test_wash:])

        results[f"narma_{k}"] = {
            "k": k,
            "nrmse": nrmse,
            "nmse": nmse,
            "fit_time": fit_time,
            "y_true": y_te[test_wash:],
            "y_pred": y_pred[test_wash:],
        }

        if verbose:
            print(f"  NARMA-{k:2d} | NRMSE: {nrmse:.4f} | NMSE: {nmse:.5f} | Fit Time: {fit_time*1000:6.1f}ms")

    return results


def run_mackey_glass_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run Mackey-Glass 1-step prediction and autonomous rollout stability benchmark."""
    if verbose:
        print("\n" + "=" * 65)
        print("  3. MACKEY-GLASS CHAOTIC TIME SERIES & STABILITY BENCHMARK")
        print("=" * 65)

    data_len = 5000
    u, y = generate_mackey_glass(length=data_len, tau=17.0, seed=42)

    split = int(0.6 * data_len)
    u_tr, y_tr = u[:split], y[:split]
    u_te, y_te = u[split:], y[split:]

    esn = EchoStateNetwork(
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

    t0 = time.time()
    esn.fit(u_tr, y_tr, washout=200, reg=1e-5)
    fit_time = time.time() - t0

    # 1. One-step ahead open-loop prediction
    y_pred_1step = esn.predict(u_te)
    test_wash = 100
    nrmse_1step = compute_nrmse(y_te[test_wash:], y_pred_1step[test_wash:])

    # 2. Closed-loop autonomous rollout stability
    rollout_len = 500
    warmup_steps = 200
    auto_res = evaluate_autonomous_stability(
        esn,
        warmup_u=u_te[:warmup_steps],
        warmup_y=y_te[:warmup_steps],
        ground_truth_continuation=y_te[warmup_steps : warmup_steps + rollout_len],
        n_rollout_steps=rollout_len,
    )

    vpt = auto_res["vpt"]
    diag = auto_res["attractor_diagnostics"]
    auto_nrmse = auto_res["rollout_nrmse"]

    if verbose:
        print(f"  1-Step Ahead Open-Loop NRMSE   : {nrmse_1step:.6f}")
        print(f"  Autonomous Rollout Steps       : {rollout_len}")
        print(f"  Valid Prediction Time (VPT)    : {vpt} steps (threshold=0.4)")
        print(f"  Attractor Stability Status     : {'STABLE' if diag['is_stable'] else 'UNSTABLE'}")
        print(f"  Variance Ratio (Pred / True)   : {diag['variance_ratio']:.4f}")
        print(f"  Fit + Eval Time                : {fit_time*1000:6.1f}ms")

    return {
        "1step_nrmse": nrmse_1step,
        "vpt": vpt,
        "auto_res": auto_res,
        "y_true_1step": y_te[test_wash:],
        "y_pred_1step": y_pred_1step[test_wash:],
        "fit_time": fit_time,
    }


def run_esp_and_sweep_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Evaluate Echo State Property (ESP) and spectral radius sweep."""
    if verbose:
        print("\n" + "=" * 65)
        print("  4. ECHO STATE PROPERTY & SPECTRAL RADIUS SWEEP")
        print("=" * 65)

    # 1. Single model ESP contraction test
    esn_sample = EchoStateNetwork(
        n_inputs=1,
        n_reservoir=200,
        spectral_radius=0.95,
        leaking_rate=0.8,
        input_scaling=0.5,
        random_state=42,
    )
    esp_test = evaluate_echo_state_property(esn_sample, input_length=1000, n_trials=5, seed=42)

    if verbose:
        print(f"  Echo State Property Satisfied  : {esp_test['esp_satisfied']}")
        print(f"  Mean Final State Distance      : {esp_test['mean_final_distance']:.2e}")
        print(f"  Mean State Contraction Rate    : {esp_test['mean_contraction_rate']:.4f} per step")
        print(f"  Mean Settling Steps            : {esp_test['mean_settling_steps']:.1f}")

    # 2. Spectral radius sweep
    if verbose:
        print("\n  Running Spectral Radius Sweep (rho in [0.2, 1.5])...")
    sweep_results = evaluate_spectral_radius_sweep(
        spectral_radii=[0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4],
        n_reservoir=200,
        mc_length=3500,
        max_delay=50,
        mg_length=2500,
        mg_rollout=250,
        seed=42,
    )

    if verbose:
        print("  " + "-" * 61)
        print("   rho  |  MC Score  | Contraction Rate | MG 1-Step | MG VPT | Stable?")
        print("  " + "-" * 61)
        for i, sr in enumerate(sweep_results["spectral_radii"]):
            mc_val = sweep_results["memory_capacity"][i]
            crate = sweep_results["esp_contraction_rate"][i]
            mg_err = sweep_results["mg_1step_nrmse"][i]
            vpt_val = sweep_results["mg_autonomous_vpt"][i]
            stable_val = sweep_results["mg_autonomous_stable"][i]
            print(f"  {sr:5.2f} |   {mc_val:6.2f}   |      {crate:7.4f}     |  {mg_err:8.5f} |  {vpt_val:4d}  | {'Yes' if stable_val else 'No'}")
        print("  " + "-" * 61)

    return {
        "esp_test": esp_test,
        "sweep_results": sweep_results,
    }


def run_lorenz_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run Lorenz-63 3D chaotic attractor benchmark."""
    if verbose:
        print("\n" + "=" * 65)
        print("  5. LORENZ-63 MULTIVARIATE CHAOTIC BENCHMARK")
        print("=" * 65)

    data_len = 4000
    u, y = generate_lorenz(length=data_len, dt=0.01, seed=42)

    split = int(0.7 * data_len)
    u_tr, y_tr = u[:split], y[:split]
    u_te, y_te = u[split:], y[split:]

    esn = EchoStateNetwork(
        n_inputs=3,
        n_reservoir=400,
        n_outputs=3,
        spectral_radius=0.90,
        leaking_rate=0.5,
        input_scaling=0.3,
        bias_scaling=0.1,
        density=0.1,
        include_input=True,
        include_bias=True,
        random_state=42,
    )
    t0 = time.time()
    esn.fit(u_tr, y_tr, washout=200, reg=1e-5)
    fit_time = time.time() - t0

    y_pred = esn.predict(u_te)
    nrmse = compute_nrmse(y_te[100:], y_pred[100:])

    # Autonomous rollout
    rollout_len = 400
    auto_res = evaluate_autonomous_stability(
        esn,
        warmup_u=u_te[:200],
        warmup_y=y_te[:200],
        ground_truth_continuation=y_te[200 : 200 + rollout_len],
        n_rollout_steps=rollout_len,
    )

    if verbose:
        print(f"  Lorenz-63 1-Step NRMSE         : {nrmse:.6f}")
        print(f"  Autonomous Rollout VPT (3D)    : {auto_res['vpt']} steps")
        print(f"  Attractor Stable               : {auto_res['attractor_diagnostics']['is_stable']}")
        print(f"  Fit Time                       : {fit_time*1000:6.1f}ms")

    return {
        "nrmse": nrmse,
        "auto_res": auto_res,
        "y_true": y_te[100:],
        "y_pred": y_pred[100:],
    }


def plot_benchmark_dashboard(
    mc_res: Dict[str, Any],
    narma_res: Dict[str, Any],
    mg_res: Dict[str, Any],
    esp_res: Dict[str, Any],
    lorenz_res: Dict[str, Any],
    output_path: str = "reservoir/benchmark_results.png",
):
    """Generate high-quality diagnostic figures summarizing memory and stability."""
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    # Dark theme palette
    plt.rcParams["text.color"] = "#e0e6ed"
    plt.rcParams["axes.labelcolor"] = "#c9d1d9"
    plt.rcParams["xtick.color"] = "#8b949e"
    plt.rcParams["ytick.color"] = "#8b949e"

    # 1. Memory Capacity retention curves
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#161b22")
    for N, data in mc_res.items():
        profile = data["mc_profile"]
        ax1.plot(range(1, len(profile) + 1), profile, label=f"N={N} (MC={data['mc_score']:.1f})", lw=2)
    ax1.set_title("Information Memory Capacity r^2(d)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Delay horizon d (steps)")
    ax1.set_ylabel("Determination r^2")
    ax1.grid(True, color="#30363d", alpha=0.6)
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9)

    # 2. NARMA-10 vs NARMA-30 predictions
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#161b22")
    y_true_n10 = narma_res["narma_10"]["y_true"][:120]
    y_pred_n10 = narma_res["narma_10"]["y_pred"][:120]
    ax2.plot(y_true_n10, color="#7ee787", label=f"NARMA-10 True", lw=2)
    ax2.plot(y_pred_n10, color="#ffa657", linestyle="--", label=f"NARMA-10 Pred (NRMSE={narma_res['narma_10']['nrmse']:.3f})", lw=1.8)
    ax2.set_title("NARMA Time Series Tracking", color="#58a6ff", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Output y(t)")
    ax2.grid(True, color="#30363d", alpha=0.6)
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9)

    # 3. Mackey-Glass Autonomous Rollout & VPT
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#161b22")
    y_gen = mg_res["auto_res"]["y_generated"][:300, 0]
    y_gt = mg_res["auto_res"]["y_ground_truth"][:300, 0]
    vpt = mg_res["vpt"]
    ax3.plot(y_gt, color="#79c0ff", label="True MG-17", lw=2)
    ax3.plot(y_gen, color="#f778ba", linestyle="--", label="Autonomous ESN", lw=1.8)
    if vpt < len(y_gt):
        ax3.axvline(x=vpt, color="#ff7b72", linestyle=":", lw=2, label=f"VPT = {vpt} steps")
    ax3.set_title("Mackey-Glass Autonomous Rollout (Closed-Loop)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Autonomous steps")
    ax3.set_ylabel("x(t)")
    ax3.grid(True, color="#30363d", alpha=0.6)
    ax3.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9)

    # 4. Echo State Property: State Distance Contraction
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#161b22")
    trajs = esp_res["esp_test"]["distance_trajectories"]
    for i, tr in enumerate(trajs[:3]):
        ax4.semilogy(tr[:400], alpha=0.8, lw=1.5, label=f"Trial {i+1}")
    ax4.axhline(y=1e-4, color="#ff7b72", linestyle=":", label="ESP Threshold")
    ax4.set_title("Echo State Property: ||x_a(t) - x_b(t)|| -> 0", color="#58a6ff", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Driving step t")
    ax4.set_ylabel("State Euclidean distance (log scale)")
    ax4.grid(True, color="#30363d", alpha=0.6)
    ax4.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=9)

    # 5. Spectral Radius Edge of Chaos Sweep (Memory vs Stability)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#161b22")
    sw = esp_res["sweep_results"]
    srs = sw["spectral_radii"]
    mcs = sw["memory_capacity"]
    crates = sw["esp_contraction_rate"]

    ax5_twin = ax5.twinx()
    p1 = ax5.plot(srs, mcs, color="#d2a8ff", marker="o", lw=2, label="Memory Capacity (MC)")
    p2 = ax5_twin.plot(srs, crates, color="#7ee787", marker="s", linestyle="--", lw=2, label="Contraction Rate")
    ax5.axvline(x=1.0, color="#ffa657", linestyle=":", lw=1.5, label="Boundary (rho=1.0)")

    ax5.set_title("Edge of Chaos: Memory vs Stability Sweep", color="#58a6ff", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Spectral Radius rho(W)")
    ax5.set_ylabel("Memory Capacity", color="#d2a8ff")
    ax5_twin.set_ylabel("Contraction Rate (negative = stable)", color="#7ee787")
    ax5.grid(True, color="#30363d", alpha=0.6)

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, facecolor="#21262d", edgecolor="#30363d", fontsize=8, loc="upper right")

    # 6. Lorenz Attractor 3D Phase Portrait
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    ax6.set_facecolor("#161b22")
    l_gt = lorenz_res["auto_res"]["y_ground_truth"][:300]
    l_gen = lorenz_res["auto_res"]["y_generated"][:300]

    ax6.plot(l_gt[:, 0], l_gt[:, 1], l_gt[:, 2], color="#58a6ff", lw=1.5, alpha=0.8, label="True Orbit")
    ax6.plot(l_gen[:, 0], l_gen[:, 1], l_gen[:, 2], color="#f778ba", linestyle="--", lw=1.5, alpha=0.8, label="ESN Orbit")
    ax6.set_title("Lorenz-63 Attractor Rollout", color="#58a6ff", fontsize=12, fontweight="bold")
    ax6.grid(False)
    ax6.legend(facecolor="#21262d", edgecolor="#30363d", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n  [Dashboard Plot Saved] -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Echo State Network Memory & Stability Benchmark Suite")
    parser.add_argument("--save-plot", type=str, default="reservoir/benchmark_results.png", help="Path to save summary figure")
    parser.add_argument("--skip-plot", action="store_true", help="Skip generating plot figure")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  ECHO STATE NETWORK (ESN) MEMORY & STABILITY BENCHMARK SUITE")
    print("=" * 65)

    mc_res = run_memory_capacity_benchmark(verbose=True)
    narma_res = run_narma_benchmark(verbose=True)
    mg_res = run_mackey_glass_benchmark(verbose=True)
    esp_res = run_esp_and_sweep_benchmark(verbose=True)
    lorenz_res = run_lorenz_benchmark(verbose=True)

    if not args.skip_plot:
        plot_benchmark_dashboard(
            mc_res=mc_res,
            narma_res=narma_res,
            mg_res=mg_res,
            esp_res=esp_res,
            lorenz_res=lorenz_res,
            output_path=args.save_plot,
        )

    print("\n" + "=" * 65)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
