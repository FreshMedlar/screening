"""
Stability and Echo State Property (ESP) analysis tools for Reservoir Computing.

Measures:
1. Echo State Property empirical verification (state contraction from divergent initial conditions)
2. Spectral radius edge-of-chaos phase transition sweep (Memory vs Stability trade-off)
3. Autonomous free-running rollout stability on chaotic attractors
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .esn import EchoStateNetwork
from .metrics import (
    compute_nrmse,
    compute_memory_capacity,
    compute_vpt,
    compute_attractor_divergence,
)
from .data import generate_memory_capacity, generate_mackey_glass


def evaluate_echo_state_property(
    esn: EchoStateNetwork,
    input_length: int = 1500,
    n_trials: int = 5,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Empirically verify the Echo State Property (ESP).

    For each trial:
    - Initialize two states x_a(0) and x_b(0) with random vectors ||x_a(0) - x_b(0)|| > 0.5.
    - Drive both with the exact same input sequence u(t).
    - Track the state divergence distance delta(t) = ||x_a(t) - x_b(t)||_2 over time.
    - If the ESP holds, delta(t) -> 0 asymptotically (the network forgets its initial state).

    Parameters
    ----------
    esn : EchoStateNetwork
    input_length : int
        Number of steps to drive the reservoirs.
    n_trials : int
        Number of independent initial state pairs tested.
    tolerance : float
        Threshold for concluding that trajectories have converged.
    seed : int

    Returns
    -------
    results : Dict[str, Any]
        Contains:
        - esp_satisfied: bool (True if all trials converged below tolerance)
        - mean_final_distance: float
        - mean_settling_steps: float (average steps to reach tolerance, or input_length if not reached)
        - mean_contraction_rate: float (Lyapunov-like state contraction rate per step)
        - distance_trajectories: List[np.ndarray] (sample trajectories of delta(t))
    """
    rng = np.random.default_rng(seed)
    u_seq = rng.uniform(-1.0, 1.0, size=(input_length, esn.n_inputs))

    final_distances: List[float] = []
    settling_steps_list: List[int] = []
    contraction_rates: List[float] = []
    sample_trajectories: List[np.ndarray] = []

    for trial in range(n_trials):
        x_a = rng.uniform(-1.0, 1.0, size=esn.n_reservoir)
        x_b = rng.uniform(-1.0, 1.0, size=esn.n_reservoir)

        # Ensure initial separation
        initial_dist = float(np.linalg.norm(x_a - x_b))
        if initial_dist < 1e-3:
            x_b += 0.5
            initial_dist = float(np.linalg.norm(x_a - x_b))

        X_a, _ = esn.harvest_states(u_seq, initial_state=x_a)
        X_b, _ = esn.harvest_states(u_seq, initial_state=x_b)

        dists = np.linalg.norm(X_a - X_b, axis=1)  # Shape (input_length,)
        sample_trajectories.append(dists)

        final_d = float(dists[-1])
        final_distances.append(final_d)

        # Settling time
        converged_indices = np.where(dists < tolerance)[0]
        if len(converged_indices) > 0:
            settling_step = int(converged_indices[0])
        else:
            settling_step = input_length
        settling_steps_list.append(settling_step)

        # Empirical contraction rate lambda: d(t) ~= d(0) * exp(lambda * t)
        safe_final = max(final_d, 1e-15)
        rate = float(np.log(safe_final / initial_dist) / input_length)
        contraction_rates.append(rate)

    mean_final_dist = float(np.mean(final_distances))
    esp_satisfied = bool(mean_final_dist < tolerance)

    return {
        "esp_satisfied": esp_satisfied,
        "mean_final_distance": mean_final_dist,
        "mean_settling_steps": float(np.mean(settling_steps_list)),
        "mean_contraction_rate": float(np.mean(contraction_rates)),
        "distance_trajectories": sample_trajectories,
    }


def evaluate_autonomous_stability(
    esn: EchoStateNetwork,
    warmup_u: np.ndarray,
    warmup_y: np.ndarray,
    ground_truth_continuation: np.ndarray,
    n_rollout_steps: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate autonomous closed-loop stability and accuracy on continuation of a time series.

    1. Prime reservoir state with warmup sequence (teacher forcing).
    2. Switch to autonomous generation for n_rollout_steps.
    3. Measure VPT (Valid Prediction Time), NRMSE, and attractor divergence indicators.
    """
    warmup_u = np.asarray(warmup_u, dtype=np.float64)
    if warmup_u.ndim == 1:
        warmup_u = warmup_u.reshape(-1, 1)

    gt = np.asarray(ground_truth_continuation, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)

    # Prime reservoir
    esn.reset_state()
    _, Z_warm = esn.harvest_states(warmup_u, Y_fb=warmup_y)
    last_u = warmup_y[-1] if warmup_y is not None else warmup_u[-1]

    # Generate autonomously
    y_gen = esn.generate(n_rollout_steps, initial_u=last_u, continuation=True)

    # Truncate gt to rollout steps
    gt_eval = gt[:n_rollout_steps]
    vpt = compute_vpt(gt_eval, y_gen, threshold=0.4)
    nrmse = compute_nrmse(gt_eval, y_gen)
    attractor_diag = compute_attractor_divergence(gt_eval, y_gen)

    return {
        "y_generated": y_gen,
        "y_ground_truth": gt_eval,
        "vpt": vpt,
        "rollout_nrmse": nrmse,
        "attractor_diagnostics": attractor_diag,
    }


def evaluate_spectral_radius_sweep(
    spectral_radii: Optional[List[float]] = None,
    n_reservoir: int = 200,
    mc_length: int = 4000,
    max_delay: int = 60,
    mg_length: int = 3000,
    mg_rollout: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sweep spectral radius across a wide range (subcritical to supercritical)
    measuring the trade-off between Memory Capacity, ESP contraction rate,
    and Mackey-Glass autonomous stability.

    Highlights the 'Edge of Chaos' phenomenon where memory capacity peaks
    just near the stability boundary (rho ~ 0.95 - 1.05).
    """
    if spectral_radii is None:
        spectral_radii = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6]

    # Prepare Memory Capacity data
    u_mc, Y_mc = generate_memory_capacity(length=mc_length, max_delay=max_delay, seed=seed)
    split_mc = int(0.7 * mc_length)
    u_mc_tr, Y_mc_tr = u_mc[:split_mc], Y_mc[:split_mc]
    u_mc_te, Y_mc_te = u_mc[split_mc:], Y_mc[split_mc:]

    # Prepare Mackey-Glass data
    u_mg, y_mg = generate_mackey_glass(length=mg_length, tau=17.0, seed=seed)
    split_mg = int(0.6 * mg_length)
    u_mg_tr, y_mg_tr = u_mg[:split_mg], y_mg[:split_mg]
    u_mg_te, y_mg_te = u_mg[split_mg:], y_mg[split_mg:]

    results: Dict[str, List[Any]] = {
        "spectral_radii": spectral_radii,
        "memory_capacity": [],
        "esp_satisfied": [],
        "esp_contraction_rate": [],
        "mg_1step_nrmse": [],
        "mg_autonomous_vpt": [],
        "mg_autonomous_stable": [],
    }

    for sr in spectral_radii:
        # 1. Memory Capacity test
        esn_mc = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=n_reservoir,
            n_outputs=max_delay,
            spectral_radius=sr,
            leaking_rate=1.0,
            input_scaling=0.1,
            bias_scaling=0.0,
            density=0.1,
            include_input=False,
            include_bias=True,
            random_state=seed,
        )
        esn_mc.fit(u_mc_tr, Y_mc_tr, washout=200, reg=1e-4)
        Y_mc_pred = esn_mc.predict(u_mc_te)
        mc_score, _ = compute_memory_capacity(Y_mc_te[100:], Y_mc_pred[100:])
        results["memory_capacity"].append(mc_score)

        # 2. Echo State Property test
        esp_info = evaluate_echo_state_property(esn_mc, input_length=800, n_trials=3, seed=seed)
        results["esp_satisfied"].append(esp_info["esp_satisfied"])
        results["esp_contraction_rate"].append(esp_info["mean_contraction_rate"])

        # 3. Mackey-Glass prediction & Autonomous rollout stability
        esn_mg = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=n_reservoir,
            n_outputs=1,
            spectral_radius=sr,
            leaking_rate=0.8,
            input_scaling=0.5,
            bias_scaling=0.1,
            density=0.1,
            include_input=True,
            include_bias=True,
            random_state=seed,
        )
        esn_mg.fit(u_mg_tr, y_mg_tr, washout=200, reg=1e-5)

        # One-step ahead test
        y_mg_pred = esn_mg.predict(u_mg_te)
        mg_nrmse = compute_nrmse(y_mg_te[100:], y_mg_pred[100:])
        results["mg_1step_nrmse"].append(mg_nrmse)

        # Autonomous rollout
        auto_info = evaluate_autonomous_stability(
            esn_mg,
            warmup_u=u_mg_te[:200],
            warmup_y=y_mg_te[:200],
            ground_truth_continuation=y_mg_te[200 : 200 + mg_rollout],
            n_rollout_steps=mg_rollout,
        )
        results["mg_autonomous_vpt"].append(auto_info["vpt"])
        results["mg_autonomous_stable"].append(auto_info["attractor_diagnostics"]["is_stable"])

    return results
