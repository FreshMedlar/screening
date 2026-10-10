"""
Evaluation metrics for Reservoir Computing benchmarks.
Includes:
- NRMSE (Normalized Root Mean Squared Error)
- NMSE (Normalized Mean Squared Error)
- Memory Capacity (MC) score
- VPT (Valid Prediction Time) for autonomous chaotic forecasting
- Attractor stability diagnostics
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Squared Error:
        NRMSE = sqrt( mean((y_true - y_pred)^2) / var(y_true) )
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check for NaN or Inf
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return float("inf")

    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    if var < 1e-12:
        return float(np.sqrt(mse))
    return float(np.sqrt(mse / var))


def compute_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error:
        NMSE = mean((y_true - y_pred)^2) / var(y_true)
    """
    nrmse = compute_nrmse(y_true, y_pred)
    if not np.isfinite(nrmse):
        return float("inf")
    return float(nrmse ** 2)


def compute_memory_capacity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, List[float]]:
    """
    Compute Information Memory Capacity (MC) score as sum of squared
    Pearson correlation coefficients across delay horizons.

    Parameters
    ----------
    y_true : np.ndarray, shape (N_samples, max_delay)
    y_pred : np.ndarray, shape (N_samples, max_delay)

    Returns
    -------
    total_mc : float
        Sum of determination coefficients r^2(d) over all delays.
    mc_profile : List[float]
        List of r^2(d) for d = 1 ... max_delay.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    max_delay = y_true.shape[1]
    mc_profile: List[float] = []

    for d in range(max_delay):
        yt = y_true[:, d]
        yp = y_pred[:, d]

        if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
            mc_profile.append(0.0)
            continue

        cov_matrix = np.cov(yt, yp)
        var_t = cov_matrix[0, 0]
        var_p = cov_matrix[1, 1]
        cov_tp = cov_matrix[0, 1]

        if var_t > 1e-12 and var_p > 1e-12:
            r2 = float((cov_tp ** 2) / (var_t * var_p))
            # Bound determination coefficient between 0 and 1
            r2 = max(0.0, min(1.0, r2))
            mc_profile.append(r2)
        else:
            mc_profile.append(0.0)

    total_mc = float(np.sum(mc_profile))
    return total_mc, mc_profile


def compute_vpt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.4,
    lyapunov_time: Optional[float] = None,
) -> Union[int, Tuple[int, float]]:
    """
    Valid Prediction Time (VPT):
    The number of autonomous prediction steps before the normalized error
    |y_true(t) - y_pred(t)| / std(y_true) exceeds the threshold.

    Parameters
    ----------
    y_true : np.ndarray, shape (T, D) or (T,)
    y_pred : np.ndarray, shape (T, D) or (T,)
    threshold : float
        Error threshold (standard literature uses 0.4 or 0.5).
    lyapunov_time : Optional[float]
        If provided, converts steps into units of Lyapunov times (VPT_steps / steps_per_lyapunov).

    Returns
    -------
    vpt_steps : int (or tuple (vpt_steps, vpt_lyapunov) if lyapunov_time is given)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    T = len(y_true)
    std_t = np.std(y_true, axis=0)
    std_t[std_t < 1e-12] = 1.0

    err = np.abs(y_true - y_pred) / std_t  # Shape (T, D)
    max_err_per_step = np.max(err, axis=1)  # Worst component error

    vpt = T
    for t in range(T):
        if not np.isfinite(max_err_per_step[t]) or max_err_per_step[t] > threshold:
            vpt = t
            break

    if lyapunov_time is not None:
        vpt_lyap = float(vpt / lyapunov_time)
        return vpt, vpt_lyap
    return vpt


def compute_attractor_divergence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Union[bool, float]]:
    """
    Assess long-term attractor stability of autonomous rollouts:
    - is_diverged: True if predictions blow up (NaN, Inf, or absolute values > 10x true span)
    - is_collapsed: True if predictions settle into a static fixed point (var < 1e-4)
    - is_stable: True if neither diverged nor collapsed
    - variance_ratio: ratio of pred variance to true variance
    - mean_error: absolute difference in empirical means
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    has_non_finite = not (np.all(np.isfinite(y_pred)))
    if has_non_finite:
        return {
            "is_diverged": True,
            "is_collapsed": False,
            "is_stable": False,
            "variance_ratio": 0.0,
            "mean_error": float("inf"),
        }

    span_true = float(np.ptp(y_true))
    span_pred = float(np.ptp(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))

    is_diverged = (span_pred > 20.0 * (span_true + 1e-6))
    is_collapsed = (var_pred < 1e-4 * (var_true + 1e-6))
    is_stable = bool((not is_diverged) and (not is_collapsed))

    var_ratio = float(var_pred / (var_true + 1e-12))
    mean_err = float(np.abs(np.mean(y_true) - np.mean(y_pred)))

    return {
        "is_diverged": is_diverged,
        "is_collapsed": is_collapsed,
        "is_stable": is_stable,
        "variance_ratio": var_ratio,
        "mean_error": mean_err,
    }
