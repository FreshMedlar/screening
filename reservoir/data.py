"""
Benchmark datasets and dynamical systems generators for Reservoir Computing.
Includes:
- Mackey-Glass chaotic time series (Runge-Kutta 4th order delayed differential equation)
- NARMA (k=10, 20, 30) nonlinear autoregressive moving average
- Linear & Nonlinear Memory Capacity tasks
- Lorenz-63 chaotic attractor
"""

from typing import Optional, Tuple
import numpy as np


def generate_mackey_glass(
    length: int = 5000,
    tau: float = 17.0,
    dt: float = 0.1,
    sample_stride: int = 10,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    warmup: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Mackey-Glass chaotic time series using Runge-Kutta 4th order (RK4) integration.

    Differential equation with delay:
        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

    Parameters
    ----------
    length : int
        Number of output time steps to generate (after downsampling and warmup).
    tau : float
        Delay parameter (tau=17 produces standard mild chaos, tau=30 produces high chaos).
    dt : float
        Internal integration step size.
    sample_stride : int
        Subsampling factor (dt * sample_stride = output sampling interval).
    beta, gamma, n : float
        Mackey-Glass standard equation parameters.
    warmup : int
        Warm-up steps to discard to ensure the state is on the chaotic attractor.
    seed : int
        Random seed for initial history.

    Returns
    -------
    u : np.ndarray, shape (length, 1)
        Current series values x(t).
    y : np.ndarray, shape (length, 1)
        One-step-ahead target values x(t+1).
    """
    rng = np.random.default_rng(seed)

    # Delay in terms of integration steps
    delay_steps = int(round(tau / dt))
    if delay_steps < 1:
        delay_steps = 1

    total_sampled_steps = length + warmup + 1
    total_integration_steps = total_sampled_steps * sample_stride

    # Initial history buffer: values around 1.1 with small perturbation
    history_len = delay_steps + 1
    history = 1.1 + 0.1 * (rng.uniform(-1.0, 1.0, size=history_len))

    # Ring buffer for delays
    buffer = np.zeros(delay_steps, dtype=np.float64)
    buffer[:] = history[-delay_steps:]
    buf_idx = 0

    x_curr = float(history[-1])

    def mg_rhs(x_now: float, x_delayed: float) -> float:
        return (beta * x_delayed) / (1.0 + (x_delayed ** n)) - gamma * x_now

    sampled_values = np.empty(total_sampled_steps, dtype=np.float64)
    sampled_idx = 0

    for step in range(total_integration_steps):
        if step % sample_stride == 0 and sampled_idx < total_sampled_steps:
            sampled_values[sampled_idx] = x_curr
            sampled_idx += 1

        x_delayed = buffer[buf_idx]

        # RK4 step
        k1 = mg_rhs(x_curr, x_delayed)
        k2 = mg_rhs(x_curr + 0.5 * dt * k1, x_delayed)
        k3 = mg_rhs(x_curr + 0.5 * dt * k2, x_delayed)
        k4 = mg_rhs(x_curr + dt * k3, x_delayed)

        x_next = x_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Store in circular buffer
        buffer[buf_idx] = x_curr
        buf_idx = (buf_idx + 1) % delay_steps
        x_curr = x_next

    # Discard warmup
    valid_series = sampled_values[warmup:]
    u = valid_series[:-1].reshape(-1, 1)
    y = valid_series[1:].reshape(-1, 1)
    return u[:length], y[:length]


def generate_narma(
    k: int = 10,
    length: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NARMA-k (Nonlinear Auto-Regressive Moving Average) time series.

    General formula:
        y(t+1) = alpha * y(t) + beta * y(t) * sum_{i=0}^{k-1} y(t-i)
                 + gamma * u(t - (k-1)) * u(t) + delta

    Input u(t) drawn from Uniform[0, 0.5].
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=length)
    y = np.zeros(length, dtype=np.float64)

    # Standard benchmark coefficients
    if k == 10:
        alpha, beta, gamma, delta = 0.3, 0.05, 1.5, 0.1
    elif k == 20:
        alpha, beta, gamma, delta = 0.3, 0.025, 1.5, 0.1
    elif k == 30:
        alpha, beta, gamma, delta = 0.2, 0.04, 1.5, 0.001
    else:
        alpha, beta, gamma, delta = 0.3, 0.05 * (10.0 / k), 1.5, 0.1

    # Simulate recurrence
    for t in range(k, length - 1):
        y_sum = np.sum(y[t - k + 1 : t + 1])
        y[t + 1] = alpha * y[t] + beta * y[t] * y_sum + gamma * u[t - k + 1] * u[t] + delta

    return u.reshape(-1, 1), y.reshape(-1, 1)


def generate_memory_capacity(
    length: int = 5000,
    max_delay: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset for the Information Memory Capacity (MC) task (Jaeger 2002).

    Input: u(t) ~ Uniform[-1, 1]
    Target at delay d: y_d(t) = u(t - d) for d = 1 ... max_delay.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=length)
    Y = np.zeros((length, max_delay), dtype=np.float64)

    for d in range(1, max_delay + 1):
        Y[d:, d - 1] = u[:-d]

    return u.reshape(-1, 1), Y


def generate_nonlinear_memory_capacity(
    length: int = 5000,
    max_delay: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate quadratic non-linear memory task: target is u(t - d) * u(t).
    Tests reservoir nonlinear memory retrieval capacity.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=length)
    Y = np.zeros((length, max_delay), dtype=np.float64)

    for d in range(1, max_delay + 1):
        Y[d:, d - 1] = u[:-d] * u[d:]

    return u.reshape(-1, 1), Y


def generate_lorenz(
    length: int = 5000,
    dt: float = 0.01,
    sample_stride: int = 5,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    warmup: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lorenz-63 3D chaotic attractor using RK4 integration.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Returns
    -------
    u : np.ndarray, shape (length, 3)
    y : np.ndarray, shape (length, 3)
        One-step-ahead prediction targets.
    """
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0], dtype=np.float64) + 0.1 * rng.uniform(-1.0, 1.0, 3)

    def lorenz_deriv(s: np.ndarray) -> np.ndarray:
        x, y, z = s[0], s[1], s[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz], dtype=np.float64)

    total_steps = (length + warmup + 1) * sample_stride
    trajectory = []

    for step in range(total_steps):
        if step % sample_stride == 0:
            trajectory.append(state.copy())

        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + 0.5 * dt * k1)
        k3 = lorenz_deriv(state + 0.5 * dt * k2)
        k4 = lorenz_deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    trajectory = np.array(trajectory, dtype=np.float64)[warmup:]
    u = trajectory[:-1]
    y = trajectory[1:]
    return u[:length], y[:length]
