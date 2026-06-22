#!/usr/bin/env python3
"""
Benchmark script comparing Classic Reservoir (Echo State Network) and
Attention-Enhanced Reservoir Computing (AERC) on:
- NARMA-10, NARMA-20, NARMA-30
- Information Memory Capacity (MC)

Written for Projects/screening/att_res/ folder.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress PyTorch warnings
import warnings
import logging
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Import base AERC and helper from local directory
from aerc_arch import AERC, _leaky_reservoir_scan, _leaky_reservoir_scan_fb, _init_reservoir


# ---------------------------------------------------------------------------
# Benchmark Data Generators
# ---------------------------------------------------------------------------

def generate_narma(k: int, length: int, seed: int = 42):
    """
    Generates NARMA-k time series.
    Input u(t) is drawn from a uniform distribution U[0, 0.5].
    """
    np.random.seed(seed)
    u = np.random.uniform(0.0, 0.5, length)
    y = np.zeros(length)
    
    # Choose stable parameters based on order
    if k == 10:
        alpha, beta, gamma, delta = 0.3, 0.05, 1.5, 0.1
    elif k == 20:
        alpha, beta, gamma, delta = 0.3, 0.025, 1.5, 0.1
    elif k == 30:
        alpha, beta, gamma, delta = 0.2, 0.04, 1.5, 0.001
    else:
        alpha, beta, gamma, delta = 0.3, 0.05 * (10 / k), 1.5, 0.1
    
    # Simulate difference equation
    for t in range(k, length - 1):
        # sum of y(t), y(t-1), ..., y(t-k+1)
        y_sum = np.sum(y[t - k + 1 : t + 1])
        y[t + 1] = alpha * y[t] + beta * y[t] * y_sum + gamma * u[t - k + 1] * u[t] + delta
        
    return u.reshape(-1, 1), y.reshape(-1, 1)


def generate_mc(length: int, max_delay: int = 100, seed: int = 42):
    """
    Generates data for the Information Memory Capacity task.
    Input u(t) is drawn from U[-1, 1].
    Target at delay d is u(t - d).
    """
    np.random.seed(seed)
    u = np.random.uniform(-1.0, 1.0, length)
    y = np.zeros((length, max_delay))
    for d in range(1, max_delay + 1):
        y[d:, d - 1] = u[:-d]
    return u.reshape(-1, 1), y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Normalized Root Mean Square Error."""
    mse = np.mean((y_true - y_pred) ** 2)
    var_true = np.var(y_true)
    if var_true < 1e-9:
        return float(np.sqrt(mse))
    return float(np.sqrt(mse / var_true))


def compute_memory_capacity(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes Memory Capacity (MC) score as sum of squared Pearson correlations.
    y_true, y_pred: shape (N_samples, max_delay)
    """
    max_delay = y_true.shape[1]
    mc_values = []
    for d in range(max_delay):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        
        # Pearson correlation
        cov = np.cov(yt, yp)
        var_t = cov[0, 0]
        var_p = cov[1, 1]
        cov_tp = cov[0, 1]
        
        if var_t > 1e-9 and var_p > 1e-9:
            r2 = (cov_tp ** 2) / (var_t * var_p)
            mc_values.append(r2)
        else:
            mc_values.append(0.0)
            
    return float(np.sum(mc_values)), mc_values


def compute_vpt(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.4) -> int:
    """
    Valid Prediction Time (VPT) - computed as the first step where normalized absolute error
    crosses the threshold.
    """
    error = np.abs(y_true - y_pred)
    std_true = np.std(y_true)
    if std_true > 1e-9:
        norm_err = error / std_true
    else:
        norm_err = error
        
    for t, val in enumerate(norm_err):
        if np.any(val > threshold):
            return t
    return len(y_true)


# ---------------------------------------------------------------------------
# Continuous Model Wrappers
# ---------------------------------------------------------------------------

class ContinuousReservoir(nn.Module):
    """
    Classic Echo State Network for continuous-valued inputs.
    Trained analytically via Ridge Regression.
    """
    def __init__(self, input_dim: int, output_dim: int, d_e: int = 32, N: int = 300,
                 spectral_radius: float = 0.95, fb_scaling: float = 0.0, leaking_rate: float = 1.0):
        super().__init__()
        self.N = N
        self.d_e = d_e
        self.leaking_rate = leaking_rate
        
        # Fixed random input projection (analogous to embedding)
        self.W_in = nn.Parameter(torch.randn(d_e, input_dim), requires_grad=False)
        with torch.no_grad():
            self.W_in.normal_(0.0, 0.1 / math.sqrt(input_dim))
            
        # Fixed RNN reservoir
        self.rnn = nn.RNN(
            input_size=d_e,
            hidden_size=N,
            batch_first=True,
            bias=True,
            nonlinearity="tanh"
        )
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.weight_hh_l0.requires_grad = False
        self.rnn.bias_ih_l0.requires_grad = False
        self.rnn.bias_hh_l0.requires_grad = False
        
        with torch.no_grad():
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
            
        _init_reservoir(self.rnn, spectral_radius)
        
        # Optional output-state feedback
        if fb_scaling > 0.0:
            W_fb_raw = 2.0 * torch.rand(N, N) - 1.0
            with torch.no_grad():
                sr_curr = torch.max(torch.abs(torch.linalg.eigvals(W_fb_raw))).item()
                if sr_curr > 0:
                    W_fb_raw = W_fb_raw / sr_curr * fb_scaling
            self.register_buffer("W_fb", W_fb_raw)
        else:
            self.register_buffer("W_fb", None)
            
        # Readout layer (trainable)
        self.readout = nn.Linear(N, output_dim)
        
    def compute_states(self, u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = F.linear(u, self.W_in)
            
            if self.W_fb is None and self.leaking_rate == 1.0:
                out, _ = self.rnn(x)
                return out
                
            B, T, _ = x.shape
            h0 = torch.zeros(B, self.N, dtype=x.dtype, device=x.device)
            
            weight_ih = self.rnn.weight_ih_l0
            weight_hh = self.rnn.weight_hh_l0
            bias_ih   = self.rnn.bias_ih_l0
            bias_hh   = self.rnn.bias_hh_l0
            
            if self.W_fb is None:
                return _leaky_reservoir_scan(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.leaking_rate
                )
            else:
                return _leaky_reservoir_scan_fb(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.leaking_rate,
                    self.W_fb,
                )
                
    def forward(self, u: torch.Tensor, states: torch.Tensor = None) -> torch.Tensor:
        if states is None:
            states = self.compute_states(u)
        return self.readout(states)
        
    def fit_ridge(self, u: torch.Tensor, y: torch.Tensor, reg: float = 1e-3, washout: int = 200):
        """Fit linear readout weights analytically using Ridge Regression."""
        self.eval()
        with torch.no_grad():
            states = self.compute_states(u) # (B, T, N)
            
            # Slice off washout steps
            states_wash = states[:, washout:, :].reshape(-1, self.N) # (B*(T-washout), N)
            targets_wash = y[:, washout:, :].reshape(-1, y.shape[-1]) # (B*(T-washout), output_dim)
            
            # Append column of ones to fit bias
            ones = torch.ones(states_wash.shape[0], 1, device=states.device, dtype=states.dtype)
            X = torch.cat([states_wash, ones], dim=1) # (B*(T-washout), N+1)
            
            # Solve (X^T * X + reg * I) * W = X^T * Y
            XTX = torch.matmul(X.T, X)
            XTX += reg * torch.eye(self.N + 1, device=states.device, dtype=states.dtype)
            XTY = torch.matmul(X.T, targets_wash)
            W_all = torch.linalg.solve(XTX, XTY).T # (output_dim, N+1)
            
            # Copy fitted weights & bias to readout linear layer
            self.readout.weight.copy_(W_all[:, :self.N])
            self.readout.bias.copy_(W_all[:, self.N])


class ContinuousAERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) for continuous-valued inputs.
    Trained via Backpropagation (AdamW).
    """
    def __init__(self, input_dim: int, output_dim: int, d_e: int = 32, N: int = 300, H: int = 60,
                 spectral_radius: float = 0.95, fb_scaling: float = 0.0, leaking_rate: float = 1.0,
                 dropout: float = 0.0, activation: str = "silu"):
        super().__init__()
        # Set vocab_size to output_dim so final readout maps attention output H -> output_dim
        self.aerc = AERC(
            vocab_size=output_dim,
            d_e=d_e,
            N=N,
            H=H,
            spectral_radius=spectral_radius,
            fb_scaling=fb_scaling,
            dropout=dropout,
            leaking_rate=leaking_rate,
            activation=activation,
        )
        
        # Fixed random input projection mapping input_dim to d_e
        self.W_in = nn.Parameter(torch.randn(d_e, input_dim), requires_grad=False)
        with torch.no_grad():
            self.W_in.normal_(0.0, 0.1 / math.sqrt(input_dim))
            
    def compute_states(self, u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = F.linear(u, self.W_in)
            
            if self.aerc.W_fb is None and self.aerc.leaking_rate == 1.0:
                out, _ = self.aerc.rnn(x)
                return out
                
            B, T, _ = x.shape
            h0 = torch.zeros(B, self.aerc.N, dtype=x.dtype, device=x.device)
            
            weight_ih = self.aerc.rnn.weight_ih_l0
            weight_hh = self.aerc.rnn.weight_hh_l0
            bias_ih   = self.aerc.rnn.bias_ih_l0
            bias_hh   = self.aerc.rnn.bias_hh_l0
            
            if self.aerc.W_fb is None:
                return _leaky_reservoir_scan(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.aerc.leaking_rate
                )
            else:
                return _leaky_reservoir_scan_fb(
                    x, h0, weight_ih, weight_hh, bias_ih, bias_hh, self.aerc.leaking_rate,
                    self.aerc.W_fb,
                )
                
    def forward(self, u: torch.Tensor, states: torch.Tensor = None) -> torch.Tensor:
        if states is None:
            states = self.compute_states(u)
        return self.aerc(states=states)


# ---------------------------------------------------------------------------
# Training Logic for AERC
# ---------------------------------------------------------------------------

def train_aerc(model: ContinuousAERC, u: torch.Tensor, y: torch.Tensor,
               epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-4,
               washout: int = 200):
    """
    Trains the Attention layer and Readout parameters of AERC using AdamW.
    Precomputes reservoir states to make training extremely fast.
    """
    model.train()
    
    # Compute states once
    with torch.no_grad():
        states = model.compute_states(u).detach() # (B, T, N)
        
    # Slice off washout steps
    states_train = states[:, washout:, :]
    y_train = y[:, washout:, :]
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        # Forward pass using precomputed states
        pred = model(u=None, states=states_train)
        loss = F.mse_loss(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    return model


def fit_static_head_continuous(
    model: "ContinuousAERC",
    u: torch.Tensor,
    y: torch.Tensor,
    reg: float = 1e-3,
    washout: int = 200,
) -> None:
    """
    Fits ContinuousAERC.aerc.static_head via ridge regression on continuous targets,
    then freezes it.
    """
    model.eval()
    aerc = model.aerc
    N, V = aerc.N, aerc.vocab_size

    with torch.no_grad():
        states      = model.compute_states(u)
        states_wash = states[:, washout:, :].reshape(-1, N)
        states_norm = aerc.state_norm(states_wash)
        y_wash      = y[:, washout:, :].reshape(-1, V)

        ones = torch.ones(states_norm.shape[0], 1,
                          device=states_norm.device, dtype=states_norm.dtype)
        X   = torch.cat([states_norm, ones], dim=-1)
        XtX = X.T @ X
        XtX.diagonal().add_(reg)
        XtY = X.T @ y_wash
        W   = torch.linalg.solve(XtX, XtY)

        aerc.static_head.weight.copy_(W[:N, :].T)
        aerc.static_head.bias.copy_(W[N, :])

    # Freeze static head and make sure attention components are trainable
    aerc.set_phase(2)
    model.train()


def train_aerc_ridge(
    model: "ContinuousAERC",
    u: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    reg: float = 1e-3,
    washout: int = 200,
) -> "ContinuousAERC":
    """
    Two-phase AERC training:
      Phase 1 — fit static_head analytically via ridge regression (no SGD).
      Phase 2 — freeze static_head; train attention network via AdamW.
    """
    fit_static_head_continuous(model, u, y, reg=reg, washout=washout)
    return train_aerc(model, u, y, epochs=epochs, lr=lr,
                      weight_decay=weight_decay, washout=washout)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reservoir Computing NARMA and MC Benchmarks")
    parser.add_argument("--length", type=int, default=5000, help="Total sequence length")
    parser.add_argument("--washout", type=int, default=200, help="Washout steps")
    parser.add_argument("--N", type=int, default=300, help="Reservoir size (N)")
    parser.add_argument("--H", type=int, default=60, help="Attention hidden dimension (H)")
    parser.add_argument("--d_e", type=int, default=32, help="Embedding dimension (d_e)")
    parser.add_argument("--leaking_rate", type=float, default=1.0, help="Reservoir leaking rate")
    parser.add_argument("--spectral_radius", type=float, default=0.95, help="Reservoir spectral radius")
    parser.add_argument("--fb_scaling", type=float, default=0.0, help="Feedback scaling coefficient")
    parser.add_argument("--reg", type=float, default=1e-3, help="Ridge regression regularization")
    parser.add_argument("--lr", type=float, default=1e-3, help="AERC learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="AERC training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_only", action="store_true", help="Run a quick verification test")
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "tanh", "relu"],
        help="Attention activation function (uses H dimension)",
    )
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.test_only:
        print(">>> Fast verification test requested. Overriding configuration parameters.")
        args.length = 1000
        args.epochs = 20
        args.washout = 50
        args.N = 100
        args.H = 20
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Device: {device}")
    print(f"Parameters: N={args.N}, H={args.H}, d_e={args.d_e}, LR={args.leaking_rate}, SR={args.spectral_radius}")
    _gate_params = (args.N + 1) * args.H
    print(f"Activation: {args.activation}  (net_gate params ≈ {_gate_params:,})")
    print("=" * 80)
    
    results = {}
    mc_profiles = {}
    
    # -----------------------------------------------------------------------
    # Task 1: NARMA Benchmarks (10, 20, 30)
    # -----------------------------------------------------------------------
    for order in [10, 20, 30]:
        print(f"\n--- Running NARMA-{order} ---")
        u, y = generate_narma(order, args.length, seed=args.seed)
        
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0) # (1, length, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0) # (1, length, 1)
        
        train_len = int(args.length * 0.8)
        u_train = u_tensor[:, :train_len, :]
        y_train = y_tensor[:, :train_len, :]
        
        # A. Train Classic Reservoir (Ridge Regression)
        model_classic = ContinuousReservoir(
            input_dim=1,
            output_dim=1,
            d_e=args.d_e,
            N=args.N,
            spectral_radius=args.spectral_radius,
            fb_scaling=args.fb_scaling,
            leaking_rate=args.leaking_rate
        ).to(device)
        
        t0 = time.time()
        model_classic.fit_ridge(u_train, y_train, reg=args.reg, washout=args.washout)
        classic_time = time.time() - t0
        
        # B. Train AERC (AdamW, end-to-end)
        model_aerc = ContinuousAERC(
            input_dim=1,
            output_dim=1,
            d_e=args.d_e,
            N=args.N,
            H=args.H,
            spectral_radius=args.spectral_radius,
            fb_scaling=args.fb_scaling,
            leaking_rate=args.leaking_rate,
            activation=args.activation,
        ).to(device)
        
        t0 = time.time()
        train_aerc(model_aerc, u_train, y_train, epochs=args.epochs, lr=args.lr, washout=args.washout)
        aerc_time = time.time() - t0

        # C. Train AERC with Ridge prior (two-phase)
        model_aerc_ridge = ContinuousAERC(
            input_dim=1,
            output_dim=1,
            d_e=args.d_e,
            N=args.N,
            H=args.H,
            spectral_radius=args.spectral_radius,
            fb_scaling=args.fb_scaling,
            leaking_rate=args.leaking_rate,
            activation=args.activation,
        ).to(device)

        t0 = time.time()
        train_aerc_ridge(model_aerc_ridge, u_train, y_train,
                         epochs=args.epochs, lr=args.lr,
                         reg=args.reg, washout=args.washout)
        aerc_ridge_time = time.time() - t0
        
        # Evaluate all three models on test split
        with torch.no_grad():
            states_classic = model_classic.compute_states(u_tensor)
            pred_classic = model_classic(u_tensor, states=states_classic)
            pred_classic_np = pred_classic[:, train_len:, :].cpu().numpy().squeeze()
            
            states_aerc = model_aerc.compute_states(u_tensor)
            pred_aerc = model_aerc(u_tensor, states=states_aerc)
            pred_aerc_np = pred_aerc[:, train_len:, :].cpu().numpy().squeeze()

            states_aerc_ridge = model_aerc_ridge.compute_states(u_tensor)
            pred_aerc_ridge = model_aerc_ridge(u_tensor, states=states_aerc_ridge)
            pred_aerc_ridge_np = pred_aerc_ridge[:, train_len:, :].cpu().numpy().squeeze()
            
            y_test_np = y_tensor[:, train_len:, :].cpu().numpy().squeeze()
            
        # Compute metrics
        nrmse_classic    = compute_nrmse(y_test_np, pred_classic_np)
        nrmse_aerc       = compute_nrmse(y_test_np, pred_aerc_np)
        nrmse_aerc_ridge = compute_nrmse(y_test_np, pred_aerc_ridge_np)
        vpt_classic      = compute_vpt(y_test_np, pred_classic_np)
        vpt_aerc         = compute_vpt(y_test_np, pred_aerc_np)
        vpt_aerc_ridge   = compute_vpt(y_test_np, pred_aerc_ridge_np)
        
        results[f"NARMA-{order}"] = {
            "classic_nrmse":    nrmse_classic,
            "classic_vpt":      vpt_classic,
            "classic_time":     classic_time,
            "aerc_nrmse":       nrmse_aerc,
            "aerc_vpt":         vpt_aerc,
            "aerc_time":        aerc_time,
            "aerc_ridge_nrmse": nrmse_aerc_ridge,
            "aerc_ridge_vpt":   vpt_aerc_ridge,
            "aerc_ridge_time":  aerc_ridge_time,
            "y_test":           y_test_np,
            "pred_classic":     pred_classic_np,
            "pred_aerc":        pred_aerc_np,
            "pred_aerc_ridge":  pred_aerc_ridge_np,
        }
        
        print(f"  Classic Reservoir (Ridge) | NRMSE: {nrmse_classic:.4f} | VPT: {vpt_classic:4d} | Fit Time: {classic_time:.3f}s")
        print(f"  AERC (AdamW)              | NRMSE: {nrmse_aerc:.4f} | VPT: {vpt_aerc:4d} | Train Time: {aerc_time:.3f}s")
        print(f"  AERC + Ridge prior        | NRMSE: {nrmse_aerc_ridge:.4f} | VPT: {vpt_aerc_ridge:4d} | Train Time: {aerc_ridge_time:.3f}s")

    # -----------------------------------------------------------------------
    # Task 2: Information Memory Capacity (MC)
    # -----------------------------------------------------------------------
    print(f"\n--- Running Memory Capacity (MC) ---")
    max_delay = 100
    u, y = generate_mc(args.length, max_delay=max_delay, seed=args.seed)
    
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    
    train_len = int(args.length * 0.8)
    u_train = u_tensor[:, :train_len, :]
    y_train = y_tensor[:, :train_len, :]
    
    # A. Train Classic Reservoir (Ridge Regression)
    model_classic = ContinuousReservoir(
        input_dim=1,
        output_dim=max_delay,
        d_e=args.d_e,
        N=args.N,
        spectral_radius=args.spectral_radius,
        fb_scaling=args.fb_scaling,
        leaking_rate=args.leaking_rate
    ).to(device)
    
    t0 = time.time()
    model_classic.fit_ridge(u_train, y_train, reg=args.reg, washout=args.washout)
    classic_time = time.time() - t0
    
    # B. Train AERC (AdamW, end-to-end)
    model_aerc = ContinuousAERC(
        input_dim=1,
        output_dim=max_delay,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        fb_scaling=args.fb_scaling,
        leaking_rate=args.leaking_rate,
        activation=args.activation,
    ).to(device)
    
    t0 = time.time()
    train_aerc(model_aerc, u_train, y_train, epochs=args.epochs, lr=args.lr, washout=args.washout)
    aerc_time = time.time() - t0

    # C. Train AERC with Ridge prior (two-phase)
    model_aerc_ridge = ContinuousAERC(
        input_dim=1,
        output_dim=max_delay,
        d_e=args.d_e,
        N=args.N,
        H=args.H,
        spectral_radius=args.spectral_radius,
        fb_scaling=args.fb_scaling,
        leaking_rate=args.leaking_rate,
        activation=args.activation,
    ).to(device)

    t0 = time.time()
    train_aerc_ridge(model_aerc_ridge, u_train, y_train,
                     epochs=args.epochs, lr=args.lr,
                     reg=args.reg, washout=args.washout)
    aerc_ridge_time = time.time() - t0
    
    # Evaluate
    with torch.no_grad():
        states_classic = model_classic.compute_states(u_tensor)
        pred_classic = model_classic(u_tensor, states=states_classic)
        pred_classic_np = pred_classic[:, train_len:, :].cpu().numpy().squeeze(0)
        
        states_aerc = model_aerc.compute_states(u_tensor)
        pred_aerc = model_aerc(u_tensor, states=states_aerc)
        pred_aerc_np = pred_aerc[:, train_len:, :].cpu().numpy().squeeze(0)

        states_aerc_ridge = model_aerc_ridge.compute_states(u_tensor)
        pred_aerc_ridge = model_aerc_ridge(u_tensor, states=states_aerc_ridge)
        pred_aerc_ridge_np = pred_aerc_ridge[:, train_len:, :].cpu().numpy().squeeze(0)
        
        y_test_np = y_tensor[:, train_len:, :].cpu().numpy().squeeze(0)
        
    mc_classic_total,    mc_classic_profile    = compute_memory_capacity(y_test_np, pred_classic_np)
    mc_aerc_total,       mc_aerc_profile       = compute_memory_capacity(y_test_np, pred_aerc_np)
    mc_aerc_ridge_total, mc_aerc_ridge_profile = compute_memory_capacity(y_test_np, pred_aerc_ridge_np)
    
    results["MC"] = {
        "classic_mc":      mc_classic_total,
        "classic_time":    classic_time,
        "aerc_mc":         mc_aerc_total,
        "aerc_time":       aerc_time,
        "aerc_ridge_mc":   mc_aerc_ridge_total,
        "aerc_ridge_time": aerc_ridge_time,
    }
    
    mc_profiles["classic"]    = mc_classic_profile
    mc_profiles["aerc"]       = mc_aerc_profile
    mc_profiles["aerc_ridge"] = mc_aerc_ridge_profile
    
    print(f"  Classic Reservoir (Ridge) | Memory Capacity: {mc_classic_total:.4f} | Fit Time: {classic_time:.3f}s")
    print(f"  AERC (AdamW)              | Memory Capacity: {mc_aerc_total:.4f} | Train Time: {aerc_time:.3f}s")
    print(f"  AERC + Ridge prior        | Memory Capacity: {mc_aerc_ridge_total:.4f} | Train Time: {aerc_ridge_time:.3f}s")
    
    # Summary Table
    W = 108
    print("\n" + "=" * W)
    print(f"{'BENCHMARK SUMMARY':^{W}}")
    print("=" * W)
    print(f"{'Task':<15} | {'Classic (Ridge)':<16} | {'AERC (AdamW)':<14} | {'AERC+Ridge':<12} | {'Cls t':<7} | {'AERC t':<7} | {'A+R t':<6}")
    print("-" * W)
    for task in ["NARMA-10", "NARMA-20", "NARMA-30"]:
        c_n  = results[task]["classic_nrmse"]
        a_n  = results[task]["aerc_nrmse"]
        ar_n = results[task]["aerc_ridge_nrmse"]
        c_t  = results[task]["classic_time"]
        a_t  = results[task]["aerc_time"]
        ar_t = results[task]["aerc_ridge_time"]
        print(f"{task:<15} | {c_n:<16.4f} | {a_n:<14.4f} | {ar_n:<12.4f} | {c_t:<6.2f}s | {a_t:<6.2f}s | {ar_t:<5.2f}s")
        
    c_mc  = results["MC"]["classic_mc"]
    a_mc  = results["MC"]["aerc_mc"]
    ar_mc = results["MC"]["aerc_ridge_mc"]
    c_t   = results["MC"]["classic_time"]
    a_t   = results["MC"]["aerc_time"]
    ar_t  = results["MC"]["aerc_ridge_time"]
    print(f"{'Memory Capacity':<15} | {c_mc:<16.4f} | {a_mc:<14.4f} | {ar_mc:<12.4f} | {c_t:<6.2f}s | {a_t:<6.2f}s | {ar_t:<5.2f}s")
    print("=" * W)
    
    # Plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Subplot 1
        ax = axes[0, 0]
        y_test_10 = results["NARMA-10"]["y_test"][:150]
        p_c_10  = results["NARMA-10"]["pred_classic"][:150]
        p_a_10  = results["NARMA-10"]["pred_aerc"][:150]
        p_ar_10 = results["NARMA-10"]["pred_aerc_ridge"][:150]
        ax.plot(y_test_10, label="True Target", color="black", linewidth=2.0)
        ax.plot(p_c_10,  "--", label=f"Classic    (NRMSE: {results['NARMA-10']['classic_nrmse']:.4f})",    color="C0", alpha=0.8)
        ax.plot(p_a_10,  ":",  label=f"AERC       (NRMSE: {results['NARMA-10']['aerc_nrmse']:.4f})",       color="C1", alpha=0.8)
        ax.plot(p_ar_10, "-.", label=f"AERC+Ridge (NRMSE: {results['NARMA-10']['aerc_ridge_nrmse']:.4f})", color="C2", alpha=0.9)
        ax.set_title("NARMA-10 Time Series Prediction (Test Split)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("y(t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Subplot 2
        ax = axes[0, 1]
        y_test_30 = results["NARMA-30"]["y_test"][:150]
        p_c_30  = results["NARMA-30"]["pred_classic"][:150]
        p_a_30  = results["NARMA-30"]["pred_aerc"][:150]
        p_ar_30 = results["NARMA-30"]["pred_aerc_ridge"][:150]
        ax.plot(y_test_30, label="True Target", color="black", linewidth=2.0)
        ax.plot(p_c_30,  "--", label=f"Classic    (NRMSE: {results['NARMA-30']['classic_nrmse']:.4f})",    color="C0", alpha=0.8)
        ax.plot(p_a_30,  ":",  label=f"AERC       (NRMSE: {results['NARMA-30']['aerc_nrmse']:.4f})",       color="C1", alpha=0.8)
        ax.plot(p_ar_30, "-.", label=f"AERC+Ridge (NRMSE: {results['NARMA-30']['aerc_ridge_nrmse']:.4f})", color="C2", alpha=0.9)
        ax.set_title("NARMA-30 Time Series Prediction (Test Split)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("y(t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Subplot 3
        ax = axes[1, 0]
        delays = np.arange(1, max_delay + 1)
        ax.plot(delays, mc_profiles["classic"],    "o-", label=f"Classic    (Total MC: {c_mc:.2f})",    markersize=3, color="C0")
        ax.plot(delays, mc_profiles["aerc"],       "s-", label=f"AERC       (Total MC: {a_mc:.2f})",    markersize=3, color="C1")
        ax.plot(delays, mc_profiles["aerc_ridge"], "^-", label=f"AERC+Ridge (Total MC: {ar_mc:.2f})",   markersize=3, color="C2")
        ax.set_title("Information Memory Capacity Profile")
        ax.set_xlabel("Delay (d)")
        ax.set_ylabel("Correlation Squared (r²)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Subplot 4
        ax = axes[1, 1]
        ax.axis("off")
        table_content = [
            ["Metric",          "Classic (Ridge)",                                   "AERC (AdamW)",                                  "AERC + Ridge"],
            ["NARMA-10 NRMSE",  f"{results['NARMA-10']['classic_nrmse']:.4f}",      f"{results['NARMA-10']['aerc_nrmse']:.4f}",      f"{results['NARMA-10']['aerc_ridge_nrmse']:.4f}"],
            ["NARMA-20 NRMSE",  f"{results['NARMA-20']['classic_nrmse']:.4f}",      f"{results['NARMA-20']['aerc_nrmse']:.4f}",      f"{results['NARMA-20']['aerc_ridge_nrmse']:.4f}"],
            ["NARMA-30 NRMSE",  f"{results['NARMA-30']['classic_nrmse']:.4f}",      f"{results['NARMA-30']['aerc_nrmse']:.4f}",      f"{results['NARMA-30']['aerc_ridge_nrmse']:.4f}"],
            ["Memory Capacity", f"{c_mc:.4f}",                                       f"{a_mc:.4f}",                                   f"{ar_mc:.4f}"],
        ]
        table = ax.table(cellText=table_content, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)
        
        plt.suptitle("Reservoir Computing Benchmarks: Classic vs AERC vs AERC+Ridge", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        out_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "benchmark_comparison.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n✓ Benchmark comparison plot saved to {out_path}")
        
    except ImportError:
        print("\n⚠ matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
