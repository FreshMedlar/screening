"""
Ridge regression utilities for AERC-IP Hybrid.

Provides:
  - fit_ridge: closed-form normal-equations solver (float64 for stability).
  - collect_reservoir_and_targets: streams (states, one-hot targets) from a dataloader to CPU.
  - collect_concat_and_targets: streams (concat_logits, one-hot targets) from a dataloader to CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from aerc_ip_hybrid import AERCHybrid


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def fit_ridge(X: Tensor, Y: Tensor, alpha: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Closed-form ridge regression.

    Solves:  min ||X W - Y||^2_F  +  alpha * ||W||^2_F
    Solution: W = (X^T X + alpha I)^{-1} X^T Y

    Uses float64 arithmetic for numerical stability.
    Bias is handled by appending a column of ones to X.

    Args:
        X:     (M, d) feature matrix.
        Y:     (M, C) target matrix (e.g. one-hot).
        alpha: Ridge regularisation strength.

    Returns:
        weight: (C, d) weight matrix  (matches nn.Linear convention).
        bias:   (C,)  bias vector.
    """
    # Promote to float64 for numerical stability
    X64 = X.double()
    Y64 = Y.double()

    M, d = X64.shape

    # Augment with bias column
    ones = torch.ones(M, 1, dtype=torch.float64, device=X64.device)
    X_aug = torch.cat([X64, ones], dim=1)  # (M, d+1)

    # Normal equations: (X^T X + alpha I) W_aug = X^T Y
    # Note: we do NOT regularise the bias term (last row/col)
    reg = alpha * torch.eye(d + 1, dtype=torch.float64, device=X64.device)
    reg[-1, -1] = 0.0  # no regularisation on bias

    A = X_aug.t().mm(X_aug) + reg           # (d+1, d+1)
    B = X_aug.t().mm(Y64)                   # (d+1, C)

    W_aug = torch.linalg.solve(A, B)         # (d+1, C)

    weight = W_aug[:-1, :].t().float()       # (C, d)
    bias   = W_aug[-1,  :].float()           # (C,)

    return weight, bias


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_reservoir_and_targets(
    model: "AERCHybrid",
    dataloader: "DataLoader",
    device: str,
    vocab_size: int,
    max_batches: int = 200,
) -> tuple[Tensor, Tensor]:
    """
    Collect flattened reservoir states and one-hot targets for ridge fitting.

    Returns:
        states_cpu:  (M_total, N)  float32 CPU tensor.
        targets_cpu: (M_total, V)  float32 CPU one-hot tensor.
    """
    model.eval()
    all_states  = []
    all_targets = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        # (B, T, N)  →  (B*T, N)
        states = model.inner.compute_reservoir_states(x)
        states_flat = states.reshape(-1, model.inner.N).cpu()   # (B*T, N)

        # one-hot targets: (B*T, V)
        y_flat  = y.reshape(-1)                                  # (B*T,)
        y_oh    = F.one_hot(y_flat, num_classes=vocab_size).float().cpu()

        all_states.append(states_flat)
        all_targets.append(y_oh)

    model.train()
    return torch.cat(all_states, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def collect_concat_logits_and_targets(
    model: "AERCHybrid",
    dataloader: "DataLoader",
    device: str,
    vocab_size: int,
    max_batches: int = 200,
    use_bf16: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Collect concatenated [logits_adamw | logits_ridge_R] and one-hot targets
    for fitting the meta-readout M.

    Returns:
        concat_cpu:  (M_total, 2V)  float32 CPU tensor.
        targets_cpu: (M_total,  V)  float32 CPU one-hot tensor.
    """
    model.eval()
    all_concat  = []
    all_targets = []

    autocast_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
    )

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        with autocast_ctx:
            # AdamW path
            states = model.inner.compute_reservoir_states(x)      # (B, T, N)
            logits_adamw = model.inner(states=states)              # (B, T, V)

            # Ridge-R path
            logits_ridge = model.forward_ridge_r(states)          # (B, T, V)

        # Cast to float32 before moving to CPU
        logits_adamw = logits_adamw.float()
        logits_ridge = logits_ridge.float()

        B, T, V = logits_adamw.shape
        concat = torch.cat(
            [logits_adamw.reshape(-1, V), logits_ridge.reshape(-1, V)], dim=-1
        ).cpu()                                                   # (B*T, 2V)

        y_flat = y.reshape(-1)
        y_oh   = F.one_hot(y_flat, num_classes=vocab_size).float().cpu()

        all_concat.append(concat)
        all_targets.append(y_oh)

    model.train()
    return torch.cat(all_concat, dim=0), torch.cat(all_targets, dim=0)
