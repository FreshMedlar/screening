"""
AERC-IP Hybrid: AdamW AERC-IP  +  Ridge-R (reservoir→logits)  +  Ridge-M (meta-readout).

Architecture:
  - inner: AERC (from aerc_ip.py) — trained with AdamW, exactly as in base/
  - ridge_readout_R: nn.Linear(N, V)    fitted by ridge regression from reservoir states
  - ridge_readout_M: nn.Linear(2V, V)   fitted by ridge regression from concat(logits_adamw, logits_R)

The two ridge layers have requires_grad=False at all times; they are NEVER touched by AdamW.
Ridge weights are populated after AdamW training via fit_ridge_R() and fit_ridge_M().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aerc_ip import AERC


class AERCHybrid(nn.Module):
    """
    Hybrid model: AERC-IP (AdamW) + independent Ridge readouts R and M.

    Parameters
    ----------
    vocab_size, d_e, N, H, spectral_radius:
        Forwarded to the inner AERC model (same defaults as base/).

    Usage
    -----
    1.  IP pre-training:   pretrain_reservoir_ip(model.inner, ...)
    2.  AdamW training:    loss = ce(model.forward_adamw(x), y)  →  loss.backward()
    3.  Fit Ridge-R:       model.fit_ridge_R(train_loader, ...)
    4.  Fit Ridge-M:       model.fit_ridge_M(train_loader, ...)
    5.  Full inference:    logits = model(idx)
    """

    def __init__(
        self,
        vocab_size: int,
        d_e: int = 16,
        N: int = 160,
        H: int = 30,
        spectral_radius: float = 0.95,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # --- Inner AERC-IP (all its trainable params go to AdamW) ---
        self.inner = AERC(
            vocab_size=vocab_size,
            d_e=d_e,
            N=N,
            H=H,
            spectral_radius=spectral_radius,
        )

        # --- Ridge readout R: raw reservoir states → logits ---
        self.ridge_readout_R = nn.Linear(N, vocab_size, bias=True)
        self.ridge_readout_R.weight.requires_grad_(False)
        self.ridge_readout_R.bias.requires_grad_(False)

        # --- Meta-readout M: concat(logits_adamw, logits_R) → logits ---
        self.ridge_readout_M = nn.Linear(2 * vocab_size, vocab_size, bias=True)
        self.ridge_readout_M.weight.requires_grad_(False)
        self.ridge_readout_M.bias.requires_grad_(False)

        # Track whether ridge layers have been fitted
        self._ridge_r_fitted = False
        self._ridge_m_fitted = False

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def count_adamw_parameters(self) -> int:
        """Parameters trained by AdamW (inner AERC-IP only)."""
        return sum(p.numel() for p in self.inner.parameters() if p.requires_grad)

    def adamw_parameters(self):
        """Iterator over AdamW-trainable parameters (inner AERC-IP only)."""
        return (p for p in self.inner.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_adamw(self, idx: torch.Tensor) -> torch.Tensor:
        """AdamW path only — used during backprop training."""
        return self.inner(idx=idx)

    def forward_ridge_r(self, states: torch.Tensor) -> torch.Tensor:
        """
        Ridge-R path: raw reservoir states → logits.

        Args:
            states: (B, T, N) reservoir state tensor.
        Returns:
            logits: (B, T, V)
        """
        orig = states.shape[:-1]
        flat = states.reshape(-1, self.inner.N)           # (B*T, N)
        out  = self.ridge_readout_R(flat)                 # (B*T, V)
        return out.reshape(orig + (self.vocab_size,))

    def forward(
        self,
        idx: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Full hybrid inference (requires both ridge readouts to be fitted).

        Steps:
          1. Compute reservoir states.
          2. AdamW path: inner AERC-IP forward.
          3. Ridge-R path: raw states → logits.
          4. Concatenate → Ridge-M → final logits.

        Args:
            idx:               (B, T) token indices.
            return_components: If True, also return (logits_adamw, logits_r).
        """
        if not (self._ridge_r_fitted and self._ridge_m_fitted):
            raise RuntimeError(
                "Ridge readouts have not been fitted yet. "
                "Call fit_ridge_R() and fit_ridge_M() first."
            )
        states = self.inner.compute_reservoir_states(idx)  # (B, T, N)

        logits_adamw = self.inner(states=states)            # (B, T, V)
        logits_r     = self.forward_ridge_r(states)         # (B, T, V)

        orig = logits_adamw.shape[:-1]
        V    = self.vocab_size

        concat = torch.cat(
            [logits_adamw.reshape(-1, V), logits_r.reshape(-1, V)], dim=-1
        )  # (B*T, 2V)

        logits_m = self.ridge_readout_M(concat)             # (B*T, V)
        logits_m = logits_m.reshape(orig + (V,))            # (B, T, V)

        if return_components:
            return logits_m, logits_adamw, logits_r
        return logits_m

    # ------------------------------------------------------------------
    # Ridge fitting
    # ------------------------------------------------------------------

    def fit_ridge_R(
        self,
        train_loader,
        device: str,
        alpha: float = 1.0,
        max_batches: int = 200,
    ) -> None:
        """
        Fit Ridge readout R from raw reservoir states to one-hot targets.

        Populates self.ridge_readout_R.weight and .bias.
        """
        from ridge_utils import collect_reservoir_and_targets, fit_ridge

        print(f"  Fitting Ridge-R (alpha={alpha}, max_batches={max_batches})...")
        X, Y = collect_reservoir_and_targets(
            self, train_loader, device, self.vocab_size, max_batches
        )
        print(f"    Data collected: X={tuple(X.shape)}, Y={tuple(Y.shape)}")

        W, b = fit_ridge(X, Y, alpha)
        with torch.no_grad():
            self.ridge_readout_R.weight.copy_(W)
            self.ridge_readout_R.bias.copy_(b)

        self._ridge_r_fitted = True
        print("  Ridge-R fitted.")

    def fit_ridge_M(
        self,
        train_loader,
        device: str,
        alpha: float = 1.0,
        max_batches: int = 200,
        use_bf16: bool = False,
    ) -> None:
        """
        Fit meta-readout M from concat(logits_adamw, logits_R) to one-hot targets.

        Requires Ridge-R to be fitted first.
        Populates self.ridge_readout_M.weight and .bias.
        """
        if not self._ridge_r_fitted:
            raise RuntimeError("Ridge-R must be fitted before Ridge-M.")

        from ridge_utils import collect_concat_logits_and_targets, fit_ridge

        print(f"  Fitting Ridge-M (alpha={alpha}, max_batches={max_batches})...")
        X, Y = collect_concat_logits_and_targets(
            self, train_loader, device, self.vocab_size, max_batches, use_bf16
        )
        print(f"    Data collected: X={tuple(X.shape)}, Y={tuple(Y.shape)}")

        W, b = fit_ridge(X, Y, alpha)
        with torch.no_grad():
            self.ridge_readout_M.weight.copy_(W)
            self.ridge_readout_M.bias.copy_(b)

        self._ridge_m_fitted = True
        print("  Ridge-M fitted.")
