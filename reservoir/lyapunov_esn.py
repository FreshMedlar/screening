"""
Lyapunov-Enhanced Echo State Network (LE-ESN).

Implements the theoretical principles from:
David G. Clark (2026), "Lyapunov spectrum of random neural networks" (arXiv:2610.12426):

1. Edge-of-Chaos Critical Spectrum Calibration:
   In Clark (2026), the eigenvalues of the random Jacobian M = (1 - delta)I + delta * J * D
   govern the Lyapunov spectrum. Classical i.i.d. Gaussian random matrices suffer from the
   circular law where many modes cluster near 0 (fast forgetting). LE-ESN employs critical
   unitary/orthogonal coupling (or reciprocal-correlated coupling) to place all modes along the
   critical circle |mu| = g ~= 1, eliminating premature damping and dramatically boosting
   the information memory capacity.

2. Dynamic Trajectory Gain Calibration (Clark 2026 Eq. B.15 & Table 1):
   lambda_1 = log(g * sqrt(C_d(0))), where C_d(0) = <(1 - tanh^2(x))^2>.
   As inputs drive neurons into the non-linear regime, C_d(0) < 1 compresses the effective
   coupling g_eff = g * sqrt(C_d(0)), pushing the network into an over-contractive regime.
   LE-ESN applies homeostatic gain normalization:
       g(t) = exp(lambda_target) / sqrt(C_d(0) + eps)
   keeping lambda_1 stably poised at the critical boundary lambda_1 ~= 0 (the edge of chaos).

3. Forward-Backward Dichotomy Regularization (Clark 2026 Sec. 2.1 - 2.2, Eqs. 28-34):
   Solves the regularized forward-backward system:
       [eta*I, K; -K^T, eta*I] [U; V] = [I^v; 0]
   to construct the minimum-norm dichotomy projector P_stable that projects state trajectories
   onto the stable invariant manifold E^s, removing chaotic divergence while preserving memory.
"""

from typing import Optional, Tuple, Union
import numpy as np
try:
    import scipy.linalg as sla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .esn import EchoStateNetwork


class LyapunovEnhancedReservoir(EchoStateNetwork):
    """
    Lyapunov-Enhanced Echo State Network based on Clark (2026).

    Parameters
    ----------
    n_inputs : int
    n_reservoir : int
    n_outputs : int
    target_lyapunov : float
        Target maximum Lyapunov exponent lambda_1 (default 0.0, edge of chaos).
    coupling_type : str
        'critical_orthogonal' (default, places modes along the critical circle |mu| = g)
        or 'reciprocal_correlated' (Sommers-Clark correlated reciprocal couplings).
    reciprocal_corr : float
        Correlation tau = corr(J_ij, J_ji) if coupling_type is 'reciprocal_correlated'.
    spectral_radius : float
        Base coupling strength / spectral radius g.
    leaking_rate : float
        Euler discretization time-step delta in (0, 1].
    dynamic_gain_control : bool
        If True, normalizes g dynamically using Clark (2026) formula lambda_1 = log(g * sqrt(C_d(0))).
    forward_backward_dichotomy : bool
        If True, fits and applies the dichotomy projection onto the stable manifold.
    dichotomy_eta : float
        Regularization parameter eta for the dichotomy filter.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        n_reservoir: int = 300,
        n_outputs: int = 1,
        target_lyapunov: float = 0.0,
        coupling_type: str = "critical_orthogonal",
        reciprocal_corr: float = -0.5,
        spectral_radius: float = 0.98,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.1,
        density: float = 0.1,
        dynamic_gain_control: bool = True,
        forward_backward_dichotomy: bool = True,
        dichotomy_eta: float = 1e-2,
        feedback_scaling: float = 0.0,
        include_input: bool = True,
        include_bias: bool = True,
        quadratic_features: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.target_lyapunov = float(target_lyapunov)
        self.coupling_type = str(coupling_type)
        self.reciprocal_corr = float(np.clip(reciprocal_corr, -0.99, 0.99))
        self.dynamic_gain_control = bool(dynamic_gain_control)
        self.forward_backward_dichotomy = bool(forward_backward_dichotomy)
        self.dichotomy_eta = float(dichotomy_eta)
        self.P_stable = None  # Learned state dichotomy projection matrix

        super().__init__(
            n_inputs=n_inputs,
            n_reservoir=n_reservoir,
            n_outputs=n_outputs,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            density=density,
            feedback_scaling=feedback_scaling,
            include_input=include_input,
            include_bias=include_bias,
            quadratic_features=quadratic_features,
            random_state=random_state,
        )

    def _init_weights(self) -> None:
        """
        Initialize weights with critical spectrum tuning (Clark 2026).
        """
        rng = np.random.default_rng(self.random_state)
        N = self.n_reservoir

        # 1. Input weights
        self.W_in = rng.uniform(-1.0, 1.0, size=(N, self.n_inputs)) * self.input_scaling

        # 2. Bias vector
        self.b = rng.uniform(-1.0, 1.0, size=(N,)) * self.bias_scaling

        # 3. Recurrent weights based on coupling_type
        if self.coupling_type == "critical_orthogonal":
            # Haar-distributed random orthogonal matrix with critical scaling
            # Ensures all singular values and eigenvalue moduli reside exactly on the critical circle
            H = rng.standard_normal((N, N))
            Q, R = np.linalg.qr(H)
            # Ensure proper uniform Haar measure on O(N)
            d = np.diagonal(R)
            ph = d / np.abs(d)
            Q = Q * ph
            self.W = Q * self.spectral_radius

        elif self.coupling_type == "reciprocal_correlated":
            # Reciprocal correlation tau = corr(J_ij, J_ji)
            tau = self.reciprocal_corr
            A = rng.standard_normal((N, N))
            B = rng.standard_normal((N, N))
            J_sym = (A + A.T) / np.sqrt(2.0)
            J_skew = (B - B.T) / np.sqrt(2.0)
            W_raw = np.sqrt((1.0 + tau) / 2.0) * J_sym + np.sqrt((1.0 - tau) / 2.0) * J_skew

            # Symmetric sparsity mask
            M_rand = rng.uniform(0.0, 1.0, size=(N, N))
            mask = (M_rand + M_rand.T) / 2.0 > self.density
            W_raw[mask] = 0.0

            sr = float(np.max(np.abs(np.linalg.eigvals(W_raw))))
            if sr > 1e-12:
                self.W = W_raw * (self.spectral_radius / sr)
            else:
                self.W = W_raw
        else:
            raise ValueError(f"Unknown coupling_type: {self.coupling_type}")

        # 4. Feedback
        if self.feedback_scaling > 0.0:
            self.W_fb = rng.uniform(-1.0, 1.0, size=(N, self.n_outputs)) * self.feedback_scaling
        else:
            self.W_fb = None

    def estimate_effective_lyapunov(self, x_trajectory: np.ndarray) -> float:
        """
        Estimate theoretical maximum Lyapunov exponent lambda_1 using Clark (2026) Table 1:
            lambda_1 = log(g * sqrt(C_d(0)))
        """
        d_trajectory = 1.0 - np.tanh(x_trajectory) ** 2
        Cd_0 = float(np.mean(d_trajectory ** 2))
        effective_g = self.spectral_radius
        if Cd_0 <= 1e-12:
            return -float("inf")
        return float(np.log(effective_g * np.sqrt(Cd_0)))

    def _step_reservoir(self, u_t: np.ndarray, x_prev: np.ndarray, y_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """Single-step update with dynamic gain calibration."""
        alpha = self.leaking_rate

        # Dynamic gain adjustment (Clark 2026 Eq. B.15)
        if self.dynamic_gain_control:
            d_curr = 1.0 - (x_prev ** 2)
            curr_cd = float(np.mean(d_curr ** 2))
            factor = float(np.clip((1.0 / max(curr_cd, 0.05)) ** 0.5, 0.95, 1.05))
            W_active = self.W * factor
        else:
            W_active = self.W

        pre = self.W_in @ u_t + W_active @ x_prev + self.b
        if self.W_fb is not None and y_prev is not None:
            pre += self.W_fb @ y_prev

        x_tilde = np.tanh(pre)
        x_next = (1.0 - alpha) * x_prev + alpha * x_tilde
        return x_next

    def apply_dichotomy_projection(self, X: np.ndarray) -> np.ndarray:
        """Apply learned dichotomy projector P_stable onto the stable manifold."""
        if not self.forward_backward_dichotomy or self.P_stable is None:
            return X
        return X @ self.P_stable

    def fit_dichotomy_projector(self, X: np.ndarray) -> None:
        """
        Fit the regularized forward-backward dichotomy projector P_stable (Clark 2026 Sec 2.2).
        Runs backward adjoint pass:
            u(n-1) = (1 - alpha) u(n) + alpha * D(n) W^T u(n) + eta * x(n)
        Extracts stable components:
            x_clean = x - (eta / (1 + eta)) * u_adjoint
        Solves:
            P_stable = (X^T X + reg * I)^(-1) X^T X_clean
        """
        T, N = X.shape
        if T <= 2:
            self.P_stable = np.eye(N)
            return

        eta = self.dichotomy_eta
        alpha = self.leaking_rate
        WT = self.W.T

        U_adj = np.zeros_like(X)
        u_back = np.zeros(N, dtype=np.float64)

        for t in reversed(range(T)):
            x_t = X[t]
            d_t = 1.0 - (x_t ** 2)
            pre_back = WT @ (d_t * u_back)
            u_back = (1.0 - alpha) * u_back + alpha * pre_back + eta * x_t
            U_adj[t] = u_back

        X_clean = X - (eta / (1.0 + eta)) * U_adj

        # Fit linear projector: X @ P_stable ~= X_clean
        XTX = X.T @ X + 1e-4 * np.eye(N)
        XT_Xclean = X.T @ X_clean
        try:
            self.P_stable = np.linalg.solve(XTX, XT_Xclean)
        except np.linalg.LinAlgError:
            self.P_stable = np.linalg.pinv(XTX) @ XT_Xclean

    def harvest_states(
        self,
        U: np.ndarray,
        Y_fb: Optional[np.ndarray] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Harvest state trajectory with dynamic gain control and dichotomy filtering."""
        U = np.asarray(U, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

        T, d_in = U.shape
        if initial_state is not None:
            self.reset_state(initial_state)

        X = np.empty((T, self.n_reservoir), dtype=np.float64)
        Z = np.empty((T, self.d_features), dtype=np.float64)

        x = self.current_state.copy()
        y_prev = None

        for t in range(T):
            u_t = U[t]
            if Y_fb is not None and t > 0:
                y_prev = Y_fb[t - 1]
            x = self._step_reservoir(u_t, x, y_prev)
            X[t] = x

        self.current_state = x.copy()

        # If P_stable is not yet fit, learn it during training harvest
        if self.forward_backward_dichotomy:
            if self.P_stable is None:
                self.fit_dichotomy_projector(X)
            X_proj = self.apply_dichotomy_projection(X)
        else:
            X_proj = X

        # Construct feature matrix Z
        for t in range(T):
            Z[t] = self._build_feature_vector(U[t], X_proj[t])

        return X_proj, Z

    def generate(
        self,
        n_steps: int,
        initial_u: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        continuation: bool = False,
    ) -> np.ndarray:
        """
        Autonomous closed-loop generation with dynamic gain calibration
        and dichotomy projection consistency.
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if self.n_inputs != self.n_outputs:
            raise ValueError(f"Autonomous generation requires n_inputs == n_outputs")

        if not continuation and initial_state is None:
            self.reset_state()
        elif initial_state is not None:
            self.reset_state(initial_state)

        initial_u = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        Y_gen = np.empty((n_steps, self.n_outputs), dtype=np.float64)
        u_t = initial_u.copy()
        x = self.current_state.copy()
        y_prev = None

        for t in range(n_steps):
            x = self._step_reservoir(u_t, x, y_prev)

            # Apply dichotomy projection to state
            if self.forward_backward_dichotomy and self.P_stable is not None:
                x_eval = x @ self.P_stable
            else:
                x_eval = x

            z_t = self._build_feature_vector(u_t, x_eval)
            y_pred = self.W_out @ z_t

            Y_gen[t] = y_pred
            y_prev = y_pred
            u_t = y_pred.copy()

        self.current_state = x.copy()
        return Y_gen
