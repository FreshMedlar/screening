"""
Echo State Network (ESN) implementation.

Classic reservoir computing with leaky-integrator neurons and a static linear
readout trained via Ridge Regression (Tikhonov Regularization).
"""

from typing import Optional, Tuple, Union
import numpy as np
try:
    import scipy.linalg as sla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class EchoStateNetwork:
    """
    Classic Echo State Network (ESN) with leaky-integrator recurrent units.

    State update equation:
        \\tilde{x}(t) = \\tanh(W_{in} u(t) + W x(t-1) + W_{fb} y(t-1) + b)
        x(t) = (1 - \\alpha) x(t-1) + \\alpha \\tilde{x}(t)

    Readout equation:
        y(t) = W_{out} z(t)
    where z(t) is the augmented feature state (e.g. [1, u(t), x(t)]).
    W_{out} is fitted analytically using Ridge Regression.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        n_reservoir: int = 300,
        n_outputs: int = 1,
        spectral_radius: float = 0.95,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.1,
        density: float = 0.1,
        feedback_scaling: float = 0.0,
        include_input: bool = True,
        include_bias: bool = True,
        quadratic_features: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.n_inputs = int(n_inputs)
        self.n_reservoir = int(n_reservoir)
        self.n_outputs = int(n_outputs)
        self.spectral_radius = float(spectral_radius)
        self.leaking_rate = float(leaking_rate)
        self.input_scaling = float(input_scaling)
        self.bias_scaling = float(bias_scaling)
        self.density = float(density)
        self.feedback_scaling = float(feedback_scaling)
        self.include_input = bool(include_input)
        self.include_bias = bool(include_bias)
        self.quadratic_features = bool(quadratic_features)
        self.random_state = random_state

        if not (0.0 < self.leaking_rate <= 1.0):
            raise ValueError(f"leaking_rate must be in (0, 1], got {self.leaking_rate}")

        # Dimension of the expanded readout vector z(t)
        self.d_features = self.n_reservoir
        if self.quadratic_features:
            self.d_features += self.n_reservoir
        if self.include_input:
            self.d_features += self.n_inputs
        if self.include_bias:
            self.d_features += 1

        # Internal state
        self.current_state = np.zeros(self.n_reservoir, dtype=np.float64)
        self.W_out = None  # Shape: (n_outputs, d_features)

        # Initialize fixed reservoir weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize W_in, W, b, and optional W_fb matrices."""
        rng = np.random.default_rng(self.random_state)

        # 1. Input weights W_in ~ Uniform[-1, 1] scaled by input_scaling
        self.W_in = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_inputs))
        self.W_in *= self.input_scaling

        # 2. Bias vector b ~ Uniform[-1, 1] scaled by bias_scaling
        self.b = rng.uniform(-1.0, 1.0, size=(self.n_reservoir,)) * self.bias_scaling

        # 3. Recurrent reservoir weights W with specified density and spectral radius
        W_raw = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_reservoir))
        mask = rng.uniform(0.0, 1.0, size=(self.n_reservoir, self.n_reservoir)) > self.density
        W_raw[mask] = 0.0

        # Scale spectral radius rho(W)
        eigenvalues = np.linalg.eigvals(W_raw)
        current_sr = float(np.max(np.abs(eigenvalues)))
        if current_sr > 1e-12:
            self.W = W_raw * (self.spectral_radius / current_sr)
        else:
            self.W = W_raw

        # 4. Optional output feedback weights W_fb
        if self.feedback_scaling > 0.0:
            self.W_fb = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_outputs))
            self.W_fb *= self.feedback_scaling
        else:
            self.W_fb = None

    def reset_state(self, state: Optional[np.ndarray] = None) -> None:
        """Reset internal reservoir state to zero or specified vector."""
        if state is None:
            self.current_state = np.zeros(self.n_reservoir, dtype=np.float64)
        else:
            state_arr = np.asarray(state, dtype=np.float64)
            if state_arr.shape != (self.n_reservoir,):
                raise ValueError(f"State must have shape ({self.n_reservoir},), got {state_arr.shape}")
            self.current_state = state_arr.copy()

    def _build_feature_vector(self, u_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Construct the readout design feature vector z(t).
        Order: [bias, u(t), x(t), x(t)^2 (optional)].
        """
        parts = []
        if self.include_bias:
            parts.append(np.array([1.0], dtype=np.float64))
        if self.include_input:
            parts.append(u_t)
        parts.append(x_t)
        if self.quadratic_features:
            parts.append(x_t ** 2)
        return np.concatenate(parts, axis=0)

    def harvest_states(
        self,
        U: np.ndarray,
        Y_fb: Optional[np.ndarray] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drive reservoir with sequence U and return reservoir state trajectory X
        and readout design matrix Z.

        Parameters
        ----------
        U : np.ndarray, shape (T, n_inputs)
        Y_fb : Optional[np.ndarray], shape (T, n_outputs)
            Teacher-forced feedback sequence (if model uses feedback).
        initial_state : Optional[np.ndarray]

        Returns
        -------
        X : np.ndarray, shape (T, n_reservoir)
        Z : np.ndarray, shape (T, d_features)
        """
        U = np.asarray(U, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

        T, d_in = U.shape
        if d_in != self.n_inputs:
            raise ValueError(f"Expected U to have {self.n_inputs} inputs, got {d_in}")

        if initial_state is not None:
            self.reset_state(initial_state)

        X = np.empty((T, self.n_reservoir), dtype=np.float64)
        Z = np.empty((T, self.d_features), dtype=np.float64)

        x = self.current_state
        alpha = self.leaking_rate
        has_fb = (self.W_fb is not None) and (Y_fb is not None)

        W_in = self.W_in
        W = self.W
        b = self.b
        W_fb = self.W_fb

        for t in range(T):
            u_t = U[t]
            pre = W_in @ u_t + W @ x + b
            if has_fb and t > 0:
                pre += W_fb @ Y_fb[t - 1]

            x_tilde = np.tanh(pre)
            x = (1.0 - alpha) * x + alpha * x_tilde
            X[t] = x
            Z[t] = self._build_feature_vector(u_t, x)

        self.current_state = x.copy()
        return X, Z

    def fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        washout: int = 100,
        reg: float = 1e-4,
    ) -> "EchoStateNetwork":
        """
        Harvest states and solve for static readout weights W_out using Ridge Regression.

        Parameters
        ----------
        U : np.ndarray, shape (T, n_inputs)
        Y : np.ndarray, shape (T, n_outputs)
        washout : int
            Initial transient steps to discard before regression.
        reg : float
            L2 regularization strength (lambda).

        Returns
        -------
        self
        """
        U = np.asarray(U, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T = len(U)
        if T <= washout:
            raise ValueError(f"Sequence length {T} must be greater than washout {washout}")
        if len(Y) != T:
            raise ValueError(f"Length mismatch: U has {T} samples, Y has {len(Y)} samples")

        # Harvest state trajectory
        _, Z = self.harvest_states(U, Y_fb=Y)

        # Discard transient
        Z_train = Z[washout:]  # Shape: (T - washout, d_features)
        Y_train = Y[washout:]  # Shape: (T - washout, n_outputs)

        # Solve Ridge Regression: (Z^T Z + reg * I) W_out^T = Z^T Y
        d = self.d_features
        ZT = Z_train.T
        ZTZ = ZT @ Z_train
        ZTY = ZT @ Y_train

        reg_eye = reg * np.eye(d, dtype=np.float64)
        A = ZTZ + reg_eye

        if HAS_SCIPY:
            try:
                # Solve using Cholesky since A is symmetric positive definite
                W_out_T = sla.solve(A, ZTY, assume_a="pos")
            except (np.linalg.LinAlgError, sla.LinAlgError):
                W_out_T = sla.solve(A, ZTY)
        else:
            try:
                W_out_T = np.linalg.solve(A, ZTY)
            except np.linalg.LinAlgError:
                W_out_T = np.linalg.pinv(A) @ ZTY

        self.W_out = W_out_T.T  # Shape: (n_outputs, d_features)
        return self

    def predict(
        self,
        U: np.ndarray,
        continuation: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Open-loop (teacher-forced) prediction for input sequence U.

        Parameters
        ----------
        U : np.ndarray, shape (T, n_inputs)
        continuation : bool
            If True, retain internal state from previous call.
            If False and initial_state is None, reset state to zeros.
        initial_state : Optional[np.ndarray]

        Returns
        -------
        Y_pred : np.ndarray, shape (T, n_outputs)
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if not continuation and initial_state is None:
            self.reset_state()

        _, Z = self.harvest_states(U, initial_state=initial_state)
        Y_pred = Z @ self.W_out.T
        return Y_pred

    def generate(
        self,
        n_steps: int,
        initial_u: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        continuation: bool = False,
    ) -> np.ndarray:
        """
        Closed-loop / autonomous generative rollout.
        The predicted output at step t is fed back as the input at step t+1.
        (Requires n_inputs == n_outputs).

        Parameters
        ----------
        n_steps : int
            Number of autonomous steps to generate.
        initial_u : np.ndarray, shape (n_inputs,) or (1, n_inputs)
        initial_state : Optional[np.ndarray]
        continuation : bool

        Returns
        -------
        Y_gen : np.ndarray, shape (n_steps, n_outputs)
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if self.n_inputs != self.n_outputs:
            raise ValueError(
                f"Autonomous generation requires n_inputs == n_outputs, got {self.n_inputs} and {self.n_outputs}"
            )

        if not continuation and initial_state is None:
            self.reset_state()
        elif initial_state is not None:
            self.reset_state(initial_state)

        initial_u = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        if len(initial_u) != self.n_inputs:
            raise ValueError(f"initial_u length must match n_inputs={self.n_inputs}")

        Y_gen = np.empty((n_steps, self.n_outputs), dtype=np.float64)
        u_t = initial_u.copy()
        x = self.current_state
        alpha = self.leaking_rate

        W_in = self.W_in
        W = self.W
        b = self.b
        W_fb = self.W_fb
        W_out = self.W_out

        y_prev = None

        for t in range(n_steps):
            pre = W_in @ u_t + W @ x + b
            if W_fb is not None and y_prev is not None:
                pre += W_fb @ y_prev

            x_tilde = np.tanh(pre)
            x = (1.0 - alpha) * x + alpha * x_tilde
            z_t = self._build_feature_vector(u_t, x)
            y_pred = W_out @ z_t

            Y_gen[t] = y_pred
            y_prev = y_pred
            u_t = y_pred.copy()  # Autoregressive closed-loop feedback

        self.current_state = x.copy()
        return Y_gen

    def jacobian(self, u_t: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """
        Compute the local analytical Jacobian matrix J = dx(t) / dx(t-1).

        J = (1 - alpha) * I + alpha * diag(1 - x_tilde^2) * W
        """
        u_t = np.asarray(u_t, dtype=np.float64).reshape(-1)
        x_prev = np.asarray(x_prev, dtype=np.float64).reshape(-1)

        pre = self.W_in @ u_t + self.W @ x_prev + self.b
        x_tilde = np.tanh(pre)
        dtanh = 1.0 - x_tilde ** 2  # Shape: (n_reservoir,)

        # diag(dtanh) @ W
        scaled_W = dtanh[:, None] * self.W
        J = (1.0 - self.leaking_rate) * np.eye(self.n_reservoir) + self.leaking_rate * scaled_W
        return J
