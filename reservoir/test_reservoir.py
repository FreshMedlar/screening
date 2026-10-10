"""
Unit tests for the reservoir package.
Tests:
- EchoStateNetwork initialization, state harvesting, ridge regression fit, predict, generate
- Data generators (Mackey-Glass, NARMA, MC, Lorenz)
- Metrics (NRMSE, NMSE, MC score, VPT, attractor divergence)
- Stability analyzers (ESP verification, autonomous rollout)
"""

import unittest
import numpy as np

from reservoir.esn import EchoStateNetwork
from reservoir.data import (
    generate_mackey_glass,
    generate_narma,
    generate_memory_capacity,
    generate_nonlinear_memory_capacity,
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


class TestEchoStateNetwork(unittest.TestCase):
    def setUp(self):
        self.n_in = 2
        self.n_res = 50
        self.n_out = 1
        self.esn = EchoStateNetwork(
            n_inputs=self.n_in,
            n_reservoir=self.n_res,
            n_outputs=self.n_out,
            spectral_radius=0.9,
            leaking_rate=0.7,
            input_scaling=0.5,
            density=0.2,
            random_state=42,
        )

    def test_initialization_and_spectral_radius(self):
        eigs = np.linalg.eigvals(self.esn.W)
        sr = np.max(np.abs(eigs))
        self.assertAlmostEqual(sr, 0.9, places=4)
        self.assertEqual(self.esn.W.shape, (self.n_res, self.n_res))
        self.assertEqual(self.esn.W_in.shape, (self.n_res, self.n_in))

    def test_state_harvesting(self):
        T = 100
        U = np.random.randn(T, self.n_in)
        X, Z = self.esn.harvest_states(U)
        self.assertEqual(X.shape, (T, self.n_res))
        self.assertEqual(Z.shape, (T, self.esn.d_features))
        self.assertTrue(np.all(np.abs(X) <= 1.0))

    def test_fit_and_predict(self):
        T = 400
        U = np.random.randn(T, self.n_in)
        # Target is a linear combination of input with delay
        Y = np.roll(U[:, 0:1], 1, axis=0)

        self.esn.fit(U, Y, washout=50, reg=1e-4)
        self.assertIsNotNone(self.esn.W_out)
        self.assertEqual(self.esn.W_out.shape, (self.n_out, self.esn.d_features))

        U_test = np.random.randn(150, self.n_in)
        Y_pred = self.esn.predict(U_test)
        self.assertEqual(Y_pred.shape, (150, self.n_out))
        self.assertTrue(np.all(np.isfinite(Y_pred)))

    def test_autonomous_generation(self):
        esn_auto = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=40,
            n_outputs=1,
            spectral_radius=0.85,
            random_state=42,
        )
        t = np.linspace(0, 20, 300)
        u = np.sin(t).reshape(-1, 1)
        y = np.sin(t + 0.1).reshape(-1, 1)

        esn_auto.fit(u, y, washout=50, reg=1e-3)
        gen = esn_auto.generate(n_steps=50, initial_u=u[-1:])
        self.assertEqual(gen.shape, (50, 1))
        self.assertTrue(np.all(np.isfinite(gen)))

    def test_jacobian(self):
        u_t = np.array([0.5, -0.2])
        x_prev = np.zeros(self.n_res)
        J = self.esn.jacobian(u_t, x_prev)
        self.assertEqual(J.shape, (self.n_res, self.n_res))
        self.assertTrue(np.all(np.isfinite(J)))


class TestDataGenerators(unittest.TestCase):
    def test_mackey_glass(self):
        u, y = generate_mackey_glass(length=500, tau=17.0, seed=42)
        self.assertEqual(u.shape, (500, 1))
        self.assertEqual(y.shape, (500, 1))
        # Mackey-Glass chaotic attractor typically stays bounded in [0.3, 1.6]
        self.assertTrue(np.all(u > 0.0))
        self.assertTrue(np.all(u < 2.5))
        self.assertTrue(np.all(np.isfinite(u)))

    def test_narma(self):
        for k in [10, 20, 30]:
            u, y = generate_narma(k=k, length=400, seed=42)
            self.assertEqual(u.shape, (400, 1))
            self.assertEqual(y.shape, (400, 1))
            self.assertTrue(np.all(np.isfinite(y)))

    def test_memory_capacity_data(self):
        u, Y = generate_memory_capacity(length=300, max_delay=20, seed=42)
        self.assertEqual(u.shape, (300, 1))
        self.assertEqual(Y.shape, (300, 20))
        # Verify delay alignment
        np.testing.assert_array_equal(Y[5:, 4], u[:-5, 0])

    def test_lorenz(self):
        u, y = generate_lorenz(length=300, seed=42)
        self.assertEqual(u.shape, (300, 3))
        self.assertEqual(y.shape, (300, 3))
        self.assertTrue(np.all(np.isfinite(u)))


class TestMetricsAndStability(unittest.TestCase):
    def test_nrmse_and_nmse(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        self.assertAlmostEqual(compute_nrmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(compute_nmse(y_true, y_pred), 0.0)

        # Distorted
        y_dist = y_true + 1.0
        nrmse = compute_nrmse(y_true, y_dist)
        self.assertGreater(nrmse, 0.0)

    def test_memory_capacity_metric(self):
        y_true = np.random.randn(200, 5)
        # Perfect prediction
        mc, profile = compute_memory_capacity(y_true, y_true)
        self.assertAlmostEqual(mc, 5.0, places=4)
        for r2 in profile:
            self.assertAlmostEqual(r2, 1.0, places=4)

    def test_vpt(self):
        y_true = np.ones(100)
        y_pred = np.ones(100)
        y_pred[20:] = 10.0  # Divergence at step 20
        vpt = compute_vpt(y_true, y_pred, threshold=0.5)
        self.assertEqual(vpt, 20)

    def test_attractor_divergence(self):
        y_true = np.sin(np.linspace(0, 10, 200))
        y_stable = y_true + 0.05 * np.random.randn(200)
        diag = compute_attractor_divergence(y_true, y_stable)
        self.assertTrue(diag["is_stable"])
        self.assertFalse(diag["is_diverged"])
        self.assertFalse(diag["is_collapsed"])

    def test_echo_state_property(self):
        esn = EchoStateNetwork(
            n_inputs=1,
            n_reservoir=40,
            spectral_radius=0.7,
            leaking_rate=0.8,
            random_state=42,
        )
        res = evaluate_echo_state_property(esn, input_length=500, n_trials=3, seed=42)
        self.assertTrue(res["esp_satisfied"])
        self.assertLess(res["mean_final_distance"], 1e-4)


class TestLyapunovEnhancedReservoir(unittest.TestCase):
    def test_initialization_and_reciprocal_correlation(self):
        from reservoir.lyapunov_esn import LyapunovEnhancedReservoir
        esn = LyapunovEnhancedReservoir(
            n_inputs=1,
            n_reservoir=60,
            coupling_type="reciprocal_correlated",
            reciprocal_corr=-0.6,
            dynamic_gain_control=True,
            forward_backward_dichotomy=True,
            random_state=42,
        )
        self.assertEqual(esn.W.shape, (60, 60))
        # Verify reciprocal symmetry skew
        J_sym = (esn.W + esn.W.T) / 2.0
        J_skew = (esn.W - esn.W.T) / 2.0
        norm_sym = np.linalg.norm(J_sym)
        norm_skew = np.linalg.norm(J_skew)
        # With negative reciprocal correlation tau=-0.6, skew component should dominate
        self.assertGreater(norm_skew, norm_sym)

    def test_fit_and_dichotomy_filtering(self):
        from reservoir.lyapunov_esn import LyapunovEnhancedReservoir
        esn = LyapunovEnhancedReservoir(
            n_inputs=1,
            n_reservoir=50,
            n_outputs=1,
            forward_backward_dichotomy=True,
            dynamic_gain_control=True,
            random_state=42,
        )
        U = np.random.randn(200, 1)
        Y = np.roll(U, 2, axis=0)
        esn.fit(U, Y, washout=30, reg=1e-4)
        pred = esn.predict(U[:50])
        self.assertEqual(pred.shape, (50, 1))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_lyapunov_estimation(self):
        from reservoir.lyapunov_esn import LyapunovEnhancedReservoir
        esn = LyapunovEnhancedReservoir(n_inputs=1, n_reservoir=40, random_state=42)
        X = np.random.randn(100, 40) * 0.5
        lam1 = esn.estimate_effective_lyapunov(X)
        self.assertTrue(np.isfinite(lam1))


if __name__ == "__main__":
    unittest.main()

