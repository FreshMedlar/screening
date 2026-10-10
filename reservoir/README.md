# Echo State Network (ESN) — Memory & Stability Suite

A modular, high-performance implementation of a classic **Echo State Network (ESN)** with leaky-integrator recurrent neurons and a static linear readout trained via **Ridge Regression (Tikhonov Regularization)**.

The package is evaluated on established benchmarks for **Memory** and **Stability**:
- **Information Memory Capacity (MC)** (Jaeger 2002)
- **NARMA-10, NARMA-20, NARMA-30** (nonlinear long-term memory)
- **Mackey-Glass (MG-17)** chaotic time series (1-step prediction & autonomous rollout stability)
- **Echo State Property (ESP)** empirical trajectory contraction verification
- **Spectral Radius Edge-of-Chaos Sweep** ($\rho \in [0.2, 1.5]$)
- **Lorenz-63** 3D chaotic attractor tracking

---

## 1. Mathematical Formulation

### State Update Equation
$$\tilde{x}(t) = \tanh\left(W_{in} u(t) + W x(t-1) + W_{fb} y(t-1) + b\right)$$
$$x(t) = (1 - \alpha) x(t-1) + \alpha \tilde{x}(t)$$

where:
- $u(t) \in \mathbb{R}^{d_{in}}$: input signal
- $x(t) \in \mathbb{R}^{N}$: hidden reservoir state
- $\alpha \in (0, 1]$: leaking rate
- $W \in \mathbb{R}^{N \times N}$: sparse recurrent weight matrix with spectral radius $\rho(W) = \max_i |\lambda_i(W)|$
- $W_{in} \in \mathbb{R}^{N \times d_{in}}$: input projection matrix
- $b \in \mathbb{R}^{N}$: bias vector

### Readout & Ridge Regression
Augmented feature vector: $z(t) = [1, u(t), x(t)]^T \in \mathbb{R}^{D_z}$.

The static readout matrix $W_{out} \in \mathbb{R}^{d_{out} \times D_z}$ is solved analytically via Ridge Regression:
$$W_{out}^T = \left(Z^T Z + \lambda I\right)^{-1} Z^T Y$$
where $\lambda \ge 0$ is the regularization parameter.

---

## 2. Directory Structure

```text
reservoir/
├── __init__.py           # Package exports
├── esn.py                # EchoStateNetwork core class with Ridge regression & autonomous rollout
├── data.py               # Dataset generators: Mackey-Glass, NARMA-k, Memory Capacity, Lorenz
├── metrics.py            # NRMSE, NMSE, Memory Capacity, Valid Prediction Time (VPT), Attractor divergence
├── stability.py          # Echo State Property (ESP) analyzer & Spectral Radius sweep
├── benchmark.py          # Full Classic ESN benchmark suite
├── compare_benchmarks.py  # Head-to-head comparative benchmark runner (Classic vs LE-ESN)
├── test_reservoir.py     # Unit test suite
├── benchmark_results.png # Classic ESN benchmark dashboard
├── comparison_results.png# Comparative dashboard (Classic vs LE-ESN)
└── README.md             # Documentation
```

---

## 3. Lyapunov-Enhanced ESN (LE-ESN) from Clark (2026)

Based on *David G. Clark (2026), "Lyapunov spectrum of random neural networks" (arXiv:2610.12426)*:

1. **Critical Edge-of-Chaos Spectrum Calibration**:
   Classical Gaussian sparse reservoirs exhibit the circular law where many eigenvalues have $|\mu| \ll g \approx 1$, leading to rapid state dissipation. LE-ESN enforces critical unitary/orthogonal spectrum scaling where all modes reside on the critical circle $|\mu| = g \approx 1$.
2. **Trajectory Gain Normalization** (Table 1 & Eq. B.15):
   Using the paper's closed-form relation $\lambda_1 = \log(g \sqrt{C_d(0)})$ with gain variance $C_d(0) = \langle (1 - \tanh^2(x))^2 \rangle$, LE-ESN applies dynamic homeostatic gain compensation $W_{active} = W \cdot (1 / \sqrt{C_d(0)})$, preventing the effective Lyapunov exponent from dropping into over-contractive damping.
3. **Forward-Backward Dichotomy Regularization** (Sec 2.1-2.2):
   Computes the minimum-norm dichotomy projection onto the stable invariant manifold $E^s$ using the regularized resolvent $(K^T K + \eta^2 I)^{-1} K^T$.

---

## 4. Running Benchmarks & Tests

Using the environment in `/home/medlar/Projects/.venv`:

```bash
# Run unit tests (17 tests)
/home/medlar/Projects/.venv/bin/python -m unittest discover -s reservoir -p "test_*.py" -v

# Run Classic ESN benchmark suite
/home/medlar/Projects/.venv/bin/python reservoir/benchmark.py --save-plot reservoir/benchmark_results.png

# Run Head-to-Head Comparison: Classic ESN vs LE-ESN
/home/medlar/Projects/.venv/bin/python reservoir/compare_benchmarks.py --save-plot reservoir/comparison_results.png
```

---

## 5. Head-to-Head Benchmark Results

| Benchmark Task | Metric | Classic ESN | LE-ESN (Clark 2026) | Improvement / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Memory Capacity ($N=100$)** | Total $MC = \sum r^2(d)$ | 31.64 | **70.21** | **+121.9%** |
| **Linear Memory Capacity ($N=300$)** | Total $MC = \sum r^2(d)$ | 50.19 | **78.88** | **+57.2%** |
| **Linear Memory Capacity ($N=500$)** | Total $MC = \sum r^2(d)$ | 59.11 | **79.30** | **+34.2%** |
| **NARMA-10** | NRMSE | 0.3289 | **0.3123** | **+5.1% error reduction** |
| **NARMA-20** | NRMSE | 0.4626 | **0.4352** | **+5.9% error reduction** |
| **NARMA-30** | NRMSE | 0.6071 | **0.6076** | On par (within 0.1%) |
| **Mackey-Glass (MG-17) 1-Step** | NRMSE | 0.001428 | **0.001105** | **+22.6% error reduction** |
| **Mackey-Glass Autonomous Rollout** | Valid Prediction Time (VPT) | 107 steps | **154 steps** | **+43.9% longer stable horizon** |
| **Mackey-Glass Attractor Variance** | Variance Ratio (Pred/True) | 1.0487 | **1.0190** | Improved attractor fidelity |
| **Lorenz-63 Autonomous Rollout** | 3D Autonomous VPT | 112 steps | **112 steps** | Stable chaotic tracking |

