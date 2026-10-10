"""
Echo State Network (ESN) / Reservoir Computing package.
Includes classic reservoir with static readout trained using ridge regression,
benchmarks for memory (Memory Capacity, NARMA-10/20/30) and stability (Mackey-Glass,
Lorenz, Echo State Property verification, spectral radius sweep).
"""

from .esn import EchoStateNetwork
from .lyapunov_esn import LyapunovEnhancedReservoir
from .data import (
    generate_mackey_glass,
    generate_narma,
    generate_memory_capacity,
    generate_lorenz,
)
from .metrics import (
    compute_nrmse,
    compute_nmse,
    compute_memory_capacity,
    compute_vpt,
    compute_attractor_divergence,
)
from .stability import (
    evaluate_echo_state_property,
    evaluate_spectral_radius_sweep,
    evaluate_autonomous_stability,
)

__all__ = [
    "EchoStateNetwork",
    "LyapunovEnhancedReservoir",
    "generate_mackey_glass",
    "generate_narma",
    "generate_memory_capacity",
    "generate_lorenz",
    "compute_nrmse",
    "compute_nmse",
    "compute_memory_capacity",
    "compute_vpt",
    "compute_attractor_divergence",
    "evaluate_echo_state_property",
    "evaluate_spectral_radius_sweep",
    "evaluate_autonomous_stability",
]
