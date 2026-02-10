"""
HoloG Differentiable Calibration — Gradient Descent Optimizer

Gradient descent optimization loop for calibrating plaquette control
pulses using JAX's automatic differentiation.

Requires: pip install unified-qec[jax]

Migrated from: HoloG/holog_diffg/calibrate.py
"""

import numpy as np
from typing import Tuple, Optional, Dict
from unified_qec.calibration.config import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

from unified_qec.calibration.simulator import simulate_plaquette


def calibrate_plaquette(
    initial_params: Optional[np.ndarray] = None,
    n_data: int = 6,
    learning_rate: float = 0.01,
    num_steps: int = 200,
    t1_us: float = 30.0,
    zz_strength: float = 0.02,
    verbose: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Calibrate plaquette control pulses via gradient descent.

    Uses JAX's automatic differentiation to minimize syndrome
    error probability by adjusting Rx rotation angles.

    Parameters
    ----------
    initial_params : ndarray, optional
        Initial pulse angles. Defaults to π + small perturbation.
    n_data : int
        Number of data qubits.
    learning_rate : float
        Gradient descent step size.
    num_steps : int
        Number of optimization steps.
    t1_us : float
        T1 relaxation time.
    zz_strength : float
        ZZ crosstalk strength.
    verbose : bool
        Print progress.

    Returns
    -------
    optimal_params : ndarray
        Optimized pulse angles.
    error_history : list
        Syndrome error probability at each step.

    Raises
    ------
    ImportError
        If JAX is not available.
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for calibration. "
            "Install with: pip install unified-qec[jax]"
        )

    if initial_params is None:
        initial_params = jnp.ones(n_data) * jnp.pi + jnp.array(
            np.random.normal(0, 0.1, n_data)
        )

    params = jnp.array(initial_params, dtype=jnp.float64)

    def loss_fn(theta):
        return simulate_plaquette(theta, t1_us, zz_strength, n_data)

    grad_fn = jax.grad(loss_fn)

    error_history = []

    for step in range(num_steps):
        error = loss_fn(params)
        error_history.append(float(error))

        grads = grad_fn(params)
        params = params - learning_rate * grads

        if verbose and step % 50 == 0:
            print(f"  Step {step:4d}: error = {error:.6f}")

    if verbose:
        print(f"  Final:     error = {error_history[-1]:.6f}")

    return np.array(params), error_history


def run_benchmark(
    t1_us: float = 30.0,
    zz_strength: float = 0.02,
    n_data: int = 6,
    num_steps: int = 200,
) -> Dict:
    """
    Run HoloG calibration benchmark.

    Simulates a specific hardware scenario and optimizes the
    control parameters.

    Parameters
    ----------
    t1_us : float
        T1 relaxation time.
    zz_strength : float
        ZZ crosstalk strength.
    n_data : int
        Number of data qubits.
    num_steps : int
        Optimization steps.

    Returns
    -------
    dict
        Benchmark results including initial/final error,
        optimized parameters, and full error history.
    """
    initial_error = simulate_plaquette(
        np.ones(n_data) * np.pi, t1_us, zz_strength, n_data
    )

    optimal_params, error_history = calibrate_plaquette(
        n_data=n_data,
        num_steps=num_steps,
        t1_us=t1_us,
        zz_strength=zz_strength,
        verbose=False,
    )

    return {
        "initial_error": initial_error,
        "final_error": error_history[-1],
        "improvement_factor": initial_error / max(error_history[-1], 1e-15),
        "optimal_params": optimal_params.tolist(),
        "error_history": error_history,
        "t1_us": t1_us,
        "zz_strength": zz_strength,
    }
