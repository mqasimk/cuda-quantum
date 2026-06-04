# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for cudaq.contrib.propagator (issue #3437). The dynamics backend needs
# an NVIDIA GPU, so the whole module is skipped when none is available. The
# oracle is the closed-form U(t) = expm(-i H t) for time-independent H.

import numpy as np
import pytest

import cudaq

# Skip everything if there is no GPU (dynamics target is unavailable).
if cudaq.num_available_gpus() == 0:
    pytest.skip("propagator requires the dynamics backend (GPU)",
                allow_module_level=True)

scipy_linalg = pytest.importorskip("scipy.linalg")

from cudaq.dynamics import Schedule  # noqa: E402  TODO: confirm import path


# Pauli matrices for building numpy oracles.
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _expm_reference(h_matrix, t):
    """Closed-form propagator U(t) = expm(-i H t) for time-independent H."""
    return scipy_linalg.expm(-1j * h_matrix * t)


def test_propagator_time_independent_single_qubit():
    """U(t) for H = (omega/2) Z matches expm and is unitary."""
    omega = 2.0 * np.pi * 0.1
    h_op = 0.5 * omega * cudaq.spin.z(0)
    h_mat = 0.5 * omega * _Z
    t_end = 1.3
    schedule = Schedule(np.linspace(0.0, t_end, 21), ["t"])

    u = cudaq.contrib.propagator(h_op, {0: 2}, schedule)

    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-6)  # unitary
    np.testing.assert_allclose(u, _expm_reference(h_mat, t_end), atol=1e-4)


def test_propagator_intermediate_states():
    """store_intermediate_results returns U at every step of the schedule."""
    omega = 2.0 * np.pi * 0.1
    h_op = 0.5 * omega * cudaq.spin.z(0)
    h_mat = 0.5 * omega * _Z
    steps = np.linspace(0.0, 2.0, 11)
    schedule = Schedule(steps, ["t"])

    us = cudaq.contrib.propagator(h_op, {0: 2}, schedule,
                                  store_intermediate_results=True)

    assert len(us) == len(steps)
    for u, t in zip(us, steps):
        np.testing.assert_allclose(u, _expm_reference(h_mat, t), atol=1e-4)


def test_propagator_batched_parameter_sweep():
    """A list of Hamiltonians returns one propagator per parameter."""
    omegas = [2.0 * np.pi * 0.1, 2.0 * np.pi * 0.25]
    h_ops = [0.5 * w * cudaq.spin.z(0) for w in omegas]
    t_end = 0.9
    schedule = Schedule(np.linspace(0.0, t_end, 19), ["t"])

    us = cudaq.contrib.propagator(h_ops, {0: 2}, schedule)

    assert len(us) == len(omegas)
    for u, w in zip(us, omegas):
        np.testing.assert_allclose(u, _expm_reference(0.5 * w * _Z, t_end),
                                   atol=1e-4)


@pytest.mark.skip(reason="TODO: time-dependent oracle (scipy ODE / QuTiP)")
def test_propagator_time_dependent():
    """U(t) for a time-dependent H vs an independent fine-grained integration.

    TODO: build the oracle by integrating dU/dt = -i H(t) U on the CPU with a
    fine scipy.integrate step (or qutip.propagator) and compare.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="TODO(api): operator adjoint + qutip oracle")
def test_propagator_open_system_dynamical_map():
    """Open-system dynamical map Phi(t) vs an independent master-equation solve.

    The open-system path is implemented (propagator with collapse_operators
    returns a d^2 x d^2 map), but two things must land first:
      1. TODO(api): the operator-adjoint helper (_dagger) needs a confirmed
         CUDA-Q adjoint API.
      2. Validate by *action*, not by raw matrix, to avoid any vec-convention
         mismatch: pick a test rho0, form rho_t = unvec(Phi @ vec(rho0)) and
         compare to qutip.mesolve(H, rho0, [0, t], c_ops).rho_final. Comparing
         Phi to qutip.propagator(...) directly would require matching column-
         vs row-stacking conventions.

    Example skeleton (amplitude damping on a qubit):
        gamma = 0.1
        H = 0.5 * (2*np.pi*0.1) * cudaq.spin.z(0)
        c = np.sqrt(gamma) * <annihilate(0)>           # collapse operator
        Phi = cudaq.contrib.propagator(H, {0: 2}, schedule,
                                       collapse_operators=[c])
        rho_t = (Phi @ rho0.reshape(-1)).reshape(2, 2)  # row-major unvec
        # compare rho_t to a qutip.mesolve reference
    """
    raise NotImplementedError
