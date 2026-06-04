# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Propagator computation for the CUDA-Q ``dynamics`` backend.

Compute the propagator ``U(t)`` of a (possibly time-dependent) Hamiltonian via
time-ordered exponential, analogous to QuTiP's ``propagator``. For a closed
system the propagator satisfies

    dU/dt = -i H(t) U(t),    U(0) = I,

which is integrated on the ``dynamics`` backend by encoding the right-hand side
as a ``cudaq.SuperOperator`` left-multiplication and evolving the identity as
the initial condition.

Note:
    The ``dynamics`` target requires an NVIDIA GPU (cuQuantum / cuDensityMat);
    there is no CPU code path. Items tagged ``TODO(gpu)`` below are values that
    must be confirmed on hardware (they cannot be exercised on a CPU-only box).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = ["propagator"]


def _total_dimension(dimensions: Mapping[int, int]) -> int:
    """Product of the per-degree Hilbert-space dimensions."""
    dim = 1
    for d in dimensions.values():
        dim *= d
    return dim


def _identity_initial_state(dim: int):
    """Build the U(0) = I initial condition for the dynamics solver.

    The master-equation solver consumes a density-matrix-shaped state as a
    flattened ``dim*dim`` array (cf. ``system_models.py`` where ``rho0`` is a
    ``cp.zeros(N*N)`` vector). ``vec(I)`` is identical under row- and
    column-major stacking, so the *initial* condition is unambiguous; only the
    *read-back* reshape (see ``_state_to_propagator``) is convention-sensitive.
    """
    import cudaq

    # TODO(gpu): confirm State.from_data accepts a host (numpy) array here, or
    # whether a device array (cupy) is required as in the dynamics tests.
    data = np.eye(dim, dtype=np.complex128).reshape(-1)
    return cudaq.State.from_data(data)


def _state_to_propagator(state, dim: int) -> np.ndarray:
    """Reshape an evolved dynamics state back into the ``dim x dim`` matrix U.

    TODO(gpu): confirm (a) how to extract the raw array from a ``cudaq.State``
    (``np.array(state)`` vs a tensor accessor), and (b) the reshape order
    (row- vs column-major), by checking U(t) ~= expm(-i H t) for a
    time-independent H. ``vec(I)`` does not disambiguate the convention, so the
    sanity check against ``scipy.linalg.expm`` is what pins it down.
    """
    arr = np.array(state)  # TODO(gpu): verify State -> ndarray extraction
    return arr.reshape(dim, dim)  # TODO(gpu): verify reshape order


def propagator(
    hamiltonian,
    dimensions: Mapping[int, int],
    schedule,
    *,
    collapse_operators: Sequence = (),
    integrator=None,
    store_intermediate_results: bool = False,
    max_batch_size: Optional[int] = None,
):
    """Compute the propagator ``U(t)`` of a Hamiltonian on the dynamics backend.

    Analogous to QuTiP's ``propagator``: integrates ``dU/dt = -i H(t) U`` with
    ``U(0) = I`` over the times in ``schedule`` using the ``dynamics`` target.

    Args:
        hamiltonian: A CUDA-Q ``Operator``, or a sequence of operators to
            compute a batch of propagators (e.g. a parameter sweep) in one call.
        dimensions: Mapping from degree-of-freedom index to Hilbert-space
            dimension, e.g. ``{0: 2}`` for a single qubit.
        schedule: A ``cudaq.Schedule`` of time points at which to evaluate U.
        collapse_operators: Optional Lindblad collapse operators. When provided,
            the result is the open-system dynamical map (a ``d^2 x d^2``
            superoperator) rather than a unitary. [stretch goal — see TODO]
        integrator: Optional ``BaseIntegrator``; defaults to the backend default.
        store_intermediate_results: If True, return U at every step of the
            schedule; otherwise return only the final ``U(t_end)``.
        max_batch_size: Optional cap on the batch size handled at once.

    Returns:
        Closed system: the propagator as a ``dim x dim`` complex
        ``numpy.ndarray``. With ``store_intermediate_results`` a list of such
        arrays over the schedule; with a batched ``hamiltonian`` a list over the
        batch.

    Note:
        Requires the ``dynamics`` target and an NVIDIA GPU.
    """
    import cudaq  # deferred import: avoid a circular import at package load

    if len(collapse_operators) > 0:
        # TODO(open-system): build the full Lindblad SuperOperator (left,
        # right, and left_right terms as in system_models.py:586-598) and
        # evolve the d^2 basis matrices (batched) from the identity map to
        # assemble the d^2 x d^2 dynamical map Phi(t). Validate against
        # qutip.propagator(H, t, c_ops). Targeted for this PR (no follow-up PR).
        raise NotImplementedError(
            "Open-system propagator (dynamical map) is not implemented yet.")

    is_batched = isinstance(hamiltonian, (list, tuple))
    hamiltonians = list(hamiltonian) if is_batched else [hamiltonian]

    dim = _total_dimension(dimensions)

    super_ops = []
    for h in hamiltonians:
        super_op = cudaq.SuperOperator()
        # dU/dt = -i H(t) U  ->  left-multiplication only (no commutator term).
        super_op += cudaq.SuperOperator.left_multiply(-1j * h)
        super_ops.append(super_op)

    initial_states = [_identity_initial_state(dim) for _ in super_ops]

    save_mode = (cudaq.IntermediateResultSave.ALL
                 if store_intermediate_results else
                 cudaq.IntermediateResultSave.NONE)

    results = cudaq.evolve(
        super_ops if is_batched else super_ops[0],
        dimensions,
        schedule,
        initial_states if is_batched else initial_states[0],
        collapse_operators=[],  # already empty; SuperOperator path forbids them
        store_intermediate_results=save_mode,
        integrator=integrator,
        max_batch_size=max_batch_size,
    )

    def _extract(result):
        if store_intermediate_results:
            # TODO(gpu): confirm the intermediate-states accessor name/shape.
            return [
                _state_to_propagator(s, dim)
                for s in result.intermediate_states()
            ]
        # TODO(gpu): confirm the final-state accessor (method vs property).
        return _state_to_propagator(result.final_state(), dim)

    if is_batched:
        return [_extract(r) for r in results]
    return _extract(results)
