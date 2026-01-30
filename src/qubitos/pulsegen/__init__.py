# Copyright 2026 QubitOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pulse optimization module for QubitOS.

This module provides GRAPE and DRAG pulse optimization algorithms
for quantum gate synthesis.

Submodules:
    grape: GRAPE (Gradient Ascent Pulse Engineering) optimizer
    hamiltonians: Hamiltonian construction and Pauli string parsing
    shapes: Standard pulse shapes (Gaussian, square, DRAG, etc.)

Example:
    >>> from qubitos.pulsegen import generate_pulse, GrapeConfig
    >>>
    >>> # Simple usage
    >>> result = generate_pulse(
    ...     gate="X",
    ...     num_qubits=1,
    ...     duration_ns=20,
    ...     target_fidelity=0.999
    ... )
    >>> print(f"Fidelity: {result.fidelity:.4f}")
    >>>
    >>> # Advanced usage with custom config
    >>> config = GrapeConfig(
    ...     num_time_steps=200,
    ...     max_iterations=2000,
    ...     learning_rate=0.05,
    ... )
    >>> result = generate_pulse("CZ", num_qubits=2, config=config)
"""

from .grape import (
    GateType,
    GrapeConfig,
    GrapeOptimizer,
    GrapeResult,
    generate_pulse,
)
from .hamiltonians import (
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_MATRICES,
    STANDARD_GATES,
    build_hamiltonian,
    embed_gate,
    get_target_unitary,
    parse_pauli_string,
    pauli_string_to_matrix,
    rotation_gate,
    tensor_product,
)
from .shapes import (
    PulseEnvelope,
    PulseShapeType,
    apply_window,
    cosine,
    drag,
    gaussian,
    gaussian_square,
    generate_envelope,
    sech,
    square,
)

__all__ = [
    # GRAPE
    "GateType",
    "GrapeConfig",
    "GrapeOptimizer",
    "GrapeResult",
    "generate_pulse",
    # Hamiltonians
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "PAULI_MATRICES",
    "STANDARD_GATES",
    "build_hamiltonian",
    "embed_gate",
    "get_target_unitary",
    "parse_pauli_string",
    "pauli_string_to_matrix",
    "rotation_gate",
    "tensor_product",
    # Shapes
    "PulseEnvelope",
    "PulseShapeType",
    "apply_window",
    "cosine",
    "drag",
    "gaussian",
    "gaussian_square",
    "generate_envelope",
    "sech",
    "square",
]
