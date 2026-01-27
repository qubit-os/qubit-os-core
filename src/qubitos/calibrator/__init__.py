# Copyright 2026 QubitOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Calibration management module for QubitOS.

This module handles loading, validation, and management of
calibration data for quantum backends.

Submodules:
    loader: Load calibration from YAML files
    fingerprint: Compute and validate calibration fingerprints
    fitting: T1/T2 and fidelity fitting routines

Example:
    >>> from qubitos.calibrator import load_calibration
    >>>
    >>> calibration = load_calibration("qutip_simulator")
    >>> print(f"T1 for Q0: {calibration.qubits['Q0'].t1.value_us} us")
"""

# Calibration implementation will be added in Phase 1
__all__: list[str] = []
