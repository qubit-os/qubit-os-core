# Copyright 2026 QubitOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Calibration management module for QubitOS.

This module handles loading, validation, and management of
calibration data for quantum backends.

Submodules:
    loader: Load calibration from YAML files
    fingerprint: Compute and validate calibration fingerprints

Example:
    >>> from qubitos.calibrator import load_calibration, CalibrationLoader
    >>>
    >>> # Simple loading
    >>> calibration = load_calibration("calibration/qutip_simulator.yaml")
    >>> print(f"T1 for qubit 0: {calibration.qubits[0].t1_us} us")
    >>>
    >>> # With loader for multiple backends
    >>> loader = CalibrationLoader(calibration_dir="./calibration")
    >>> cal = loader.load_for_backend("qutip_simulator")
    >>>
    >>> # Drift detection with fingerprints
    >>> from qubitos.calibrator import CalibrationFingerprint
    >>> fp = CalibrationFingerprint.from_calibration(calibration)
    >>> # ... later ...
    >>> new_fp = CalibrationFingerprint.from_calibration(new_calibration)
    >>> drift = fp.compare(new_fp)
    >>> if drift.needs_recalibration:
    ...     print(f"Recalibration needed: {drift.reason}")
"""

from .fingerprint import (
    CalibrationFingerprint,
    DriftMetrics,
    FingerprintConfig,
    FingerprintStore,
)
from .loader import (
    BackendCalibration,
    CalibrationError,
    CalibrationLoader,
    CouplerCalibration,
    QubitCalibration,
    get_default_loader,
    load_calibration,
)

__all__ = [
    # Loader
    "QubitCalibration",
    "CouplerCalibration",
    "BackendCalibration",
    "CalibrationError",
    "CalibrationLoader",
    "get_default_loader",
    "load_calibration",
    # Fingerprint
    "CalibrationFingerprint",
    "DriftMetrics",
    "FingerprintConfig",
    "FingerprintStore",
]
