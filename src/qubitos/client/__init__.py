# Copyright 2026 QubitOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HAL gRPC client for QubitOS.

This module provides the Python client for communicating with the
Hardware Abstraction Layer (HAL) via gRPC.

Example:
    >>> from qubitos.client import HALClient
    >>>
    >>> async with HALClient("localhost:50051") as client:
    ...     health = await client.health()
    ...     print(f"Backend status: {health.status}")
"""

# Client implementation will be added in Phase 1
__all__: list[str] = []
