# QubitOS Core

[![CI](https://github.com/qubit-os/qubit-os-core/actions/workflows/ci.yaml/badge.svg)](https://github.com/qubit-os/qubit-os-core/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Python modules for QubitOS - pulse optimization, calibration management, and quantum control.

## Overview

QubitOS Core provides:

- **pulsegen** - GRAPE/DRAG pulse optimization for quantum gates
- **calibrator** - Calibration data management and fitting
- **client** - gRPC client for the Hardware Abstraction Layer
- **cli** - Command-line interface for QubitOS

## Installation

```bash
# From PyPI (when available)
pip install qubitos

# From source
git clone https://github.com/qubit-os/qubit-os-core.git
cd qubit-os-core
pip install -e ".[dev]"
```

## Quick Start

### Generate an X-gate pulse

```python
from qubitos.pulsegen import generate_pulse
from qubitos.client import HALClient

# Generate optimized pulse
pulse = generate_pulse(
    gate="X",
    qubit=0,
    duration_ns=20,
    target_fidelity=0.999,
    algorithm="grape"
)

# Execute on simulator
async with HALClient("localhost:50051") as client:
    result = await client.execute_pulse(pulse, num_shots=1000)
    print(f"Counts: {result.bitstring_counts}")
    print(f"Fidelity: {result.fidelity_estimate:.4f}")
```

### CLI Usage

```bash
# Generate a pulse
qubit-os pulse generate --gate X --duration 20 --output x_gate.json

# Execute a pulse
qubit-os pulse execute x_gate.json --shots 1000

# Check backend health
qubit-os hal health

# Show calibration
qubit-os calibration show
```

## Architecture

```
qubitos/
├── client/         # HAL gRPC client
├── pulsegen/       # Pulse optimization (GRAPE, DRAG)
├── calibrator/     # Calibration management
├── validation/     # AgentBible integration
└── cli/            # Command-line interface
```

## Configuration

QubitOS uses a configuration hierarchy (later overrides earlier):

1. Built-in defaults
2. Environment variables (`QUBITOS_*`)
3. `config.yaml`
4. CLI arguments

```bash
# Environment variables
export QUBITOS_HAL_HOST=localhost
export QUBITOS_HAL_PORT=50051
export QUBITOS_LOG_LEVEL=info
```

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/qubit-os/qubit-os-core.git
cd qubit-os-core
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/qubitos/

# Linting
ruff check src/
ruff format src/
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=qubitos --cov-report=html

# Specific module
pytest tests/unit/test_pulsegen.py

# Integration tests (requires HAL running)
pytest tests/integration/
```

## Documentation

- [Design Document](docs/QubitOS-Design-v0.5.0.md)
- [CLI Reference](docs/cli-reference.md)
- [API Reference](docs/api/)
- [User Guide](docs/guides/)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
