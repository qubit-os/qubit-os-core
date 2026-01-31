# QubitOS Open-Source Quantum Control Kernel
## Design Document v0.5.0 – Complete Technical Specification

**Status:** Design Complete – Ready for Implementation  
**Author:** Rylan Malarchick  
**Last Updated:** January 26, 2026  
**License:** Apache 2.0  

---

## Table of Contents

1. [Scope and Non-Goals](#1-scope-and-non-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Protocol Buffers and Type Contracts](#3-protocol-buffers-and-type-contracts)
4. [Calibration Model and Policies](#4-calibration-model-and-policies)
5. [Data Flow and Initialization](#5-data-flow-and-initialization)
6. [Error Handling and Observability](#6-error-handling-and-observability)
7. [Testing and Reproducibility](#7-testing-and-reproducibility)
8. [Backend Implementation](#8-backend-implementation)
9. [Validation and AgentBible Integration](#9-validation-and-agentbible-integration)
10. [Deployment Model](#10-deployment-model)
11. [Cross-Repository Integration](#11-cross-repository-integration)
12. [Repository Structure](#12-repository-structure)
13. [Build and Distribution](#13-build-and-distribution)
14. [CI/CD Pipeline](#14-cicd-pipeline)
15. [CLI Specification](#15-cli-specification)
16. [REST API Specification](#16-rest-api-specification)
17. [Security Considerations](#17-security-considerations)
18. [Resource Limits](#18-resource-limits)
19. [Documentation Plan](#19-documentation-plan)
20. [Phase 0 Completion Criteria](#20-phase-0-completion-criteria)
21. [Appendix A: Technical Specifications](#appendix-a-technical-specifications)
22. [Appendix B: Default Configurations](#appendix-b-default-configurations)

---

## 1. Scope and Non-Goals

### 1.1 In-Scope for v0.1-alpha

- Single-qubit and two-qubit pulse optimization and execution
- Supported gates: X, Y, Z, SX, H (single-qubit); CZ, CNOT, iSWAP (two-qubit)
- Backends:
  - **QuTiP** simulator (default, fully offline)
  - **IQM Garnet** hardware (optional, cloud, Phase 1B)
- Deterministic, reproducible GRAPE/DRAG pulse optimization
- Versioned protocol contracts (Protocol Buffers) between all layers
- YAML-based calibration storage with explicit schema
- gRPC for all inter-service communication
- REST API facade with full OpenAPI specification
- Logging-only control loop (no active corrections)
- AgentBible integration for scientific validation

### 1.2 Explicit Non-Goals for v0.1-alpha

- Multi-qubit optimization beyond 2 qubits (>4-dimensional Hilbert space for gates)
- Distributed execution or job scheduling
- Adaptive, online calibration with active feedback
- Web UI (CLI and Jupyter notebooks only)
- Automatic backend fallback
- Real-time pulse streaming

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Layer                                 │
│   CLI (qubit-os) │ Jupyter Notebooks │ Python Scripts               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Pulse Optimization    │     │  Calibration Management │
│   (qubitos.pulsegen)    │     │  (qubitos.calibrator)   │
│   - GRAPE optimizer     │     │  - T1/T2 fitting        │
│   - DRAG pulses         │     │  - Fingerprinting       │
│   - Validation          │     │  - Policy enforcement   │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │ gRPC
                            ▼
              ┌─────────────────────────┐
              │  Hardware Abstraction   │
              │  Layer (HAL)            │
              │  - Rust + tonic         │
              │  - Backend registry     │
              │  - Request validation   │
              └───────────┬─────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  QuTiP   │   │   IQM    │   │  Future  │
    │ Backend  │   │ Backend  │   │ Backends │
    └──────────┘   └──────────┘   └──────────┘
```

### 2.2 Communication Protocols

| Layer | Protocol | Format |
|-------|----------|--------|
| User → Core | Python API | Native objects |
| Core → HAL | gRPC | Protocol Buffers |
| HAL → Backends | Backend-specific | Varies |
| REST Facade | HTTP/1.1 + HTTP/2 | JSON |
| Configuration | File-based | YAML |
| Calibration | File-based | YAML |

### 2.3 Design Principles

1. **Single Responsibility** – Each module has one well-defined job
2. **Strict Contracts** – All inter-module communication uses versioned protos
3. **Backend Independence** – Backends implement a common trait; no special-casing
4. **Reproducibility First** – Every result traces to code version, seed, and calibration
5. **Fail Loud** – Clear error semantics; no silent failures
6. **Science-Aware Validation** – Domain validators catch physics errors early
7. **Documentation as Code** – Specs are machine-readable where possible

---

## 3. Protocol Buffers and Type Contracts

### 3.1 Proto Package Structure

```
qubit-os-proto/
├── quantum/
│   ├── common/
│   │   └── v1/
│   │       └── common.proto      # Shared types (Error, Timestamp, etc.)
│   ├── pulse/
│   │   └── v1/
│   │       ├── hamiltonian.proto # HamiltonianSpec
│   │       ├── pulse.proto       # PulseShape
│   │       └── grape.proto       # GRAPE request/response
│   └── backend/
│       └── v1/
│           ├── service.proto     # QuantumBackend service
│           ├── execution.proto   # ExecutePulse messages
│           └── hardware.proto    # HardwareInfo, Health
└── buf.yaml
```

### 3.2 Versioning Policy

- Proto packages use path-based versioning: `quantum.pulse.v1`, `quantum.pulse.v2`
- `proto_version` field in messages is informational only
- Breaking changes require a new version namespace
- Old versions supported for minimum 2 minor releases after deprecation

### 3.3 Common Types

```protobuf
syntax = "proto3";
package quantum.common.v1;

message TraceContext {
  string trace_id = 1;           // UUID, generated at request origin
  string span_id = 2;            // Optional, for distributed tracing
  string parent_span_id = 3;     // Optional
}

message Timestamp {
  int64 seconds = 1;             // Unix epoch seconds
  int32 nanos = 2;               // Nanosecond offset
}

message Error {
  enum Severity {
    INFO = 0;
    WARNING = 1;
    DEGRADED = 2;
    FATAL = 3;
  }

  int32 code = 1;                // gRPC status code
  Severity severity = 2;
  string message = 3;
  string details = 4;
  string trace_id = 5;
  Timestamp timestamp = 6;
}
```

### 3.4 Hamiltonian Specification

```protobuf
syntax = "proto3";
package quantum.pulse.v1;

message HamiltonianSpec {
  enum RepresentationFormat {
    PAULI_STRING = 0;
    MATRIX_SPARSE = 1;
    MATRIX_DENSE = 2;
  }

  RepresentationFormat format = 1;
  string content = 2;              // Format-specific payload (see Appendix A.1)
  int32 hilbert_space_dim = 3;     // e.g., 2 for 1-qubit, 4 for 2-qubit
  int32 num_qubits = 4;            // Number of qubits in system
  double validation_tolerance = 5; // For Hermiticity checks (default: 1e-10)
  
  // Control Hamiltonians (for GRAPE)
  repeated string control_operators = 6;  // Pauli strings for control terms
}
```

**Pauli String Format:** See [Appendix A.1](#a1-pauli-string-grammar) for formal grammar.

**Matrix Sparse Format:** See [Appendix A.2](#a2-matrix-sparse-format) for JSON schema.

### 3.5 Pulse Specification

```protobuf
syntax = "proto3";
package quantum.pulse.v1;

import "quantum/common/v1/common.proto";

message PulseShape {
  // Identification
  string pulse_id = 1;                    // UUID

  // Gate specification
  string algorithm = 2;                   // "grape", "drag", "gaussian", "square"
  GateType gate_type = 3;
  repeated int32 target_qubit_indices = 4; // e.g., [0] or [0, 1]
  double target_fidelity = 5;

  // Time structure
  int32 duration_ns = 6;
  int32 num_time_steps = 7;
  double time_step_ns = 8;                // Computed: duration_ns / num_time_steps

  // Waveform data (piecewise-constant)
  repeated double i_envelope = 9;         // In-phase, length == num_time_steps
  repeated double q_envelope = 10;        // Quadrature, length == num_time_steps
  double max_amplitude_mhz = 11;

  // For two-qubit gates: coupling pulse
  repeated double coupling_envelope = 12; // Optional, for tunable couplers

  // Validation
  bool validated = 13;
  string validation_error = 14;

  // Provenance
  int32 proto_version = 15;
  quantum.common.v1.Timestamp created_at = 16;
  string calibration_fingerprint = 17;
  string code_version = 18;               // Git SHA or semver
  int32 random_seed = 19;                 // Seed used for optimization
}

enum GateType {
  GATE_UNSPECIFIED = 0;
  // Single-qubit gates
  GATE_X = 1;
  GATE_Y = 2;
  GATE_Z = 3;
  GATE_SX = 4;        // sqrt(X)
  GATE_H = 5;         // Hadamard
  GATE_RX = 6;        // Rotation around X
  GATE_RY = 7;        // Rotation around Y
  GATE_RZ = 8;        // Rotation around Z (virtual, no pulse)
  // Two-qubit gates
  GATE_CZ = 10;
  GATE_CNOT = 11;
  GATE_ISWAP = 12;
  GATE_SQISWAP = 13;  // sqrt(iSWAP)
  // Custom
  GATE_CUSTOM = 99;
}
```

**Validation Rules:**
- `num_time_steps > 0`
- `duration_ns > 0`
- `|time_step_ns - (duration_ns / num_time_steps)| < 1e-9`
- `len(i_envelope) == len(q_envelope) == num_time_steps`
- `|i_envelope[k]| <= max_amplitude_mhz` and `|q_envelope[k]| <= max_amplitude_mhz`
- `len(target_qubit_indices) >= 1`
- For 2Q gates: `len(target_qubit_indices) == 2`

### 3.6 GRAPE Optimization

```protobuf
syntax = "proto3";
package quantum.pulse.v1;

import "quantum/common/v1/common.proto";
import "quantum/pulse/v1/hamiltonian.proto";
import "quantum/pulse/v1/pulse.proto";

message GRAPERequest {
  quantum.common.v1.TraceContext trace = 1;
  
  // System specification
  HamiltonianSpec system_hamiltonian = 2;
  GateType target_gate = 3;
  repeated int32 target_qubit_indices = 4;
  
  // Optimization parameters
  double target_fidelity = 5;             // e.g., 0.999
  int32 max_iterations = 6;               // e.g., 1000
  double learning_rate = 7;               // e.g., 0.01
  int32 random_seed = 8;
  
  // Time discretization
  int32 num_time_steps = 9;               // e.g., 100
  int32 duration_ns = 10;                 // e.g., 20
  
  // Advanced options
  GRAPEOptions options = 11;
  
  // Request metadata
  int32 timeout_ms = 12;                  // 0 = use default
  string calibration_fingerprint = 13;    // Required calibration version
}

message GRAPEOptions {
  // Optimizer
  string optimizer = 1;                   // "adam", "lbfgs", "sgd" (default: "adam")
  int32 lbfgs_memory = 2;                 // For L-BFGS (default: 10)
  
  // Learning rate schedule
  double learning_rate_decay = 3;         // e.g., 0.95
  int32 decay_interval = 4;               // Iterations between decay (default: 50)
  
  // Regularization
  double l2_amplitude_penalty = 5;        // Penalize large amplitudes
  double smoothness_penalty = 6;          // Penalize rapid changes
  double bandwidth_limit_mhz = 7;         // Frequency cutoff (0 = disabled)
  
  // Convergence
  double convergence_threshold = 8;       // Fidelity change threshold (default: 1e-8)
  int32 convergence_window = 9;           // Iterations to check (default: 10)
  double gradient_clip_norm = 10;         // Max gradient norm (default: 1.0)
  
  // Initial guess
  string initial_pulse_id = 11;           // Start from existing pulse (optional)
  
  // Noise modeling
  bool include_decoherence = 12;          // Use T1/T2 from calibration
  bool include_leakage = 13;              // Model leakage to non-computational states
}

message GRAPEResponse {
  quantum.common.v1.TraceContext trace = 1;
  
  bool success = 2;
  quantum.common.v1.Error error = 3;
  
  // Result
  PulseShape optimized_pulse = 4;
  double achieved_fidelity = 5;
  
  // Convergence info
  int32 iterations_used = 6;
  string convergence_reason = 7;          // "target_reached", "max_iterations", "stalled", "cancelled"
  
  // Diagnostics
  repeated double fidelity_history = 8;
  repeated double gradient_norms = 9;
  double final_regularization_cost = 10;
  int64 wall_time_ms = 11;
  
  // Warnings
  repeated string warnings = 12;
}

// Cancellation
message CancelGRAPERequest {
  string trace_id = 1;
}

message CancelGRAPEResponse {
  bool cancelled = 1;
  GRAPEResponse partial_result = 2;       // Best result so far, if available
}
```

### 3.7 Backend Service

```protobuf
syntax = "proto3";
package quantum.backend.v1;

import "quantum/common/v1/common.proto";
import "quantum/pulse/v1/pulse.proto";

service QuantumBackend {
  // Pulse execution
  rpc ExecutePulse(ExecutePulseRequest) returns (ExecutePulseResponse);
  rpc ExecutePulseBatch(ExecutePulseBatchRequest) returns (ExecutePulseBatchResponse);
  
  // System info
  rpc GetHardwareInfo(GetHardwareInfoRequest) returns (HardwareInfo);
  rpc Health(HealthRequest) returns (HealthResponse);
  
  // Streaming (future)
  // rpc StreamPulses(stream PulseShape) returns (stream MeasurementResult);
}

message ExecutePulseRequest {
  quantum.common.v1.TraceContext trace = 1;
  
  string backend_name = 2;
  quantum.pulse.v1.PulseShape pulse = 3;
  int32 num_shots = 4;
  string measurement_basis = 5;           // "z", "x", "y"
  repeated int32 measurement_qubits = 6;  // Which qubits to measure
  
  // Options
  bool return_state_vector = 7;           // Simulator only
  bool include_noise = 8;                 // Use calibrated noise model
  int32 timeout_ms = 9;
}

message ExecutePulseResponse {
  quantum.common.v1.TraceContext trace = 1;
  
  bool success = 2;
  quantum.common.v1.Error error = 3;
  
  MeasurementResult result = 4;
  repeated string warnings = 5;
}

message ExecutePulseBatchRequest {
  quantum.common.v1.TraceContext trace = 1;
  
  repeated ExecutePulseRequest requests = 2;
  bool stop_on_first_error = 3;           // Abort batch on first failure
}

message ExecutePulseBatchResponse {
  quantum.common.v1.TraceContext trace = 1;
  
  repeated ExecutePulseResponse responses = 2;
  int32 successful_count = 3;
  int32 failed_count = 4;
}

message MeasurementResult {
  enum Quality {
    QUALITY_UNSPECIFIED = 0;
    FULL_SUCCESS = 1;
    DEGRADED = 2;
    PARTIAL_FAILURE = 3;
    TOTAL_FAILURE = 4;
  }

  // Counts
  map<string, int32> bitstring_counts = 1;  // e.g., {"00": 450, "01": 50, "10": 50, "11": 450}
  int32 total_shots = 2;
  int32 successful_shots = 3;
  Quality quality = 4;

  // Fidelity estimate (if computable)
  double fidelity_estimate = 5;
  string fidelity_method = 6;             // "state_tomography", "direct_comparison", etc.

  // Metadata
  string backend_name = 7;
  quantum.common.v1.Timestamp measured_at = 8;
  string calibration_fingerprint = 9;

  // Optional state vector (simulator only, if requested)
  StateVector state_vector = 10;

  // Noise info used
  NoiseParameters noise_applied = 11;
}

message StateVector {
  // Complex amplitudes as interleaved real/imag pairs
  // For n qubits: length = 2 * 2^n
  repeated double amplitudes = 1;
  int32 num_qubits = 2;
}

message NoiseParameters {
  double t1_us = 1;
  double t2_us = 2;
  double readout_error = 3;
  double gate_error = 4;
  double thermal_population = 5;          // Excited state population at equilibrium
}
```

### 3.8 Hardware Info and Health

```protobuf
message GetHardwareInfoRequest {
  string backend_name = 1;
}

message HardwareInfo {
  string backend_name = 1;
  string backend_type = 2;                // "simulator" or "hardware"
  string tier = 3;                        // "local", "cloud"

  // Capabilities
  int32 num_qubits = 4;
  repeated int32 available_qubit_indices = 5;
  repeated quantum.pulse.v1.GateType supported_gates = 6;
  repeated string supported_algorithms = 7;  // "grape", "drag", etc.
  bool supports_state_vector = 8;
  bool supports_noise_model = 9;

  // Connectivity (for 2Q gates)
  repeated QubitPair connectivity = 10;

  // Performance hints
  PerformanceHints performance = 11;

  // Resource limits
  ResourceLimits limits = 12;

  // Authentication
  bool requires_auth = 13;

  // Version info
  string software_version = 14;           // e.g., "qutip-5.0.0" or "iqm-client-1.2.3"
  int32 proto_version = 15;

  // Validation status
  ValidationStatus validation = 16;
}

message QubitPair {
  int32 qubit_a = 1;
  int32 qubit_b = 2;
  repeated quantum.pulse.v1.GateType supported_gates = 3;
}

message PerformanceHints {
  double typical_latency_ms = 1;          // P50
  double p95_latency_ms = 2;
  double p99_latency_ms = 3;
  int32 max_shots_per_request = 4;
  int32 recommended_batch_size = 5;
}

message ResourceLimits {
  int32 max_hilbert_dim = 1;              // Max supported dimension
  int32 max_qubits = 2;
  int32 max_shots = 3;
  int32 max_pulse_duration_ns = 4;
  int32 max_time_steps = 5;
  int32 max_batch_size = 6;
  int32 max_concurrent_requests = 7;
}

message ValidationStatus {
  enum Status {
    NOT_VALIDATED = 0;
    PASSED = 1;
    FAILED = 2;
    EXPIRED = 3;
  }

  Status status = 1;
  string method = 2;                      // e.g., "hellinger_crosscheck_v1"
  quantum.common.v1.Timestamp validated_at = 3;
  string details = 4;
}

message HealthRequest {
  string backend_name = 1;                // Empty = all backends
}

message HealthResponse {
  enum Status {
    HEALTHY = 0;
    DEGRADED = 1;
    UNAVAILABLE = 2;
  }

  Status status = 1;
  string message = 2;
  quantum.common.v1.Timestamp checked_at = 3;
  double latency_ms = 4;

  // Per-backend status (if backend_name was empty)
  map<string, Status> backend_statuses = 5;
}
```

**Health Thresholds:**
- `HEALTHY`: latency < 2s, no errors
- `DEGRADED`: 2s <= latency < 10s, or intermittent errors
- `UNAVAILABLE`: latency >= 10s, or 3+ consecutive failures

---

## 4. Calibration Model and Policies

### 4.1 Calibration File Structure

```
qubit-os-core/
└── calibration/
    ├── defaults/
    │   ├── qutip_simulator.yaml      # Shipped default
    │   └── iqm_garnet_template.yaml  # Template for IQM
    ├── qutip_simulator/
    │   ├── 2026-01-25T18-55-00Z.yaml
    │   ├── 2026-01-26T10-30-00Z.yaml
    │   └── current -> 2026-01-26T10-30-00Z.yaml
    └── iqm_garnet/
        └── ...
```

**File Naming Convention:**
- Format: `{ISO8601-timestamp}.yaml` with colons replaced by hyphens
- Symlink `current` points to active calibration
- Never auto-deleted; manual cleanup only

### 4.2 Calibration Schema

```yaml
# calibration/qutip_simulator/2026-01-26T10-30-00Z.yaml
schema_version: "1.0"

metadata:
  backend: qutip_simulator
  created_at: "2026-01-26T10:30:00Z"
  fingerprint: "sha256:a1b2c3d4e5f6..."  # First 16 hex chars of SHA256
  source: "default"                       # "default", "measured", "imported"
  notes: "Default calibration for QuTiP simulator"

system:
  num_qubits: 2
  qubit_labels: ["Q0", "Q1"]
  connectivity:
    - [0, 1]  # Q0-Q1 connected

qubits:
  Q0:
    frequency_ghz: 4.8734
    anharmonicity_mhz: -200.0
    
    t1:
      value_us: 45.2
      uncertainty_us: 2.1
      measured_at: "2026-01-26T10:30:00Z"
      method: "exponential_decay"
      raw_data:
        times_us: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        populations: [0.98, 0.95, 0.88, 0.76, 0.57, 0.33]
      fit:
        model: "A * exp(-t/T1) + C"
        parameters:
          A: 0.99
          T1: 45.2
          C: 0.01
        r_squared: 0.9971
        residuals: [0.002, -0.001, 0.0005, -0.002, 0.0015, -0.001]

    t2:
      value_us: 32.5
      uncertainty_us: 1.8
      measured_at: "2026-01-26T10:30:00Z"
      method: "ramsey"
      detuning_mhz: 0.5
      raw_data:
        times_us: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        populations: [0.95, 0.90, 0.78, 0.52, 0.28, 0.12]
      fit:
        model: "A * exp(-t/T2) * cos(2*pi*f*t + phi) + C"
        parameters:
          A: 0.98
          T2: 32.5
          f: 0.5
          phi: 0.1
          C: 0.02
        r_squared: 0.9945

    readout:
      fidelity: 0.9785
      measured_at: "2026-01-26T10:30:00Z"
      method: "confusion_matrix"
      num_shots: 10000
      confusion_matrix:
        p00: 0.976   # P(measure 0 | prepared 0)
        p01: 0.024   # P(measure 1 | prepared 0)
        p10: 0.019   # P(measure 0 | prepared 1)
        p11: 0.981   # P(measure 1 | prepared 1)

    single_qubit_gates:
      X:
        fidelity: 0.9991
        method: "randomized_benchmarking"
        num_cliffords: [1, 2, 4, 8, 16, 32, 64]
        num_sequences: 100
        error_per_clifford: 0.00045
      SX:
        fidelity: 0.9993
        method: "randomized_benchmarking"
        error_per_clifford: 0.00035

  Q1:
    frequency_ghz: 5.1023
    anharmonicity_mhz: -195.0
    t1:
      value_us: 42.1
      # ... similar structure
    t2:
      value_us: 29.8
      # ...
    readout:
      fidelity: 0.9812
      # ...

two_qubit_gates:
  Q0_Q1:
    CZ:
      fidelity: 0.982
      method: "interleaved_randomized_benchmarking"
      gate_time_ns: 40
      measured_at: "2026-01-26T10:30:00Z"
    iSWAP:
      fidelity: 0.978
      method: "interleaved_randomized_benchmarking"
      gate_time_ns: 35

crosstalk:
  Q0_to_Q1:
    zz_coupling_khz: 25.3
    static_zz_khz: 12.1
```

### 4.3 Calibration Fingerprint Algorithm

```python
import hashlib
import yaml

def compute_fingerprint(calibration_data: dict) -> str:
    """Compute calibration fingerprint.
    
    1. Remove metadata.fingerprint field (avoid self-reference)
    2. Serialize to canonical YAML (sorted keys, no flow style)
    3. Compute SHA256
    4. Return first 16 hex characters with prefix
    """
    data = deep_copy(calibration_data)
    if 'metadata' in data:
        data['metadata'].pop('fingerprint', None)
    
    canonical = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True
    )
    
    digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return f"sha256:{digest[:16]}"
```

### 4.4 Calibration Policies

**Drift Detection:**
- Baseline: the `current` calibration symlink target
- Drift: percentage change from baseline value
- Threshold: 1% relative change in any of {T1, T2, gate_fidelity, readout_fidelity}

**Fingerprint Mismatch Handling:**

| Drift | Action |
|-------|--------|
| <= 1% | Warning; execute with `Quality = DEGRADED` |
| > 1% | Reject with `INVALID_ARGUMENT`; require explicit override |

**Override Flag:**
```protobuf
message ExecutePulseRequest {
  // ... existing fields
  bool allow_calibration_mismatch = 10;  // If true, execute anyway with warning
}
```

### 4.5 Missing Calibration Behavior

On HAL startup:
1. Check for `calibration/{backend}/current` symlink
2. If exists: load and validate
3. If missing: copy from `calibration/defaults/{backend}.yaml`
4. If default also missing: fail startup with clear error

---

## 5. Data Flow and Initialization

### 5.1 Request Flow

```
User Request
    │
    ▼
┌─────────────────┐
│ Python Client   │ ─── Validate inputs locally
└────────┬────────┘
         │ gRPC
         ▼
┌─────────────────┐
│ HAL Server      │ ─── Validate proto, check calibration fingerprint
├─────────────────┤
│ Backend Router  │ ─── Select backend, check health
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Backend         │ ─── Execute pulse, apply noise model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Result          │ ─── Package result, attach metadata
└─────────────────┘
```

### 5.2 Initialization Sequence

```
1. HAL Start
   ├── Load config (defaults → env → yaml → cli)
   ├── Validate config
   ├── Load calibration (current or default)
   ├── Initialize backends
   │   ├── QuTiP: always
   │   └── IQM: if credentials present
   ├── Start gRPC server
   └── Start REST server (if enabled)

2. Backend Health Check (per backend)
   ├── Run health probe
   ├── Record latency
   └── Set initial status

3. Ready
   └── Accept requests
```

### 5.3 Configuration Hierarchy

Priority (highest wins):
1. CLI arguments
2. Environment variables
3. `config.yaml`
4. Built-in defaults

**Environment Variables:**
```bash
# IQM Backend
IQM_GATEWAY_URL=https://cocos.iqm.fi
IQM_AUTH_TOKEN=...

# Server
QUBITOS_HAL_HOST=0.0.0.0
QUBITOS_HAL_GRPC_PORT=50051
QUBITOS_HAL_REST_PORT=8080
QUBITOS_LOG_LEVEL=info

# Validation
QUBITOS_STRICT_VALIDATION=true
```

---

## 6. Error Handling and Observability

### 6.1 Error Matrix

| gRPC Code | Name | Severity | Client Action |
|-----------|------|----------|---------------|
| 0 | OK | INFO | Continue |
| 3 | INVALID_ARGUMENT | FATAL | Fix input; don't retry |
| 4 | DEADLINE_EXCEEDED | DEGRADED | Retry with backoff (max 3) |
| 5 | NOT_FOUND | FATAL | Check backend name |
| 7 | PERMISSION_DENIED | FATAL | Check credentials |
| 8 | RESOURCE_EXHAUSTED | DEGRADED | Wait, retry |
| 9 | FAILED_PRECONDITION | FATAL | Check calibration |
| 13 | INTERNAL | FATAL | Log, investigate |
| 14 | UNAVAILABLE | DEGRADED | Retry with backoff |
| 16 | UNAUTHENTICATED | FATAL | Refresh credentials |

### 6.2 Logging Schema

```json
{
  "timestamp": "2026-01-26T10:30:00.123456Z",
  "level": "INFO",
  "module": "hal.backend.qutip",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "message": "Pulse execution completed",
  "context": {
    "backend": "qutip_simulator",
    "pulse_id": "550e8400-e29b-41d4-a716-446655440000",
    "num_shots": 1000,
    "duration_ms": 152.3,
    "fidelity": 0.9987,
    "calibration_fingerprint": "sha256:a1b2c3d4..."
  }
}
```

**Log Levels:**
- `DEBUG`: Detailed internal state (development only)
- `INFO`: Normal operations (pulse execution, calibration load)
- `WARNING`: Degraded operation (calibration drift, retry)
- `ERROR`: Failed operation (backend error, validation failure)

### 6.3 Log Rotation

```yaml
# config.yaml
logging:
  directory: "./logs"
  max_size_mb: 100
  max_files: 10
  compress: true
```

### 6.4 Graceful Shutdown

```
SIGTERM received
    │
    ├── Stop accepting new requests
    ├── Wait for in-flight requests (timeout: 30s)
    ├── Cancel long-running operations
    ├── Flush logs
    └── Exit
```

---

## 7. Testing and Reproducibility

### 7.1 Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| HAL (Rust) | >= 85% | Critical path |
| Proto validation | 100% | All message types |
| pulsegen | >= 75% | Math-heavy |
| calibrator | >= 80% | Fitting logic |
| CLI | >= 70% | User-facing |

### 7.2 Reproducibility Tiers

**Tier 1 (Required):** Deterministic
- Same seed + code version + calibration = identical results
- Verified by golden file tests

**Tier 2 (Expected):** Stable
- Same seed + different code version = fidelity within 0.1%
- Verified by regression tests

**Tier 3 (Statistical):** Consistent
- Different seeds = statistically equivalent distributions
- Verified by Kolmogorov-Smirnov test (p > 0.05)

### 7.3 Test Data Management

```
qubit-os-core/
└── tests/
    ├── fixtures/
    │   ├── calibrations/
    │   │   └── test_calibration.yaml
    │   ├── pulses/
    │   │   └── golden_x_gate.json
    │   └── hamiltonians/
    │       └── transmon_2q.json
    └── golden/
        ├── grape_x_gate_seed42.json
        └── qutip_counts_seed42.json
```

### 7.4 Fidelity Definition

QubitOS uses **average gate fidelity**:

```
F_avg = (d * F_pro + 1) / (d + 1)

where:
  d = Hilbert space dimension (2^n for n qubits)
  F_pro = |Tr(U_target^† @ U_actual)|² / d²
```

This is the standard definition from Nielsen (2002) and enables comparison with published benchmarks.

---

## 8. Backend Implementation

### 8.1 Backend Trait (Rust)

```rust
#[async_trait]
pub trait QuantumBackend: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> BackendType;
    
    async fn execute_pulse(
        &self,
        request: ExecutePulseRequest,
    ) -> Result<MeasurementResult, BackendError>;
    
    async fn execute_batch(
        &self,
        request: ExecutePulseBatchRequest,
    ) -> Result<ExecutePulseBatchResponse, BackendError>;
    
    async fn get_hardware_info(&self) -> Result<HardwareInfo, BackendError>;
    
    async fn health_check(&self) -> Result<HealthResponse, BackendError>;
    
    fn resource_limits(&self) -> &ResourceLimits;
}

pub enum BackendType {
    Simulator,
    Hardware,
}
```

### 8.2 QuTiP Backend

- Calls QuTiP via PyO3 (embedded Python)
- Uses `mesolve` with Lindblad operators for noise
- Pinned version: `qutip >= 5.0.0`
- State vector extraction for `return_state_vector = true`

### 8.3 IQM Backend

- HTTP client to IQM Resonance API
- Token from `IQM_AUTH_TOKEN` environment variable
- Retry logic: exponential backoff, max 3 retries
- Timeout: 30s per request

### 8.4 Backend Registry

```rust
pub struct BackendRegistry {
    backends: HashMap<String, Arc<dyn QuantumBackend>>,
}

impl BackendRegistry {
    pub fn get(&self, name: &str) -> Option<Arc<dyn QuantumBackend>>;
    pub fn list(&self) -> Vec<&str>;
    pub fn health_all(&self) -> HashMap<String, HealthStatus>;
}
```

---

## 9. Validation and AgentBible Integration

### 9.1 Validation Framework

AgentBible quantum validators are implemented upstream in `research-code-principles` and consumed by QubitOS.

**Validator Categories:**

| Category | Checks |
|----------|--------|
| Hamiltonian | Hermiticity, dimension, spectrum bounds |
| Pulse | Time consistency, amplitude bounds, smoothness |
| Calibration | T1 > T2, monotone decay, fit quality |
| Fidelity | Range [0, 1], consistency |
| Numerical | NaN/Inf detection, condition number |

### 9.2 Strictness Configuration

```python
# Environment variable
QUBITOS_STRICT_VALIDATION=true  # Exceptions on failure (default)
QUBITOS_STRICT_VALIDATION=false # Warnings only
```

```python
# Programmatic
from qubitos.validation import set_strictness, Strictness

set_strictness(Strictness.STRICT)   # Raise ValidationError
set_strictness(Strictness.LENIENT)  # Log warning, continue
```

### 9.3 Provenance Tracking

Every optimization and execution attaches:
- `code_version`: Git SHA or package version
- `random_seed`: RNG seed used
- `calibration_fingerprint`: Calibration snapshot ID
- `timestamp`: UTC timestamp
- `trace_id`: Request correlation ID

---

## 10. Deployment Model

### 10.1 Development Mode

```bash
# Start HAL
qubit-os hal start --config config.yaml

# Or with environment
QUBITOS_LOG_LEVEL=debug qubit-os hal start
```

### 10.2 Docker Deployment

```yaml
# docker-compose.yaml
version: "3.8"

services:
  hal:
    image: ghcr.io/qubit-os/qubit-os-hardware:latest
    ports:
      - "50051:50051"  # gRPC
      - "8080:8080"    # REST
    environment:
      - QUBITOS_LOG_LEVEL=info
      - IQM_GATEWAY_URL
      - IQM_AUTH_TOKEN
    volumes:
      - ./calibration:/app/calibration
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50051"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 10.3 Port Configuration

| Service | Default Port | Environment Variable |
|---------|--------------|---------------------|
| gRPC | 50051 | `QUBITOS_HAL_GRPC_PORT` |
| REST | 8080 | `QUBITOS_HAL_REST_PORT` |
| Metrics | 9090 | `QUBITOS_METRICS_PORT` |

### 10.4 Performance Targets

**QuTiP Backend:**
- P50 latency: 100-200ms (single pulse, 1000 shots)
- P95 latency: < 500ms
- Memory: < 4GB for 4-qubit simulations

**IQM Backend:**
- P50 latency: 1-3s
- P95 latency: < 10s
- Timeout: 30s

---

## 11. Cross-Repository Integration

### 11.1 Repository Dependency Graph

```
qubit-os-proto
    │
    ├──► qubit-os-hardware (Rust protos)
    │
    └──► qubit-os-core (Python protos)
              │
              └──► agentbible (validation)
```

### 11.2 Version Compatibility Matrix

Each release documents compatible versions:

```yaml
# qubit-os-core/compatibility.yaml
qubit-os-core: "0.1.0"
compatible:
  qubit-os-proto: ">=0.1.0,<0.2.0"
  qubit-os-hardware: ">=0.1.0,<0.2.0"
  agentbible: ">=1.0.0"
```

### 11.3 External Dependencies

| Dependency | Purpose | Repo |
|------------|---------|------|
| AgentBible | Validation framework | research-code-principles |
| QubitPulseOpt | GRAPE reference | QubitPulseOpt |

---

## 12. Repository Structure

### 12.1 qubit-os-proto

```
qubit-os-proto/
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── release.yaml
├── quantum/
│   ├── common/v1/
│   │   └── common.proto
│   ├── pulse/v1/
│   │   ├── hamiltonian.proto
│   │   ├── pulse.proto
│   │   └── grape.proto
│   └── backend/v1/
│       ├── service.proto
│       ├── execution.proto
│       └── hardware.proto
├── generated/
│   ├── python/
│   │   └── quantum/
│   └── rust/
│       └── src/
├── buf.yaml
├── buf.gen.yaml
├── Cargo.toml          # For Rust crate
├── pyproject.toml      # For Python package
├── README.md
├── LICENSE
└── CHANGELOG.md
```

### 12.2 qubit-os-hardware

```
qubit-os-hardware/
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── release.yaml
├── src/
│   ├── lib.rs
│   ├── main.rs           # HAL binary entry point
│   ├── config.rs
│   ├── server/
│   │   ├── mod.rs
│   │   ├── grpc.rs
│   │   └── rest.rs
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── trait.rs
│   │   ├── registry.rs
│   │   ├── qutip/
│   │   │   ├── mod.rs
│   │   │   └── executor.rs
│   │   └── iqm/
│   │       ├── mod.rs
│   │       └── client.rs
│   ├── validation/
│   │   ├── mod.rs
│   │   ├── pulse.rs
│   │   └── hamiltonian.rs
│   └── error.rs
├── tests/
│   ├── integration/
│   └── fixtures/
├── Cargo.toml
├── Cargo.lock
├── config.example.yaml
├── Dockerfile
├── README.md
├── LICENSE
└── CHANGELOG.md
```

### 12.3 qubit-os-core

```
qubit-os-core/
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── release.yaml
├── src/
│   └── qubitos/
│       ├── __init__.py
│       ├── py.typed
│       ├── client/
│       │   ├── __init__.py
│       │   └── hal.py        # gRPC client
│       ├── pulsegen/
│       │   ├── __init__.py
│       │   ├── grape.py
│       │   ├── drag.py
│       │   └── shapes.py
│       ├── calibrator/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── fingerprint.py
│       │   └── fitting.py
│       ├── validation/
│       │   ├── __init__.py
│       │   └── quantum.py    # Wraps AgentBible
│       └── cli/
│           ├── __init__.py
│           └── main.py
├── calibration/
│   └── defaults/
│       ├── qutip_simulator.yaml
│       └── iqm_garnet_template.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── golden/
├── docs/
│   ├── api/
│   │   └── openapi.yaml
│   ├── guides/
│   └── specs/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

## 13. Build and Distribution

### 13.1 Proto Generation

```yaml
# buf.gen.yaml
version: v1
plugins:
  # Python
  - plugin: buf.build/protocolbuffers/python
    out: generated/python
  - plugin: buf.build/grpc/python
    out: generated/python
  
  # Rust (via prost)
  - plugin: buf.build/community/neoeinstein-prost
    out: generated/rust/src
    opt:
      - compile_well_known_types
  - plugin: buf.build/community/neoeinstein-tonic
    out: generated/rust/src
```

```bash
# Generate all
buf generate

# Lint protos
buf lint
```

### 13.2 Python Packaging

```toml
# pyproject.toml (qubit-os-core)
[project]
name = "qubitos"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "grpcio>=1.60.0",
    "protobuf>=4.25.0",
    "numpy>=1.26.0",
    "qutip>=5.0.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "agentbible>=1.0.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]

[project.scripts]
qubit-os = "qubitos.cli.main:cli"
```

### 13.3 Rust Packaging

```toml
# Cargo.toml (qubit-os-hardware)
[package]
name = "qubit-os-hardware"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.35", features = ["full"] }
tonic = "0.11"
prost = "0.12"
pyo3 = { version = "0.20", features = ["auto-initialize"] }
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
thiserror = "1.0"
uuid = { version = "1.7", features = ["v4"] }

[build-dependencies]
tonic-build = "0.11"

[[bin]]
name = "qubit-os-hal"
path = "src/main.rs"
```

### 13.4 Version Strategy

- Each repo has independent semver
- Breaking proto changes bump minor version (0.x) or major version (1.x+)
- Changelog follows [Keep a Changelog](https://keepachangelog.com/)
- Git tags: `v0.1.0`

---

## 14. CI/CD Pipeline

### 14.1 GitHub Actions (Proto Repo)

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bufbuild/buf-setup-action@v1
      - run: buf lint
      - run: buf breaking --against 'https://github.com/qubit-os/qubit-os-proto.git#branch=main'

  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bufbuild/buf-setup-action@v1
      - run: buf generate
      - name: Check generated code is committed
        run: git diff --exit-code generated/
```

### 14.2 GitHub Actions (Hardware Repo)

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      
      - name: Format check
        run: cargo fmt --check
      
      - name: Clippy
        run: cargo clippy --all-targets -- -D warnings
      
      - name: Build
        run: cargo build --release
      
      - name: Test
        run: cargo test --all-features

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: qubit-os-hardware:test
```

### 14.3 GitHub Actions (Core Repo)

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Lint
        run: |
          ruff check src/
          ruff format --check src/
      
      - name: Type check
        run: mypy src/qubitos/
      
      - name: Test
        run: pytest tests/ --cov=qubitos --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      # Start HAL (would need hardware repo built)
      # - name: Integration tests
      #   run: pytest tests/integration/
```

### 14.4 Release Process

1. Update CHANGELOG.md
2. Bump version in pyproject.toml/Cargo.toml
3. Create PR, merge to main
4. Tag: `git tag v0.1.0 && git push --tags`
5. GitHub Release auto-publishes artifacts

### 14.5 CI/CD Validation Requirements

**Minimum Version Specifications:**

| Tool/Dependency | Minimum Version | Rationale |
|-----------------|-----------------|-----------|
| Rust | 1.83 | Required by icu_*, pest, indexmap |
| Python | 3.11 | Type hint syntax, performance |
| buf | 1.47 | Current stable |
| tonic | 0.11 | Must match prost version |
| prost | 0.12 | Current stable |

**Generated Code Policy:**

Proto-generated code is built at compile/install time, NOT committed:

| Language | Method |
|----------|--------|
| Rust | `build.rs` + `tonic-build` on `cargo build` |
| Python | `setup.py` + `grpcio-tools` on `pip install` |

Benefits: No sync drift, no merge conflicts, standard ecosystem patterns.

**Disabled CI Job Policy:**

Jobs may be temporarily disabled during development but must:
1. Be commented out (not deleted)
2. Include `# TODO: <reason>` comment
3. Include target phase for re-enabling
4. Be tracked in ROADMAP.md

**Per-Phase Gate:**

Before proceeding to next phase, all enabled CI jobs must pass. See ROADMAP.md for per-phase CI requirements and pre-flight checklists.

---

## 15. CLI Specification

### 15.1 Command Tree

```
qubit-os
├── hal
│   ├── start [--config FILE] [--grpc-port PORT] [--rest-port PORT]
│   ├── health [--backend NAME]
│   └── info [--backend NAME]
├── pulse
│   ├── generate [--gate GATE] [--algorithm ALG] [--duration NS] [--output FILE]
│   ├── validate FILE
│   ├── show FILE
│   └── execute FILE [--backend NAME] [--shots N]
├── calibration
│   ├── show [--backend NAME]
│   ├── validate FILE
│   └── set FILE [--backend NAME]
├── config
│   ├── show
│   └── validate FILE
└── version
```

### 15.2 Output Formats

```bash
# Default: human-readable
qubit-os hal health

# JSON output
qubit-os hal health --format json

# YAML output
qubit-os hal health --format yaml
```

### 15.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Connection error |
| 5 | Validation error |
| 6 | Backend error |

---

## 16. REST API Specification

### 16.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /backends | List backends |
| GET | /backends/{name} | Get backend info |
| POST | /pulse/execute | Execute pulse |
| POST | /pulse/batch | Execute batch |
| POST | /grape/optimize | Run GRAPE |
| DELETE | /grape/{trace_id} | Cancel GRAPE |

### 16.2 OpenAPI Specification

Full OpenAPI 3.0 spec at `docs/api/openapi.yaml` in qubit-os-core repo.

Example endpoint:
```yaml
paths:
  /pulse/execute:
    post:
      summary: Execute a pulse
      operationId: executePulse
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExecutePulseRequest'
      responses:
        '200':
          description: Successful execution
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExecutePulseResponse'
        '400':
          description: Invalid request
        '503':
          description: Backend unavailable
```

---

## 17. Security Considerations

### 17.1 Trust Boundaries

```
┌─────────────────────────────────────────────────────┐
│ Untrusted: User input, external API responses       │
└──────────────────────┬──────────────────────────────┘
                       │ Validation
                       ▼
┌─────────────────────────────────────────────────────┐
│ Validated: Proto messages within HAL                │
└──────────────────────┬──────────────────────────────┘
                       │ Backend call
                       ▼
┌─────────────────────────────────────────────────────┐
│ Trusted: QuTiP internal, IQM responses              │
└─────────────────────────────────────────────────────┘
```

### 17.2 Input Validation

All inputs validated at first entry point:
- Proto deserialization (automatic bounds checking)
- Custom validators (Hamiltonian, pulse, calibration)
- AgentBible domain validators

### 17.3 Secret Management

```bash
# Do NOT use plain environment variables in production
# Use Docker secrets or secret managers

# Docker secrets
docker secret create iqm_token ./token.txt
# Reference in compose: /run/secrets/iqm_token

# Or Kubernetes secrets
kubectl create secret generic iqm-creds --from-file=token=./token.txt
```

### 17.4 Dependency Security

- Dependabot enabled on all repos
- `cargo audit` in Rust CI
- `pip-audit` in Python CI
- Allowed licenses: Apache-2.0, MIT, BSD-3-Clause

---

## 18. Resource Limits

### 18.1 Default Limits

| Resource | Default | Max Configurable |
|----------|---------|------------------|
| Hilbert dimension | 64 | 256 |
| Number of qubits | 6 | 8 |
| Shots per request | 100,000 | 1,000,000 |
| Pulse duration | 100,000 ns | 1,000,000 ns |
| Time steps | 10,000 | 100,000 |
| Batch size | 100 | 1,000 |
| GRAPE iterations | 10,000 | 100,000 |
| Concurrent requests | 10 | 100 |

### 18.2 Limit Enforcement

```rust
pub fn validate_limits(request: &ExecutePulseRequest, limits: &ResourceLimits) -> Result<(), LimitError> {
    if request.num_shots > limits.max_shots {
        return Err(LimitError::ShotsExceeded { 
            requested: request.num_shots, 
            max: limits.max_shots 
        });
    }
    // ... other checks
    Ok(())
}
```

### 18.3 Per-Backend Limits

```yaml
# config.yaml
backends:
  qutip_simulator:
    limits:
      max_qubits: 6
      max_shots: 100000
  iqm_garnet:
    limits:
      max_qubits: 20
      max_shots: 10000
```

---

## 19. Documentation Plan

### 19.1 Documentation Types

| Type | Location | Format |
|------|----------|--------|
| Design doc | qubit-os-core/docs/ | Markdown |
| API reference | qubit-os-core/docs/api/ | OpenAPI + auto-generated |
| User guides | qubit-os-core/docs/guides/ | Markdown |
| Proto spec | qubit-os-proto/docs/ | Extracted from proto comments |
| Rust docs | qubit-os-hardware | rustdoc |
| Python docs | qubit-os-core | Sphinx |

### 19.2 Documentation Hosting

- GitHub Pages from `main` branch `/docs` folder
- URL: `https://qubit-os.github.io/qubit-os-core/`

### 19.3 Required Documents

1. **Quickstart** - 15-minute walkthrough
2. **Installation Guide** - All platforms
3. **Architecture Overview** - This document
4. **CLI Reference** - All commands
5. **API Reference** - REST and gRPC
6. **Backend Guide** - How to add new backends
7. **Calibration Guide** - Managing calibrations
8. **Troubleshooting** - Common errors

---

## 20. Phase 0 Completion Criteria

### 20.1 Architecture & Protocols

- [ ] All proto messages defined and committed
- [ ] Proto generation pipeline working (buf generate)
- [ ] Proto round-trip tests passing
- [ ] Calibration fingerprint implemented
- [ ] Default calibration files created

### 20.2 Code & Tests

- [ ] HAL compiles with zero warnings
- [ ] QuTiP backend returns deterministic results (seed=42)
- [ ] Health checks work for all backends
- [ ] Test coverage meets targets
- [ ] Reproducibility Tier 1 verified

### 20.3 CI/CD

- [ ] GitHub Actions CI passing on all repos
- [ ] Docker build working
- [ ] Release workflow tested

### 20.4 Documentation

- [ ] README for each repo
- [ ] CONTRIBUTING guide
- [ ] CHANGELOG initialized
- [ ] This design doc in docs/

### 20.5 Operations

- [ ] Config hierarchy working
- [ ] Logging with trace_id working
- [ ] Graceful shutdown implemented

---

## Appendix A: Technical Specifications

### A.1 Pauli String Grammar

```ebnf
pauli_string     = term { whitespace? sign whitespace? term }
term             = coefficient? pauli_product
coefficient      = number | complex
complex          = "(" number whitespace? sign whitespace? number "j" ")"
number           = sign? digit+ ("." digit+)? (("e" | "E") sign? digit+)?
sign             = "+" | "-"
pauli_product    = pauli_op { whitespace? "*" whitespace? pauli_op }
pauli_op         = pauli_gate qubit_index
pauli_gate       = "I" | "X" | "Y" | "Z"
qubit_index      = digit+
whitespace       = (" " | "\t")+
digit            = "0" | "1" | ... | "9"
```

**Examples:**
```
0.5*X0
X0*Z1
0.5*X0 + 0.3*Y1
(0.5+0.1j)*X0*Y1 - 0.2*Z0
I0  # Identity on qubit 0
```

**Normalization:** Parser should handle arbitrary whitespace and produce canonical form with single spaces.

### A.2 Matrix Sparse Format

```json
{
  "format": "coo",
  "shape": [4, 4],
  "nnz": 4,
  "rows": [0, 1, 2, 3],
  "cols": [3, 2, 1, 0],
  "data": [
    {"re": 0.5, "im": 0.0},
    {"re": 0.5, "im": 0.0},
    {"re": 0.5, "im": 0.0},
    {"re": 0.5, "im": 0.0}
  ]
}
```

### A.3 State Vector Format

```json
{
  "num_qubits": 2,
  "amplitudes": [
    {"re": 0.707, "im": 0.0},
    {"re": 0.0, "im": 0.0},
    {"re": 0.0, "im": 0.0},
    {"re": 0.707, "im": 0.0}
  ]
}
```

### A.4 Units Table

| Field | Unit | Type | Valid Range |
|-------|------|------|-------------|
| duration_ns | nanoseconds | int32 | [1, 1000000] |
| time_step_ns | nanoseconds | double | [0.1, 10000] |
| max_amplitude_mhz | MHz | double | [0, 1000] |
| frequency_ghz | GHz | double | [1, 20] |
| anharmonicity_mhz | MHz | double | [-500, 0] |
| t1_us | microseconds | double | [1, 10000] |
| t2_us | microseconds | double | [1, 10000] |
| coupling_khz | kHz | double | [0, 1000] |

---

## Appendix B: Default Configurations

### B.1 Default config.yaml

```yaml
# QubitOS HAL Configuration
version: "1.0"

server:
  grpc_port: 50051
  rest_port: 8080
  host: "0.0.0.0"

backends:
  qutip_simulator:
    enabled: true
    default: true
  iqm_garnet:
    enabled: false
    gateway_url: "${IQM_GATEWAY_URL}"
    auth_token: "${IQM_AUTH_TOKEN}"

calibration:
  directory: "./calibration"
  auto_load: true

logging:
  level: "info"
  format: "json"
  directory: "./logs"
  max_size_mb: 100
  max_files: 10

validation:
  strict: true
```

### B.2 Default Calibration (QuTiP)

See `calibration/defaults/qutip_simulator.yaml` for full file.

---

**Document Version:** 0.5.0  
**Status:** Ready for Implementation  
**Last Updated:** January 26, 2026
