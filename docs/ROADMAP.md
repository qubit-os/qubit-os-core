# QubitOS Development Roadmap

## Overview

QubitOS is an open-source quantum control kernel providing pulse optimization and hardware abstraction for quantum computing research. This roadmap outlines the development phases from design to production-ready release.

## Current Status

**Phase 0: Design & Foundation** - In Progress

## Timeline

```
2026 Q1
├── Phase 0: Design & Foundation (Jan-Feb)
│   ├── Week 1-2: Design document finalization ✓
│   ├── Week 3-4: Proto definitions and scaffolding ✓
│   └── Week 5-6: CI/CD setup and initial structure
│
├── Phase 1: Core Implementation (Feb-Mar)
│   ├── Week 1-2: HAL server skeleton (gRPC + REST)
│   ├── Week 3-4: QuTiP backend implementation
│   ├── Week 5-6: Basic GRAPE optimizer
│   └── Week 7-8: CLI and Python client
│
└── Phase 2: Integration & Testing (Mar-Apr)
    ├── Week 1-2: End-to-end integration
    ├── Week 3-4: Reproducibility validation
    └── Week 5-6: Documentation and release prep

2026 Q2
├── Phase 3: IQM Integration (Apr-May)
│   ├── IQM backend implementation
│   ├── Sim-to-real validation
│   └── Hardware-specific calibration
│
└── Phase 4: v0.1.0 Release (May-Jun)
    ├── Public release
    ├── Documentation site
    └── Community feedback
```

---

## Phase 0: Design & Foundation

**Duration:** 6 weeks  
**Status:** In Progress  
**Goal:** Rock-solid foundation before writing implementation code

### Deliverables

| Item | Status | Notes |
|------|--------|-------|
| Design document v0.5.0 | Done | All audit items addressed |
| Repository structure | Done | 3 repos with scaffolding |
| Proto definitions | Done | All message types defined |
| CI/CD workflows | Done | GitHub Actions for all repos |
| Default calibration | Done | QuTiP simulator defaults |
| README files | Done | All 3 repos |
| License (Apache 2.0) | Done | All repos |

### Remaining Tasks

- [x] Generate proto code (run `buf generate`) - structure in place, generation on CI
- [x] Initial commits and push to GitHub
- [ ] Set up GitHub Pages for documentation
- [x] Create issue templates
- [x] Set up Dependabot

### Completed Infrastructure

| Item | Status | Notes |
|------|--------|-------|
| License headers in Python | Done | All .py files |
| GitHub issue templates | Done | Bug report, feature request |
| Dependabot config | Done | All 3 repos |
| Status badges in READMEs | Done | CI + License |
| GitHub org metadata | Done | Description, topics |
| Pre-commit hooks | Done | Secret detection, ruff |
| Local CI check script | Done | `scripts/ci-check.sh` |
| AgentBible integration | Done | Validation module with fallback |

### Exit Criteria

All items from Section 20 of Design Document v0.5.0 must be complete:

- [ ] Proto generation working
- [ ] CI passing on all repos
- [ ] Default calibration loading works
- [ ] Documentation structure in place

---

## Phase 1: Core Implementation

**Duration:** 8 weeks  
**Goal:** Working single-qubit pulse optimization and execution

### 1.1 HAL Server (Weeks 1-2)

**qubit-os-hardware:**

- [ ] gRPC server setup (tonic)
- [ ] REST API facade (axum)
- [ ] Configuration loading
- [ ] Logging infrastructure
- [ ] Health check endpoint
- [ ] Backend registry skeleton

**Tests:**
- [ ] Server starts and responds to health checks
- [ ] Configuration hierarchy works
- [ ] Graceful shutdown

### 1.2 QuTiP Backend (Weeks 3-4)

**qubit-os-hardware:**

- [ ] PyO3 integration
- [ ] QuTiP executor (mesolve)
- [ ] Noise model from calibration
- [ ] State vector extraction
- [ ] Measurement sampling

**Tests:**
- [ ] Deterministic results with fixed seed
- [ ] Noise model matches calibration
- [ ] State vector correct for known states

### 1.3 GRAPE Optimizer (Weeks 5-6)

**qubit-os-core:**

- [ ] Hamiltonian parsing (Pauli strings)
- [ ] GRAPE core algorithm
- [ ] Gradient computation
- [ ] Convergence detection
- [ ] Fidelity calculation

**Tests:**
- [ ] X-gate fidelity >= 99.9%
- [ ] Reproducibility: same seed = same result
- [ ] Convergence within 1000 iterations

### 1.4 CLI and Client (Weeks 7-8)

**qubit-os-core:**

- [ ] gRPC client implementation
- [ ] CLI structure (click)
- [ ] `qubit-os hal health`
- [ ] `qubit-os pulse generate`
- [ ] `qubit-os pulse execute`
- [ ] JSON/YAML output formats

**Tests:**
- [ ] CLI commands work end-to-end
- [ ] Error handling and exit codes
- [ ] Help text and documentation

### Phase 1 Exit Criteria

- [ ] Generate X-gate pulse with GRAPE
- [ ] Execute pulse on QuTiP backend
- [ ] Get measurement results
- [ ] Fidelity >= 99% on single-qubit gates
- [ ] Full CLI workflow works

---

## Phase 2: Integration & Testing

**Duration:** 6 weeks  
**Goal:** Production-quality code with full test coverage

### 2.1 End-to-End Integration (Weeks 1-2)

- [ ] Complete pulse generation → execution pipeline
- [ ] Calibration fingerprint validation
- [ ] Trace ID propagation
- [ ] Error handling across boundaries

### 2.2 Reproducibility Validation (Weeks 3-4)

- [ ] Tier 1: Same seed = identical results
- [ ] Golden file tests
- [ ] Cross-platform consistency
- [ ] Version pinning validation

### 2.3 Documentation & Polish (Weeks 5-6)

- [ ] Quickstart guide
- [ ] API reference (auto-generated)
- [ ] Example notebooks
- [ ] Troubleshooting guide
- [ ] Code cleanup and refactoring

### Phase 2 Exit Criteria

- [ ] Test coverage: HAL >= 85%, Core >= 75%
- [ ] All documentation written
- [ ] No known critical bugs
- [ ] Performance meets SLA targets

---

## Phase 3: IQM Integration

**Duration:** 6 weeks  
**Goal:** Working hardware backend

### 3.1 IQM Backend (Weeks 1-3)

- [ ] IQM Resonance API client
- [ ] Authentication handling
- [ ] Job submission and polling
- [ ] Result retrieval
- [ ] Error handling and retries

### 3.2 Sim-to-Real Validation (Weeks 4-5)

- [ ] Hellinger distance comparison
- [ ] Validation test suite
- [ ] Calibration from hardware
- [ ] Document discrepancies

### 3.3 Hardware Calibration (Week 6)

- [ ] Live calibration measurement
- [ ] T1/T2 fitting
- [ ] Gate fidelity benchmarking
- [ ] Calibration storage

### Phase 3 Exit Criteria

- [ ] Execute pulses on IQM Garnet
- [ ] Sim-to-real Hellinger distance < 0.05
- [ ] Hardware calibration workflow works

---

## Phase 4: v0.1.0 Release

**Duration:** 4 weeks  
**Goal:** Public release

### 4.1 Release Preparation (Weeks 1-2)

- [ ] Version bump to 0.1.0
- [ ] CHANGELOG finalization
- [ ] Release notes
- [ ] Final testing
- [ ] Security audit

### 4.2 Publication (Week 3)

- [ ] Tag releases
- [ ] Publish Python package (PyPI)
- [ ] Publish Docker images (GHCR)
- [ ] Documentation site live

### 4.3 Announcement (Week 4)

- [ ] GitHub release announcement
- [ ] Social media / community posts
- [ ] Gather initial feedback

---

## Future Phases (Post v0.1.0)

### v0.2.0 - Multi-Qubit Expansion

- 3+ qubit support
- Advanced 2Q gates (parametric)
- Pulse scheduling
- Parallel optimization

### v0.3.0 - Active Calibration

- Online drift detection
- Automatic recalibration
- Feedback control loop
- Adaptive pulse updates

### v0.4.0 - Additional Backends

- IBM Quantum backend
- AWS Braket backend
- Custom backend SDK

### v1.0.0 - Production Ready

- Stable API
- Full documentation
- Enterprise features
- Community governance

---

## CI/CD Standards

### Per-Phase CI Requirements

Each phase has explicit CI criteria that must be met before proceeding. CI jobs may be disabled with `# TODO:` comments if functionality isn't implemented yet, but **CI must be configured correctly and passing** (even if some jobs are skipped).

#### Phase 0 (Foundation)

| Repo | Required CI Jobs | Notes |
|------|-----------------|-------|
| qubit-os-proto | lint, generate-check, test-python | Rust test may be disabled if tonic compatibility unresolved |
| qubit-os-hardware | check, build, test, docker | Docker must build; tests may be minimal |
| qubit-os-core | lint, test | Type check may be disabled with TODO |

**Pre-Flight Checklist (Phase 0 → Phase 1):**
- [ ] All 3 repos have passing CI (green badges)
- [ ] Proto generation committed and CI-verified (`buf generate` output matches)
- [ ] Dockerfile builds successfully
- [ ] `buf lint` passes with no errors
- [ ] `ruff check` and `ruff format --check` pass
- [ ] `cargo fmt --check` and `cargo clippy` pass
- [ ] All README badges show passing status
- [ ] Git mirrors synced

#### Phase 1 (Core Implementation)

| Repo | Required CI Jobs | Notes |
|------|-----------------|-------|
| qubit-os-proto | All Phase 0 + breaking change detection | |
| qubit-os-hardware | All Phase 0 + integration tests | |
| qubit-os-core | All Phase 0 + type check enabled | Mypy must pass |

**Pre-Flight Checklist (Phase 1 → Phase 2):**
- [ ] All Phase 0 checks pass
- [ ] HAL server starts and responds to health checks
- [ ] QuTiP backend returns deterministic results (seed=42 golden test)
- [ ] GRAPE optimizer converges for X-gate
- [ ] CLI `qubit-os --help` works
- [ ] Test coverage: Rust >= 60%, Python >= 50%

#### Phase 2 (Integration & Testing)

| Repo | Required CI Jobs | Notes |
|------|-----------------|-------|
| All repos | All Phase 1 + coverage thresholds enforced | |
| qubit-os-core | Integration test job added | |

**Pre-Flight Checklist (Phase 2 → Phase 3):**
- [ ] Test coverage: HAL >= 85%, Core >= 75%
- [ ] Reproducibility Tier 1 golden tests passing
- [ ] End-to-end integration test passing
- [ ] All documentation written and builds successfully
- [ ] No critical or high-priority bugs open

### Minimum Version Specifications

Pin to minimum versions that are known to work:

| Tool/Dependency | Minimum Version | Notes |
|-----------------|-----------------|-------|
| Rust | 1.83 | Required by icu_*, pest, indexmap dependencies |
| Python | 3.11 | |
| buf | 1.47 | |
| tonic | 0.11 | Must match prost version |
| prost | 0.12 | |
| ruff | 0.2.0 | |
| pytest | 8.0.0 | |
| qutip | 5.0.0 | |

### Generated Code Policy

**Proto → Rust/Python bindings:**
- Generated code MUST be committed to `generated/` directory
- CI verifies regeneration matches committed code
- Workflow: edit `.proto` → run `buf generate` → commit both

**Why committed (not CI-generated):**
- Multi-repo consumption without requiring `buf` installed
- PR diffs show downstream impact of proto changes
- CI catches sync drift (proto edited but not regenerated)

### Disabled CI Jobs

When a CI job must be temporarily disabled:

1. Comment out the job (don't delete)
2. Add `# TODO: <reason>` comment explaining why
3. Add `# TODO: Re-enable in Phase X` with target phase
4. Track in ROADMAP.md or create GitHub issue

Example:
```yaml
# TODO: Re-enable once tonic version compatibility is resolved (Phase 2)
# test-rust:
#   name: Test Rust Compilation
#   ...
```

---

## Development Principles

### Code Quality

- All code reviewed before merge
- CI must pass (all enabled jobs green)
- Test coverage targets enforced
- Documentation required for new features

### Reproducibility

- Every result traceable to code + seed + calibration
- Golden file tests for numerical code
- Pinned dependencies

### Communication

- Weekly progress updates (if team grows)
- Issues for all tracked work
- PRs reference issues
- Changelog updated with each release

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| QuTiP version incompatibility | High | Pin version, test on updates |
| IQM API changes | Medium | Abstract behind interface, version client |
| Numerical instability in GRAPE | High | Extensive testing, gradient clipping |
| Performance bottlenecks | Medium | Profile early, benchmark regularly |
| Scope creep | High | Strict phase boundaries, defer features |

---

## Success Metrics

### Phase 1

- Single-qubit gate fidelity >= 99.9%
- GRAPE optimization < 30s for typical gates
- Zero critical bugs

### Phase 2

- Test coverage meets targets
- Documentation complete
- 3+ example notebooks

### Phase 3

- IQM execution success rate >= 95%
- Sim-to-real correlation established
- Hardware calibration automated

### v0.1.0

- Clean install works on Linux/macOS
- Quickstart completable in 15 minutes
- Community engagement (stars, issues, discussions)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) in each repository for contribution guidelines.

---

*Last Updated: January 26, 2026*
