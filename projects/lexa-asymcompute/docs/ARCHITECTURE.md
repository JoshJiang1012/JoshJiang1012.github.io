# Architecture

```text
JSON workload + hardware profiles
              │
              ▼
       schema/dataclass validation
              │
       ┌──────┴────────┐
       ▼               ▼
 general estimator   sparse specialization
       │               │
       └──────┬────────┘
              ▼
     constrained placement search
              │
              ▼
 latency / throughput / energy report
              │
       provenance classification
```

## Design principles

- **Asymmetry is explicit.** Each tier has its own compute, bandwidth, capacity, latency and power.
- **Infeasible means rejected.** Capacity violations are not hidden behind a soft score.
- **Observed and analytical data are separate.** Classification is part of every published result.
- **No runtime dependency.** The core uses only the Python standard library.
- **Reproducibility over opaque prediction.** Equations and assumptions remain inspectable.
- **Privacy-safe routing data.** Trace records reject prompt text, generated text, token IDs, logits, embeddings and unknown fields by default.

## Extension points

A future backend may replace grid search with MILP or nonlinear optimization while preserving `WorkloadSpec`, `HardwareSystem` and `Estimate` as the interchange contract.
