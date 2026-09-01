# Limitations

- Linear fractional scaling may not hold for tiny shards or heavily batched kernels.
- Constant efficiency factors do not capture thermal throttling or shape-dependent kernels.
- The current general optimizer is grid-based and grows combinatorially with device count and finer steps.
- Transfer and service are serialized on each device path in v2.0.
- Contention between independent links or multiple workloads is not modeled.
- Energy uses declared average power, not waveform-level measurement.
- The generic engine models one stage directly; multi-stage DAG optimization is a roadmap item.
- Synthetic traces are useful for validation but cannot substitute for observed routing data.
