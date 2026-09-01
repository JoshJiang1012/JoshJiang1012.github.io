# Mathematical model reference

The complete v2.0 equation set is embedded in the root README so that GitHub readers can audit it without navigating away. This file provides implementation mapping.

| Mathematical object | Code |
|---|---|
| Device effective rates \(P_d,B_d\) | `DeviceSpec.effective_compute_flops`, `effective_bandwidth_bytes_s` |
| Work assignment \(F_d,Q_d,R_d\) | `models.estimate` |
| Roofline service time | `_service_components` |
| Link transfer | `LinkSpec.transfer_ms` |
| Critical path | `models.estimate` |
| Simplex placement | `optimizer.placements` |
| Weighted objective | `optimizer.optimize` |
| Jain fairness / asymmetry | `metrics.jain_fairness`, `metrics.asymmetry_index` |
| Sparse specialization | `moe.py` |
| Chronological cache holdout | `trace.temporal_top_n_holdout` |

## Units

Input JSON uses:

- FLOPs as absolute floating-point operation counts;
- bytes as absolute byte counts;
- device compute in TFLOP/s;
- local and link bandwidth in decimal GB/s;
- memory capacity in GiB;
- latency in milliseconds;
- power in watts.

Conversions are explicit in the source. Do not mix GiB capacity with decimal GB/s bandwidth.

## Assumptions

The current estimator assumes:

1. divisible parallel work scales linearly with `share`;
2. arithmetic and local-memory time overlap according to a roofline maximum;
3. each device's service and transfer are serial on that device path;
4. device paths overlap and meet at one barrier;
5. efficiencies are constant within one estimate;
6. resident capacity is a hard constraint;
7. a single stage is modeled directly; DAG composition is documented but not yet solved automatically.

These assumptions are intentionally inspectable rather than hidden in a black-box predictor.
