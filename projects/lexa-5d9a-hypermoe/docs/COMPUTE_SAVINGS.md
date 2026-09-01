# Lexa 5D9A HyperMoE v1.0.0 — Compute Savings Summary

**Classification:** analytical comparison derived from the packaged model and hardware assumption profiles. This is **not** an observed GPT-OSS-120B / RTX 4080 benchmark.

## 1. Inherent GPT-OSS MoE sparsity

- Total parameters: 116.8292B
- Active parameters per token: 5.1000B
- Active fraction: 4.365%
- Parameter-work proxy skipped per token: 95.635%
- Active-parameter reduction ratio: 22.91x
- Experts selected per layer: 4/128
- Expert fraction active per token: 3.125%
- Expert work / weight access avoided: 96.875%

## 2. Analytical active-weight traffic

- Hypothetical all-expert + dense traffic: 63.9477 GB/token
- Sparse active traffic: 4.9372 GB/token
- Avoided traffic: 59.0105 GB/token (92.279%)
- Active dense traffic: 3.0336 GB/token
- Active Expert traffic: 1.9036 GB/token

## 3. HyperMoE placement target

| Path | Expert fraction | Expert traffic/token |
|---|---:|---:|
| GPU | 68.250% | 1.2992 GB |
| RAM/CPU | 28.750% | 0.5473 GB |
| NVMe | 3.000% | 0.0571 GB |

- GPU active traffic after placement, including dense: 4.3328 GB/token
- GPU traffic avoided versus all-active-on-GPU: 604.38 MB/token
- Expert GPU traffic reduction: 31.750%
- Total GPU weight-traffic reduction: 12.241%

## 4. Analytical latency / throughput scenarios

All rows use the packaged assumptions: GPU effective bandwidth 394.24 GB/s, RAM 50 GB/s, NVMe 5.5 GB/s, fixed overhead 1.5 ms/token, and ideal path overlap.

| Scenario | GPU Expert service | Time/token | Estimated tok/s | Speedup vs CPU-MoE | Latency saved vs CPU-MoE |
|---|---:|---:|---:|---:|---:|
| CPU-MoE baseline | 0% | 39.571 ms | 25.27 | 1.00x | 0% |
| Uniform-routing capacity floor (649/4608 slots) | 14.08% | 30.968 ms | 32.29 | 1.28x | 21.7% |
| 50% GPU Expert hit/service | 50% | 18.649 ms | 53.62 | 2.12x | 52.9% |
| v1.0 idealized optimum | 68.25% | 12.490 ms | 80.06 | 3.17x | 68.4% |
| Theoretical all-active-on-GPU | 100% | 14.023 ms | 71.31 | 2.82x | 64.6% |

At the idealized v1.0 target, 1,000 generated tokens take about 12.49 seconds instead of 39.57 seconds in the CPU-MoE analytical baseline, saving 27.08 seconds.

## 5. Residency capacity

- Conservative direct-residency threshold used by v1.0 validation: 65 GiB.
- User hardware aggregate RAM + VRAM: 48 GiB.
- Cold-paging design avoids 17 GiB of immediate residency, a 26.154% reduction, while keeping the full model on NVMe.
- The 8 GiB GPU Expert cache holds 649 layer-specific Expert slots, averaging 18.03 per layer, or 14.084% of the full Expert-slot population.

## Evidence boundary

The 68.25% GPU Expert-service fraction is a placement optimum, **not a measured cache hit rate**. A real GPT-OSS-120B Router Trace is required to determine whether the top ~18 Experts per layer can achieve it. If routing were uniform, the capacity-only hit rate would be about 14.08%, corresponding to roughly 32.29 analytical tok/s under the same assumptions.
