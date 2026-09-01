# RTX 4080 baseline analytical report

This report is generated from assumptions, not a measured GPT-OSS-120B run.

## Inputs

- GPU peak bandwidth: 716.8 GB/s
- Assumed GPU effective efficiency: 55.00%
- Effective GPU bandwidth: 394.24 GB/s
- Assumed RAM effective bandwidth: 50.00 GB/s
- Assumed NVMe effective path: 5.50 GB/s
- Fixed software overhead: 1.50 ms/token
- VRAM reserved outside expert cache: 8.00 GiB
- Estimated expert slots: 649 total, 18.03 per layer on average

## Derived model traffic

- Expert size: 12.6068 MiB
- Active expert traffic: 1.9036 GB/token
- Estimated dense traffic: 3.0336 GB/token
- Estimated total active traffic: 4.9372 GB/token

## Grid-optimized idealized placement

- GPU fraction: 68.25%
- RAM fraction: 28.75%
- NVMe fraction: 3.00%
- Critical path before fixed overhead: 10.9903 ms/token
- Total modeled time: 12.4903 ms/token
- Modeled throughput: 80.06 tok/s

## Interpretation

The estimate is an optimistic systems model. It assumes GPU, CPU/RAM, and NVMe
expert paths overlap perfectly and represents dequantization/compute through
*effective bandwidth*. It does not include router prediction errors, cache churn,
CUDA launch tails, OS page-fault tails, thermal throttling, or implementation
limitations. Replace assumed rates with observed measurements before drawing a
hardware conclusion.
