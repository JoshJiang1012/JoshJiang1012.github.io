# Analytical compute-savings summary

Using the included GPT-OSS-120B geometry:

- active parameters per token: 5.1B of 116.8292B;
- active-parameter ratio: 4.365%;
- parameter-work proxy avoided: 95.635%;
- active Experts per layer: 4 of 128;
- Expert-work proxy avoided: 96.875%;
- active weight-traffic proxy: 4.9372 GB/token;
- full-Expert traffic proxy: approximately 63.95 GB/token;
- analytical traffic reduction: approximately 92.28%.

For the bundled RTX 4080 / i7-13700 assumptions, grid search returns roughly
68.25% GPU, 28.75% RAM, and 3.00% NVMe Expert service. The analytical model
estimates 80.06 token/s versus 25.27 token/s for a CPU-MoE baseline, or about
3.17x.

These numbers are **not observed hardware benchmarks**. They rely on assumed
effective bandwidth, ideal overlap between execution paths, and a fixed
software-overhead term. Real results require a model run, Router Trace,
P50/P95/P99 latency, power data, and full runtime provenance.
