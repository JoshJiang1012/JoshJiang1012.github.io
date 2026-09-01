# Lexa 5D9A HyperMoE

[![CI](https://github.com/JoshJiang1012/JoshJiang1012.github.io/actions/workflows/lexa-hypermoe-ci.yml/badge.svg)](https://github.com/JoshJiang1012/JoshJiang1012.github.io/actions/workflows/lexa-hypermoe-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A reproducible analytical toolkit and dataset for studying **heterogeneous,
memory-tiered Mixture-of-Experts inference** on consumer hardware.

The first reference target is OpenAI GPT-OSS-120B on a system shaped like:

- NVIDIA RTX 4080 16 GiB;
- Intel Core i7-13700;
- 32 GiB system RAM;
- NVMe backing storage.

The project models a proposed **5D9A HyperMoE** scheduler that distributes active
expert work across GPU VRAM, CPU/RAM, and NVMe-backed cold storage while tracking
cache hit rate, critical misses, prefetch distance, and fixed runtime overhead.

> **Important:** the included RTX 4080 numbers are analytical assumptions unless
> explicitly marked `observed`. This repository does **not** claim that GPT-OSS-120B
> has already reached 90 tokens/s on that machine.

## What is included

```text
src/lexa_hypermoe/
├── model.py          # equations, tier optimizer, Zipf and deadline helpers
├── trace.py          # JSONL router-trace analysis
└── cli.py            # reproducible CLI

data/
├── model/            # official constants + derived traffic estimates
├── hardware/         # explicit assumption profiles
├── synthetic/        # formula-generated parameter sweeps
├── observed/         # reserved for real benchmark submissions
└── schemas/          # router-trace and benchmark schemas

scripts/
├── generate_data.py  # rebuild all synthetic datasets and reports
├── calibrate_host.py # safe host calibration helper
└── analyze_router_trace.py
```

## Core model

For a model with `L` layers, `K` active experts per layer, hidden dimension `d`,
and expert intermediate dimension `m`, the approximation uses:

```text
parameters per expert = 3 d m
active expert bytes/token = L K (3 d m) q
```

where `q = 17/32 bytes/weight` approximates native MXFP4 storage including one
E8M0 scale byte per 32 packed FP4 values.

The three expert paths overlap:

```text
GPU time   = (dense bytes + h_gpu × expert bytes) / GPU effective bandwidth
RAM time   = h_ram × expert bytes / RAM effective bandwidth
NVMe time  = h_nvme × expert bytes / NVMe effective path bandwidth

token time = max(GPU time, RAM time, NVMe time)
           + fixed overhead
           + expected critical-miss penalty
```

`h_gpu + h_ram + h_nvme = 1`.

See [`docs/MATHEMATICAL_MODEL.md`](docs/MATHEMATICAL_MODEL.md) for assumptions,
derivations, capacity constraints, and limitations.

## Quick start

Python 3.10 or newer is sufficient.

```bash
python scripts/generate_data.py
python -m unittest discover -s tests -v
```

Install the CLI locally:

```bash
python -m pip install -e .
```

Run the RTX 4080 assumption profile:

```bash
lexa-hypermoe estimate \
  --model data/model/gpt_oss_120b.json \
  --hardware data/hardware/rtx4080_i7_13700_32gb_assumed.json
```

Analyze a real router trace:

```bash
lexa-hypermoe trace \
  --trace data/observed/router-trace.jsonl \
  --cached-experts-per-layer 18
```

## Generated datasets

| File | Meaning |
|---|---|
| `throughput_sweep.csv` | GPU efficiency × RAM bandwidth × NVMe path × overhead |
| `zipf_cache_sweep.csv` | synthetic expert-skew sensitivity |
| `prefetch_deadline_sweep.csv` | layers of look-ahead required by target throughput |
| `critical_miss_requirements.csv` | per-layer miss bound for clean-token probability |
| `vram_capacity_sweep.csv` | expert slots under different reserved-VRAM budgets |

All rows include a `data_class` field. Synthetic rows must never be presented as
observed benchmark results.

## Collecting real data

1. Calibrate the host:

   ```bash
   python scripts/calibrate_host.py --output data/observed/host-profile.local.json
   ```

2. Instrument an inference runtime to emit only numeric routing events:

   ```json
   {"token": 42, "layer": 7, "experts": [3, 19, 81, 122], "domain": "coding"}
   ```

3. Analyze top-N cache performance with `lexa-hypermoe trace`.
4. Publish model hash, runtime commit, context, warm-up, P50/P95/P99, and raw logs.

The exact schema is in [`data/schemas/router_trace.schema.json`](data/schemas/router_trace.schema.json).

## Research status

Implemented:

- deterministic analytical model;
- constrained GPU/RAM/NVMe placement search;
- VRAM expert-slot estimates;
- Zipf sensitivity sweeps;
- critical-miss probability requirements;
- multi-horizon prefetch calculations;
- router-trace hit-rate analysis;
- reproducible CI and datasets.

Not implemented:

- a patched llama.cpp runtime;
- GPU-resident dynamic expert slots;
- fused MXFP4 CPU/GPU kernels;
- residual-based future-router prediction;
- exact speculative decoding integration;
- a verified 90 tok/s GPT-OSS-120B result on RTX 4080.

## Source provenance

The model constants are based on the official GPT-OSS configuration and model
card. MXFP4 packing follows the official GPT-OSS reference code. External
references and exact links are listed in
[`docs/REFERENCES.md`](docs/REFERENCES.md).

## License

Apache-2.0. Model weights are not included and retain their own license and usage
requirements.
