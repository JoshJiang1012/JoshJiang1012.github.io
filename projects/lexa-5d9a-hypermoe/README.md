# Lexa 5D9A HyperMoE

[![CI](https://github.com/JoshJiang1012/JoshJiang1012.github.io/actions/workflows/lexa-hypermoe-ci.yml/badge.svg)](https://github.com/JoshJiang1012/JoshJiang1012.github.io/actions/workflows/lexa-hypermoe-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A reproducible research toolkit for **heterogeneous, memory-tiered
Mixture-of-Experts inference** on consumer hardware.

The reference target is GPT-OSS-120B on a system shaped like:

- NVIDIA RTX 4080 16 GiB;
- Intel Core i7-13700;
- 32 GiB system RAM;
- NVMe backing storage.

Version **0.2.0** adds a pinned `llama.cpp` Router Trace collector so the
synthetic cache assumptions can be replaced with observed per-layer Expert
routing metadata.

> **Evidence boundary:** RTX 4080 throughput values remain analytical assumptions
> unless explicitly marked `observed`. This repository does not claim that
> GPT-OSS-120B has reached 90 tokens/s on that system.

## Included

```text
src/lexa_hypermoe/
├── model.py          # equations, tier optimizer, Zipf/deadline helpers
├── trace.py          # privacy-safe route audit, cache sweep, temporal locality
└── cli.py            # reproducible CLI

patches/llama.cpp/
├── d08c7872-router-trace.patch
├── manifest.json
└── README.md

scripts/
├── generate_data.py
├── calibrate_host.py
├── verify_llama_patch.py
├── collect_router_trace.py
└── analyze_router_trace.py

data/
├── model/            # official constants + derived traffic estimates
├── hardware/         # explicit assumption profiles
├── synthetic/        # formula-generated sweeps and a synthetic trace example
├── observed/         # reserved for real submissions
└── schemas/          # trace and run-manifest contracts
```

## Analytical core

For `L` layers, `K` active Experts per layer, hidden dimension `d`, Expert
intermediate dimension `m`, and storage cost `q` bytes/weight:

```text
parameters per Expert       = 3 d m
active Expert bytes/token   = L K (3 d m) q
```

The three Expert-service paths overlap:

```text
GPU time   = (dense bytes + h_gpu × Expert bytes) / GPU effective bandwidth
RAM time   = h_ram × Expert bytes / RAM effective bandwidth
NVMe time  = h_nvme × Expert bytes / NVMe effective path bandwidth

token time = max(GPU time, RAM time, NVMe time)
           + fixed overhead
           + expected critical-miss penalty
```

with `h_gpu + h_ram + h_nvme = 1`.

See [`docs/MATHEMATICAL_MODEL.md`](docs/MATHEMATICAL_MODEL.md).

## Quick start

```bash
python -m pip install -e .
python scripts/generate_data.py
python -m unittest discover -s tests -v
```

Run the RTX 4080 assumption profile:

```bash
lexa-hypermoe estimate \
  --model data/model/gpt_oss_120b.json \
  --hardware data/hardware/rtx4080_i7_13700_32gb_assumed.json
```

## Build the real Router Trace collector

The patch is pinned to:

```text
ggml-org/llama.cpp
d08c7872d6ffe3f059f8647840a29aa390413e27
```

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout d08c7872d6ffe3f059f8647840a29aa390413e27

python /path/to/lexa-5d9a-hypermoe/scripts/verify_llama_patch.py \
  --llama-source . \
  --apply

cmake -S . -B build \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON
cmake --build build --target llama-router-trace -j
```

The patch adds a dedicated executable without changing model weights, GPT-OSS
routing, `ggml_mul_mat_id`, CUDA kernels, or sampling.

## Collect decode routing

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --output data/observed/coding-session-001.jsonl \
  --prompt-file /private/prompts/coding-session-001.txt \
  --domain coding \
  --n-predict 4096 \
  --ctx-size 8192 \
  --gpu-layers 99
```

The trace stores sequence indexes and Expert IDs, not prompt text, generated
text, token IDs, logits, probabilities, embeddings, or hidden states. The sidecar
run manifest stores only the prompt SHA-256 and UTF-8 byte count.

Audit the trace:

```bash
lexa-hypermoe trace-audit \
  --trace data/observed/coding-session-001.jsonl
```

Measure a top-18 cache and temporal locality:

```bash
lexa-hypermoe trace \
  --trace data/observed/coding-session-001.jsonl \
  --phase decode \
  --cached-experts-per-layer 18
```

Compare cache sizes:

```bash
lexa-hypermoe trace-sweep \
  --trace data/observed/coding-session-001.jsonl \
  --phase decode \
  --cache-sizes 4,8,12,16,18,20,24,32
```

Full instructions: [`docs/ROUTER_TRACE.md`](docs/ROUTER_TRACE.md).

## Trace event

```json
{
  "schema_version": "2.0",
  "token": 42,
  "layer": 7,
  "experts": [3, 19, 81, 122],
  "domain": "coding",
  "phase": "decode",
  "batch_size": 1,
  "source": "llama.cpp:ffn_moe_topk"
}
```

The Python parser is fail-closed: privacy-forbidden and unknown fields are
rejected by default.

## Generated analytical datasets

| File | Meaning |
|---|---|
| `throughput_sweep.csv` | GPU efficiency × RAM bandwidth × NVMe path × overhead |
| `zipf_cache_sweep.csv` | synthetic Expert-skew sensitivity |
| `prefetch_deadline_sweep.csv` | look-ahead layers required by target throughput |
| `critical_miss_requirements.csv` | per-layer miss bound for clean-token probability |
| `vram_capacity_sweep.csv` | Expert slots under reserved-VRAM budgets |

All synthetic rows include a `data_class`; they must not be presented as
observed benchmark results.

## Research status

Implemented:

- deterministic GPU/RAM/NVMe analytical model;
- constrained placement optimizer;
- cache-capacity, Zipf, prefetch, and critical-miss sweeps;
- pinned `llama.cpp` route-collector patch;
- strict privacy audit and schema;
- static top-N cache analysis;
- consecutive-token Expert-overlap analysis;
- run provenance manifest;
- tests and CI for Python 3.10–3.12;
- CI build validation of the patched `llama-router-trace` target.

Not implemented:

- GPU-resident dynamic Expert slots;
- fused MXFP4 CPU/GPU kernels;
- residual-based future-router prediction;
- asynchronous NVMe Expert paging;
- exact speculative-decoding integration;
- a verified GPT-OSS-120B performance result on RTX 4080.

## Source provenance

Model constants come from the official GPT-OSS configuration and model card.
MXFP4 packing follows the official GPT-OSS reference code. The route patch is
pinned to a specific upstream `llama.cpp` commit and verifies the expected
`ffn_moe_topk` graph marker before applying.

See [`docs/REFERENCES.md`](docs/REFERENCES.md).

## License

Apache-2.0. Model weights are not included and retain their own license and usage
requirements.
