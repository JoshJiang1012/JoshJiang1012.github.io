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

Version **1.0.0** is the first stable public research snapshot. It combines
the pinned `llama.cpp` Router Trace collector, privacy-minimized run manifests,
trace-geometry validation, chronological 70/30 holdout evaluation, and an
explicit compute-savings model for GPU/RAM/NVMe Expert placement.

> **Evidence boundary:** RTX 4080 throughput values remain analytical assumptions
> unless explicitly marked `observed`. This repository does not claim that
> GPT-OSS-120B has reached 90 tokens/s on that system.

## Compute-savings snapshot

The following figures are **analytical proxies**, not observed RTX 4080
benchmarks:

| Metric | Analytical result |
|---|---:|
| MoE inactive-parameter work proxy | 95.635% skipped per token |
| Expert work / weight access | 96.875% skipped per token |
| Active-weight traffic vs all-Expert execution | 92.279% lower |
| Idealized HyperMoE speedup vs CPU-MoE baseline | 3.17× |
| Idealized latency reduction vs CPU-MoE baseline | 68.4% |

The 3.17× result assumes highly overlapped GPU, RAM, and NVMe service paths and
a 68.25% GPU Expert-service fraction. Real performance depends on measured
Router locality, MXFP4 kernels, RAM bandwidth, PCIe behavior, NVMe tail latency,
and synchronization overhead. See
[`docs/COMPUTE_SAVINGS.md`](docs/COMPUTE_SAVINGS.md).

## Included

```text
src/lexa_hypermoe/
├── model.py          # equations, tier optimizer, Zipf/deadline helpers
├── trace.py          # privacy-safe route audit, holdout, temporal locality
├── cache_sim.py      # causal LRU, EMA, and warm-up/static cache policies
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
├── run_router_trace_suite.py
└── analyze_router_trace.py

data/
├── model/            # official constants + derived traffic estimates
├── hardware/         # explicit assumption profiles
├── synthetic/        # formula-generated sweeps and a synthetic trace example
├── workloads/        # public non-sensitive cross-domain trace prompts
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
routing, `ggml_mul_mat_id`, CUDA kernels, or sampling. The executable links
`llama-common`, so the standard `llama.cpp` runtime controls remain available,
including `--cpu-moe`, `--n-cpu-moe`, automatic GPU fitting, lazy tensor loading,
KV-cache types, batch sizes, threads, devices, and tensor overrides.

## Collect decode routing

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --model-id HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive \
  --output data/observed/coding-session-001.jsonl \
  --prompt-file /private/prompts/coding-session-001.txt \
  --domain coding \
  --n-predict 512 \
  --ctx-size 4096 \
  --gpu-layers auto \
  --n-cpu-moe 36 \
  --fit on \
  --fit-target 1536 \
  --lazy-mode on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

The trace stores processed-token sequence indexes and Expert IDs, not prompt
text, generated text, token IDs, logits, probabilities, embeddings, or hidden
states. Decode rows describe the token positions actually evaluated by the model;
the first sampled output has no separate decode row until it is fed back as input.
The sidecar manifest stores no absolute paths, command line, raw stderr, or prompt hash by
default. It records only prompt byte length; `--record-prompt-hash` is an
explicit opt-in.

Audit the trace:

```bash
lexa-hypermoe trace-audit \
  --trace data/observed/coding-session-001.jsonl
```

Measure a top-18 cache with chronological 70/30 holdout, trace-quality checks,
and temporal locality:

```bash
lexa-hypermoe trace \
  --trace data/observed/coding-session-001.jsonl \
  --phase decode \
  --cached-experts-per-layer 18 \
  --calibration-fraction 0.70 \
  --require-quality
```

Compare cache sizes:

```bash
lexa-hypermoe trace-sweep \
  --trace data/observed/coding-session-001.jsonl \
  --phase decode \
  --cache-sizes 4,8,12,16,18,20,24,32 \
  --calibration-fraction 0.70 \
  --require-quality
```

The single-trace report also includes three causal policies that never use a
future event to decide the current residency:

- online LRU;
- online exponentially-decayed frequency (EMA);
- static hot set learned on an initial warm-up prefix and scored on the suffix.

Run the included public, non-sensitive workload suite:

```bash
python scripts/run_router_trace_suite.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --model-id HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive \
  --output-dir data/observed/public-suite-001 \
  --gpu-layers auto \
  --n-cpu-moe 36 \
  --ctx-size 4096 \
  --cached-experts-per-layer 18
```

The suite report stores workload IDs, prompt hashes, file basenames, process
output hashes, and analysis results. It does not embed prompt text, absolute
paths, raw stdout, or raw stderr.

Full instructions: [`docs/ROUTER_TRACE.md`](docs/ROUTER_TRACE.md).

For the existing GitHub Pages monorepo layout, copy
[`integrations/github-pages/lexa-hypermoe-ci.yml`](integrations/github-pages/lexa-hypermoe-ci.yml)
to the repository-root `.github/workflows/lexa-hypermoe-ci.yml`.

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
- optimistic in-sample and primary chronological-holdout top-N cache analysis;
- causal online LRU, online EMA, and warm-up/static cache simulations;
- public multi-domain Router Trace workload suite and privacy-minimized suite runner;
- trace geometry checks for 36 layers, Top-4 routing, Expert range, and complete grids;
- consecutive-token Expert-overlap analysis;
- privacy-minimized run provenance manifest;
- tests and CI for Python 3.10–3.12;
- CI build validation of the patched `llama-router-trace` target;
- inheritance of upstream CPU-MoE, fit, lazy-loading, KV, batch, thread, device,
  and tensor-placement controls through `llama-common`.

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
