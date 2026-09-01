# Router Trace v2: real Expert-routing data

Version 0.2.0 moves the project from a purely analytical model to the first
runtime-observation stage. The goal is to measure whether GPT-OSS-120B routing is
sufficiently concentrated and temporally local for a per-layer hot-Expert cache
to be useful on an RTX 4080.

## Upstream hook

The pinned upstream revision is:

```text
repository: ggml-org/llama.cpp
commit: d08c7872d6ffe3f059f8647840a29aa390413e27
```

At this revision, the generic MoE graph builder creates the selected Expert IDs
with `ggml_argsort_top_k` and names the resulting tensor
`ffn_moe_topk-<layer>`. GPT-OSS uses that generic builder from
`src/models/openai-moe.cpp`.

The patch adds `llama-router-trace`, a dedicated example that installs the public
`llama_context_params.cb_eval` callback. During the callback's `ask=true` phase,
it requests only tensors whose name starts with `ffn_moe_topk-`. All other graph
tensors are ignored.

For GPT-OSS-120B decode:

```text
4 Expert IDs/layer × 36 layers × 4 bytes = 576 raw bytes/token
```

JSONL encoding is larger and adds measurement overhead, so route collection and
uninstrumented speed benchmarking must be separate runs.

## Privacy contract

Allowed trace fields:

- schema version;
- sequence index, not token ID;
- layer index;
- selected Expert IDs;
- prefill/decode phase;
- batch size;
- coarse workload domain label;
- collector source identifier;
- optional timing and cache-tier counters in later runtime versions.

Explicitly forbidden:

- prompt or generated text;
- token IDs or token pieces;
- logits or routing probabilities;
- embeddings, residuals, or hidden states;
- credentials, API keys, or secrets.

The Python parser uses a top-level field allowlist by default. A row containing a
forbidden or unknown field fails validation instead of being silently accepted.

## Build the collector

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

## Recommended collection matrix

Do not begin with a single prompt. Collect multiple independent sessions for each
workload class.

| Domain | Sessions | Decode tokens/session | Purpose |
|---|---:|---:|---|
| coding-python | 8 | 2,048–4,096 | common local development |
| coding-rust | 8 | 2,048–4,096 | compiler-heavy code reasoning |
| security-defensive | 8 | 2,048–4,096 | authorized defensive analysis |
| mathematics | 8 | 2,048–4,096 | non-code reasoning control |
| general | 8 | 2,048–4,096 | general-language control |

Use non-sensitive prompts or locally approved benchmark inputs. The trace itself
will not contain prompt text, but the model still processes the prompt.

## Collection command

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --output data/observed/coding-python-001.jsonl \
  --prompt-file /private/prompts/coding-python-001.txt \
  --domain coding-python \
  --n-predict 4096 \
  --ctx-size 8192 \
  --gpu-layers 99 \
  --model-sha256 MODEL_DIGEST_IF_ALREADY_KNOWN
```

The wrapper passes the prompt through a temporary mode-0600 file, deletes the
file after execution, and stores only the prompt SHA-256 and UTF-8 byte count in
the sidecar run manifest.

## Audit before analysis

```bash
lexa-hypermoe trace-audit \
  --trace data/observed/coding-python-001.jsonl
```

The audit computes:

- trace SHA-256;
- number of events, unique token positions, and layers;
- number of Expert selections;
- observed phase/domain/schema values;
- maximum Expert ID;
- privacy allowlist status.

## Cache-size sweep

```bash
lexa-hypermoe trace-sweep \
  --trace data/observed/coding-python-001.jsonl \
  --phase decode \
  --cache-sizes 4,8,12,16,18,20,24,32
```

Two different hit rates are reported:

1. `per_selection_hit_rate`: fraction of individual Expert selections found in
   the static top-N cache;
2. `all_selected_hit_rate`: fraction of layer/token events for which all selected
   Experts were cached.

The second metric is stricter and more relevant to avoiding a blocking cold
Expert load.

## Temporal locality

`lexa-hypermoe trace` also measures, per layer:

- mean fraction of selected Experts shared with the next token;
- exact top-k-set repeat rate.

High static hit rate supports a domain cache. High temporal overlap supports a
session-adaptive cache. They are related but not interchangeable.

## Minimum evidence for a published result

A pull request adding observed data should include:

- model repository and exact model-file SHA-256;
- `llama.cpp` commit and patched binary SHA-256;
- GPU driver, CUDA/backend configuration, CPU, RAM layout, and NVMe device;
- context size, batch settings, GPU layer settings, and warm-up policy;
- trace and run-manifest hashes;
- session count and domain labels;
- P50/P95/P99 latency from a separate trace-disabled benchmark;
- clear separation among observed, derived, and synthetic values.

## What v0.2.0 does not do

It does not yet:

- move Experts dynamically between VRAM and RAM;
- predict future router decisions;
- change `ggml_mul_mat_id` execution;
- add fused MXFP4 CPU/GPU kernels;
- perform NVMe asynchronous Expert paging;
- prove any GPT-OSS-120B token/s result on RTX 4080.

The trace is the evidence needed to decide whether those runtime changes are
worth implementing.
