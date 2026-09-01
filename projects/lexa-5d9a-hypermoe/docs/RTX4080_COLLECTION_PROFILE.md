# RTX 4080 16GB + 32GB RAM Router Trace profile

This profile is for collecting **real GPT-OSS-120B Expert-routing metadata** on a
consumer host. It is not a throughput claim and should not be used as the final
performance benchmark configuration.

## Why CPU-MoE is required

GPT-OSS-120B has 36 MoE layers and its complete weights do not fit in 16 GiB of
VRAM. At the pinned `llama.cpp` revision:

```text
-cmoe,  --cpu-moe       keep all MoE weights on CPU
-ncmoe, --n-cpu-moe N   keep MoE weights of the first N layers on CPU
```

For the first trace collection on an RTX 4080, use:

```text
--gpu-layers 99
--n-cpu-moe 36
```

This asks `llama.cpp` to keep all 36 Expert layers on the CPU/mmap path while
offloading other eligible tensors to the GPU. With only 32GB of RAM, the full
model cannot remain resident in memory, so the run may page from NVMe and be
slow. That is acceptable for collecting the Router distribution; speed must be
measured separately with tracing disabled.

## Stage 1: smoke trace

Start with a short context and output to prove that the model loads and the trace
schema is valid:

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --output data/observed/smoke-001.jsonl \
  --prompt-file /private/prompts/smoke-001.txt \
  --domain smoke \
  --n-predict 128 \
  --ctx-size 2048 \
  --gpu-layers 99 \
  --n-cpu-moe 36 \
  --timeout-seconds 7200
```

Then validate:

```bash
lexa-hypermoe trace-audit \
  --trace data/observed/smoke-001.jsonl
```

Expected structural checks for GPT-OSS-120B decode include:

- 36 observed layers per decoded position;
- 4 distinct Expert IDs per layer event;
- Expert IDs in the range 0–127;
- no privacy-forbidden or unknown fields.

## Stage 2: useful session

After the smoke trace succeeds, collect 2,048 decode positions:

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --output data/observed/coding-python-001.jsonl \
  --prompt-file /private/prompts/coding-python-001.txt \
  --domain coding-python \
  --n-predict 2048 \
  --ctx-size 4096 \
  --gpu-layers 99 \
  --n-cpu-moe 36 \
  --model-sha256 MODEL_DIGEST_IF_ALREADY_KNOWN \
  --timeout-seconds 14400
```

## Analysis

```bash
lexa-hypermoe trace \
  --trace data/observed/coding-python-001.jsonl \
  --phase decode \
  --cached-experts-per-layer 18

lexa-hypermoe trace-sweep \
  --trace data/observed/coding-python-001.jsonl \
  --phase decode \
  --cache-sizes "4,8,12,16,18,20,24,32"
```

The most important outputs are:

- `per_selection_hit_rate`;
- `all_selected_hit_rate`;
- per-layer hot Expert identities;
- consecutive-position Expert overlap;
- exact top-k-set repeat rate.

`all_selected_hit_rate` is the stricter metric because one uncached selected
Expert can create a blocking cold load.

## Data-handling boundary

The trace does not store prompt or generated text, token IDs, logits, routing
probabilities, embeddings, residuals, or hidden states. The run sidecar stores a
prompt SHA-256 and byte count, plus runtime provenance. Use non-sensitive or
locally approved prompts because the model still processes the input even though
the collector does not persist it.

## Before publishing observed data

Record:

- exact GGUF SHA-256;
- patched binary SHA-256;
- `llama.cpp` commit;
- NVIDIA driver and CUDA backend;
- CPU, RAM channel/layout and NVMe model;
- context, batch, GPU-layer and CPU-MoE settings;
- warm-up policy;
- trace and run-manifest SHA-256;
- P50/P95/P99 latency from a separate trace-disabled benchmark.
