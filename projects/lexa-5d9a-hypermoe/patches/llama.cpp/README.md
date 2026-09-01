# llama.cpp router-trace patch

This directory contains a pinned, reviewable patch for upstream `llama.cpp`.

- **Upstream:** `ggml-org/llama.cpp`
- **Pinned commit:** `d08c7872d6ffe3f059f8647840a29aa390413e27`
- **Added target:** `llama-router-trace`
- **Patch:** `d08c7872-router-trace.patch`

## Why this hook is small

At the pinned commit, the generic MoE graph builder names the selected-expert
I32 tensor `ffn_moe_topk-<layer>`. The public context evaluation callback can
request only that tensor after computation. The patch therefore adds a dedicated
example executable and does not modify GPT-OSS weights, CUDA kernels, routing
logic, or sampling behavior.

The callback copies only `n_expert_used × n_tokens` 32-bit expert IDs per layer.
For GPT-OSS-120B decode this is 4 IDs × 36 layers, or 576 bytes of raw IDs per
generated token before JSON encoding.

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout d08c7872d6ffe3f059f8647840a29aa390413e27

python ../lexa-5d9a-hypermoe/scripts/verify_llama_patch.py \
  --llama-source . \
  --apply

cmake -S . -B build \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON
cmake --build build --target llama-router-trace -j
```

The binary location depends on the CMake generator. Common paths include:

```text
build/bin/llama-router-trace
build/bin/Release/llama-router-trace.exe
```

## Collect decode-only routing data

Use the Python wrapper so the prompt is placed in a mode-0600 temporary file and
is represented in the run manifest only by its SHA-256 digest and byte length.

```bash
python scripts/collect_router_trace.py \
  --binary /path/to/llama-router-trace \
  --model /models/gpt-oss-120b.gguf \
  --output data/observed/coding-session-001.jsonl \
  --prompt-file prompts/coding.txt \
  --domain coding \
  --n-predict 4096 \
  --ctx-size 8192 \
  --gpu-layers 99
```

By default only decode routing is written. Add `--trace-prefill` only for a
separate prefill study.

## Trace performance warning

JSONL writing and GPU-to-host retrieval add overhead. Do not report token speed
from a trace-enabled run as the uninstrumented model benchmark. Use routing traces
to design the cache, then benchmark the runtime again with tracing disabled.

## Privacy boundary

The collector records:

```json
{"schema_version":"2.0","token":42,"layer":7,"experts":[3,19,81,122],"domain":"coding","phase":"decode","batch_size":1,"source":"llama.cpp:ffn_moe_topk"}
```

It does not record prompt text, generated text, token IDs, logits, probabilities,
embeddings, or hidden states. The Python ingestion path rejects those fields by
default.
