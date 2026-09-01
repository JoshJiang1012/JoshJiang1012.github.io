# Contributing

Contributions are welcome when they preserve the separation between:

1. **derived analytical data** generated from declared equations and assumptions;
2. **observed router metadata** collected under the privacy schema;
3. **observed benchmark data** produced by a documented real system;
4. **external reference data** copied or summarized with provenance.

## Router traces

Before submitting a trace:

```bash
lexa-hypermoe trace-audit --trace your-trace.jsonl
```

Do not submit prompts, generated text, token IDs, logits, probabilities,
embeddings, hidden states, credentials, customer code, or proprietary labels.
Include the sidecar run manifest, model/runtime hashes, collection settings, and
an explanation of whether the prompt is public or private.

## Benchmarks

Observed benchmark submissions should include hardware, OS, runtime commit,
model-file hash, context size, batch size, warm-up length, measurement window,
P50/P95/P99, and raw timing logs. Do not submit a single peak number as a
benchmark result. Benchmark with route tracing disabled.

## Before opening a pull request

```bash
python -m pip install -e .
python scripts/generate_data.py
python -m unittest discover -s tests -v
```

If modifying the pinned patch, update `patches/llama.cpp/manifest.json`, verify the
new digest, and ensure CI builds `llama-router-trace` against the declared
upstream commit.
