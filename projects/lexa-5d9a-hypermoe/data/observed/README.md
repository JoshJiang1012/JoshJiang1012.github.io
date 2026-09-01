# Observed data

This directory is intentionally empty in release 0.2.0.

Observed submissions should be added only after:

1. collecting a trace with the pinned `llama-router-trace` binary;
2. running `lexa-hypermoe trace-audit` successfully;
3. including the `.run.json` sidecar manifest;
4. removing private paths or identifiers from public metadata;
5. documenting model/runtime hashes and collection settings;
6. keeping trace-enabled routing runs separate from trace-disabled speed benchmarks.

Do not add model weights, prompts, generated text, token IDs, logits, hidden
states, customer code, credentials, or proprietary filenames.
