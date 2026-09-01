# Open-source release contract

Lexa 5D9A HyperMoE v1.0.0 is published under Apache-2.0 as a reproducible
research toolkit. The repository includes mathematical models, synthetic
parameter sweeps, privacy-preserving Router Trace instrumentation, cache-policy
simulators, tests, and CI. Model weights are not included.

## Verified software scope

- Python analytical and Router Trace tests run in CI.
- Generated analytical datasets are required to be deterministic.
- The pinned `llama.cpp` patch is rebuilt in CI against commit
  `d08c7872d6ffe3f059f8647840a29aa390413e27`.
- Trace schemas reject prompt text, generated text, token IDs, logits, hidden
  states, and unknown fields by default.

## Not yet verified

- GPT-OSS-120B loading on the target RTX 4080 / 32 GiB host.
- Observed per-layer Expert distributions from that model.
- Dynamic GPU Expert residency or asynchronous NVMe paging.
- Observed 80 or 90 token/s performance.

Any throughput number without an observed-run manifest, model/runtime hashes,
and latency distribution must remain labeled `analytical_estimate`.
