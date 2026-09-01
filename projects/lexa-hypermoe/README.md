# Lexa HyperMoE

A small, reproducible open-source toolkit for studying heterogeneous placement
of Mixture-of-Experts inference across GPU VRAM, system RAM, and NVMe.

## What it provides

- a deterministic GPT-OSS-120B active-weight traffic model;
- GPU/RAM/NVMe critical-path estimation;
- constrained grid-search placement optimization;
- analytical MoE compute-savings proxies;
- privacy-preserving Router Trace parsing and validation;
- temporal 70/30 Top-N Expert holdout evaluation;
- a zero-runtime-dependency Python CLI and unit tests.

## Evidence boundary

The bundled RTX 4080 result is an **analytical estimate, not an observed
benchmark**. This repository does not claim that GPT-OSS-120B reached 80 or 90
token/s on an RTX 4080. Hardware claims require real model weights, exact
runtime and model hashes, measured Router Trace data, warm-up policy, and
P50/P95/P99 latency.

## Reference analytical result

With the included assumptions:

```text
CPU-MoE baseline                 25.27 token/s
optimized heterogeneous model    80.06 token/s
analytical speedup                 3.17x
placement            68.25% GPU / 28.75% RAM / 3.00% NVMe
```

The MoE geometry itself activates 5.1B of 116.8292B parameters per token and 4
of 128 Experts per layer. The corresponding proxy reductions are documented in
[`docs/COMPUTE_SAVINGS.md`](docs/COMPUTE_SAVINGS.md).

## Install and reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v

lexa-hypermoe savings \
  --model data/model/gpt_oss_120b.json

lexa-hypermoe optimize \
  --model data/model/gpt_oss_120b.json \
  --hardware data/hardware/rtx4080_i7_13700_32gb_assumed.json

python scripts/generate_example_trace.py /tmp/lexa-hypermoe-example.jsonl

lexa-hypermoe trace \
  --trace /tmp/lexa-hypermoe-example.jsonl \
  --model data/model/gpt_oss_120b.json \
  --cache 18 \
  --calibration 0.70
```

## Router Trace privacy contract

Accepted records contain only routing metadata such as token sequence index,
layer index, and selected Expert IDs. The parser rejects unknown fields by
default, preventing prompt or generated text from silently entering the trace.

## License

MIT. See [`LICENSE`](LICENSE).
