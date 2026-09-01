# Data card

## Dataset classes

### `synthetic_analytical`

Rows generated directly by the equations in `src/lexa_hypermoe/model.py`. These
are reproducible parameter sweeps, not machine measurements.

### `synthetic_distribution`

Zipf-distribution sensitivity data. It explores possible expert skew but does
not assert the true GPT-OSS router distribution.

### `synthetic_latency`

Prefetch horizons computed from declared target throughput and assumed load
latencies.

### `analytical_requirement`

Probability or capacity requirements derived from formulas.

### `observed`

Reserved for real runs. No observed GPT-OSS-120B RTX 4080 benchmark is bundled in
version 0.1.0.

## Generation

```bash
python scripts/generate_data.py
```

Generation is deterministic and uses only the Python standard library.

## Intended use

- research planning;
- sensitivity analysis;
- runtime design reviews;
- comparing trace-derived cache policies;
- identifying which measurements are needed before implementing a runtime.

## Out-of-scope use

- advertising synthetic values as hardware benchmarks;
- inferring model quality from throughput equations;
- claiming an exact speed without raw logs and a documented runtime;
- training on confidential prompts embedded in router traces.

## Privacy

Router traces should contain numeric expert IDs and timing metadata only. Do not
publish prompts, completions, credentials, customer code, or private filenames.
