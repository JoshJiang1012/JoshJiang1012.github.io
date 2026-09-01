# Contributing

Contributions are welcome when they preserve the separation between:

1. **derived analytical data** generated from declared equations and assumptions;
2. **observed benchmark data** produced by a documented real system;
3. **external reference data** copied or summarized with provenance.

Observed submissions should include hardware, OS, runtime commit, model file
hash, context size, batch size, warm-up length, measurement window, P50/P95/P99,
and raw logs. Do not submit a single peak number as a benchmark result.

Run before opening a pull request:

```bash
python scripts/generate_data.py
python -m unittest discover -s tests -v
```
