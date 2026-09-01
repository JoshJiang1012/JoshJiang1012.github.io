# Contributing

Contributions are welcome.

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/generate_reference_data.py
PYTHONPATH=src python scripts/validate_release.py
```

Please include:

- tests for new equations or constraints;
- units for every numeric field;
- evidence classification for new data;
- no prompts, generated text, credentials or private code in route traces;
- measured target details for hardware claims.
