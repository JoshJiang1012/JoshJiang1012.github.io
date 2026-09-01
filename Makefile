.PHONY: test data validate benchmark

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

data:
	PYTHONPATH=src python scripts/generate_reference_data.py

validate:
	PYTHONPATH=src python scripts/validate_release.py

benchmark:
	PYTHONPATH=src python scripts/benchmark_host.py --output data/observed/host.local.json
