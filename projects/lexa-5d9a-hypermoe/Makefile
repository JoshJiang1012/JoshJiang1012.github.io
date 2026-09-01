.PHONY: data test estimate trace-example patch-check

data:
	python scripts/generate_data.py

test:
	python -m unittest discover -s tests -v

estimate:
	python -m lexa_hypermoe.cli estimate \
		--model data/model/gpt_oss_120b.json \
		--hardware data/hardware/rtx4080_i7_13700_32gb_assumed.json

trace-example:
	python -m lexa_hypermoe.cli trace \
		--trace data/synthetic/router_trace_example.jsonl \
		--phase decode \
		--cached-experts-per-layer 2

patch-check:
	@test -n "$(LLAMA_SOURCE)" || (echo "Set LLAMA_SOURCE=/path/to/llama.cpp" && exit 2)
	python scripts/verify_llama_patch.py --llama-source "$(LLAMA_SOURCE)"
