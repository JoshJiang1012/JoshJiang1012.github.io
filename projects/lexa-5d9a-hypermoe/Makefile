.PHONY: data test estimate

data:
	python scripts/generate_data.py

test:
	python -m unittest discover -s tests -v

estimate:
	python -m lexa_hypermoe.cli estimate \
		--model data/model/gpt_oss_120b.json \
		--hardware data/hardware/rtx4080_i7_13700_32gb_assumed.json
