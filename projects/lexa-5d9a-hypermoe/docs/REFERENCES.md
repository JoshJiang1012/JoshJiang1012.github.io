# References

Primary references used by the repository:

1. OpenAI GPT-OSS repository: https://github.com/openai/gpt-oss
2. GPT-OSS-120B official configuration: https://huggingface.co/openai/gpt-oss-120b/blob/main/original/config.json
3. Official MXFP4 packing reference (`BYTES_PER_BLOCK = 16` for 32 FP4 values): https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
4. llama.cpp two-tier expert-cache feature discussion and proof of concept: https://github.com/ggml-org/llama.cpp/issues/20757
5. llama.cpp main repository: https://github.com/ggml-org/llama.cpp

Secondary reports should not override official architecture constants. Any
benchmark copied into `data/observed` must preserve its source and test context.
