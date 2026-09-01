# References

Primary references used by the repository:

1. OpenAI GPT-OSS repository: https://github.com/openai/gpt-oss
2. GPT-OSS-120B official configuration: https://huggingface.co/openai/gpt-oss-120b/blob/main/original/config.json
3. Official MXFP4 packing reference (`BYTES_PER_BLOCK = 16` for 32 FP4 values): https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
4. llama.cpp main repository: https://github.com/ggml-org/llama.cpp
5. Pinned llama.cpp commit for Router Trace v2: https://github.com/ggml-org/llama.cpp/commit/d08c7872d6ffe3f059f8647840a29aa390413e27
6. GPT-OSS graph implementation at the pinned commit: https://github.com/ggml-org/llama.cpp/blob/d08c7872d6ffe3f059f8647840a29aa390413e27/src/models/openai-moe.cpp
7. Generic MoE selected-Expert tensor at the pinned commit: https://github.com/ggml-org/llama.cpp/blob/d08c7872d6ffe3f059f8647840a29aa390413e27/src/llama-graph.cpp
8. llama.cpp evaluation-callback example: https://github.com/ggml-org/llama.cpp/tree/d08c7872d6ffe3f059f8647840a29aa390413e27/examples/eval-callback
9. llama.cpp Expert-cache discussion and proof-of-concept work: https://github.com/ggml-org/llama.cpp/issues/20757

Secondary reports should not override official architecture constants. Any
benchmark copied into `data/observed` must preserve its source and test context.
