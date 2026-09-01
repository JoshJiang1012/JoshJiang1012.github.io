# Security policy

This repository contains analytical tooling, benchmark schemas, and a narrowly
scoped local inference instrumentation patch. It does not include model weights,
remote administration, credential handling, or an unrestricted shell agent.

The Router Trace collector requests only `ffn_moe_topk-<layer>` I32 tensors. The
published schema forbids prompts, generated text, token IDs, logits,
probabilities, embeddings, residuals, hidden states, and secrets.

Please report security or privacy problems privately to the repository owner
before public disclosure. Do not attach secrets, customer data, proprietary
prompts, model weights, or private traces to public issues.
