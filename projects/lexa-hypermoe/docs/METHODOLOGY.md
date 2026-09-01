# Methodology

The model separates per-token weight traffic into three concurrently serviced
paths:

```text
GPU:  dense traffic + GPU-resident Expert traffic
RAM:  CPU-serviced warm Expert traffic
NVMe: cold Expert traffic and downstream service
```

For placement fractions `g`, `r`, and `n`, where `g + r + n = 1`:

```text
T_gpu  = (D + gA) / B_gpu
T_ram  = rA / B_ram
T_nvme = nA / B_nvme
T      = max(T_gpu, T_ram, T_nvme) + fixed_overhead
```

`D` is active dense traffic, `A` is active Expert traffic, and `B` values are
effective service rates rather than vendor peak bandwidth. The optimizer uses
a deterministic grid search so every reported point is reproducible.

Router Trace evaluation is temporal: the first 70% of token indices choose the
Top-N Experts per layer, while only the remaining 30% determines the reported
holdout hit rate. Prompt text, generated text, token IDs, logits, hidden states,
and secrets are outside the trace schema.
