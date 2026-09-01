# Mathematical model

## 1. Scope

This model estimates batch-one autoregressive decode throughput for a
heterogeneous MoE runtime. It is intended to answer questions such as:

- how many layer-specific experts fit in a VRAM cache;
- what cache skew is required for a target hit rate;
- how expert traffic could be divided among GPU, CPU/RAM, and NVMe tiers;
- how far ahead an expert must be prefetched;
- how rare a blocking miss must be at a target tail-latency objective.

It is not a replacement for an implementation benchmark.

## 2. GPT-OSS-120B constants

The reference configuration uses:

| Symbol | Value |
|---|---:|
| layers `L` | 36 |
| experts per layer `E` | 128 |
| active experts `K` | 4 |
| hidden size `d` | 2880 |
| expert intermediate size `m` | 2880 |
| total parameters | about 116.83B |
| active parameters/token | about 5.1B |

The approximation for one expert is:

\[
P_e = 3dm
\]

This represents gate, up, and down projection matrices and omits comparatively
small biases.

For MXFP4, 32 values occupy 16 packed bytes and have one scale byte per block:

\[
q = \frac{17}{32}\ \text{bytes/weight}.
\]

Thus:

\[
S_e = P_e q.
\]

The active expert traffic per token is:

\[
A = LKS_e.
\]

The estimated non-expert active parameter count is:

\[
P_D = \max(0, P_{active} - LKP_e).
\]

Assuming BF16 storage for that remainder:

\[
D = 2P_D.
\]

The code derives these values rather than hard-coding the resulting byte totals.

## 3. Memory capacity

For total VRAM `C_G`, reserved non-cache VRAM `C_R`, and expert size `S_e`, the
upper bound on resident expert slots is:

\[
N_G = \left\lfloor \frac{C_G - C_R}{S_e} \right\rfloor.
\]

`C_R` must include dense model state, compute buffers, KV cache, CUDA allocator
headroom, and display/driver use. The included profile deliberately makes this an
input rather than claiming a universally correct value.

## 4. Three-tier critical path

Let expert traffic fractions be:

\[
h_G + h_R + h_N = 1.
\]

Let effective path bandwidths be `B_G`, `B_R`, and `B_N`. They must include
real kernel, dequantization, staging, and software efficiency—not vendor peak
numbers unless constructing an explicit upper bound.

\[
T_G = \frac{D + h_GA}{B_G}
\]

\[
T_R = \frac{h_RA}{B_R}
\]

\[
T_N = \frac{h_NA}{B_N}
\]

If all paths overlap perfectly:

\[
T_{critical} = \max(T_G, T_R, T_N).
\]

The simulator adds fixed overhead and an expected blocking-miss penalty:

\[
T_{token} = T_{critical} + T_{fixed}
+ Lp_{miss}T_{miss}.
\]

Finally:

\[
\text{tokens/s} = \frac{1}{T_{token}}.
\]

The expectation term is useful for sweeps but does not capture P99 tail shape.
Observed submissions should publish latency distributions.

## 5. Idealized continuous balance

Ignoring capacity and fixed costs, an optimistic balanced solution satisfies:

\[
\frac{D+h_GA}{B_G}
= \frac{h_RA}{B_R}
= \frac{h_NA}{B_N}
= T^*.
\]

This gives:

\[
T^* = \frac{D+A}{B_G+B_R+B_N}.
\]

The resulting fractions may be negative or violate cache capacity, so the CLI
uses a grid search as the safer default.

## 6. Zipf sensitivity

Before real router traces exist, the expert popularity distribution can be
explored using:

\[
p_r = \frac{r^{-s}}{\sum_{j=1}^{E}j^{-s}}.
\]

The probability mass of the hottest `C` experts is:

\[
H(C,s)=\sum_{r=1}^{C}p_r.
\]

The CSV also reports `H^K` as a simple independence approximation that all four
selected experts are resident. Actual top-k router selections are correlated, so
trace-derived values are preferred.

## 7. Layer deadlines and prefetch

At target throughput `R`:

\[
T_{token}=\frac{1000}{R}\text{ ms}
\]

and the average layer budget is:

\[
T_{layer}=\frac{1000}{RL}\text{ ms}.
\]

For load latency `T_load`, a minimum prefetch horizon is:

\[
H=\left\lceil\frac{T_{load}}{T_{layer}}\right\rceil.
\]

This is only a scheduling lower bound; it does not prove that future router
choices are predictable at that horizon.

## 8. Critical-miss probability

Under an independence approximation, if a token must have no critical miss with
probability `P_clean`, the maximum per-layer critical-miss probability is:

\[
p_{layer} \le 1-P_{clean}^{1/L}.
\]

The generated dataset also includes the simpler union-bound approximation:

\[
p_{layer} \lesssim \frac{1-P_{clean}}{L}.
\]

## 9. 5D9A scheduling state

A future runtime may represent the scheduler context as:

\[
z_t=[S_t,T_t,R_t,P_t,E_t],
\]

where the components describe semantic domain, temporal routing history,
relations, workload profile, and epistemic uncertainty. The current repository
does not train this predictor. It provides trace schemas and cache metrics needed
to evaluate one without modifying model quality.

## 10. Valid and invalid claims

Valid:

- “Under these effective-bandwidth assumptions, the analytical model predicts X.”
- “The router trace achieved Y% top-18 per-selection cache hits.”
- “The measured run produced P50/P95/P99 values under this protocol.”

Invalid:

- “RTX 4080 runs GPT-OSS-120B at 90 tok/s” based only on the synthetic sweep.
- “The full model fits in 16GB” because only active experts are counted.
- “A 98% cache hit guarantees 90 tok/s” without kernel and tail-latency evidence.
