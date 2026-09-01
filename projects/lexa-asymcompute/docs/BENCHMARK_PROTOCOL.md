# Target-hardware benchmark protocol

A credible hardware submission should include:

1. exact CPU, GPU, RAM channels/speed, storage and operating system;
2. runtime, compiler, driver and kernel versions;
3. workload/model identifier and immutable hash;
4. context/batch/parallel settings where applicable;
5. cold-start and warmed measurements separated;
6. at least 256 warm-up units before steady-state sampling when the workload has caches;
7. P50, P95 and P99 latency, not only the best short burst;
8. peak VRAM, peak RSS, storage reads and power if available;
9. baseline and optimized runs using equivalent output/correctness constraints;
10. raw privacy-safe measurements sufficient for independent recomputation.

## Minimum repetitions

For stable stages, use at least 10 repetitions. For noisy or remote systems, use enough runs to report a confidence interval.

Mean:

$$
\bar x=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

Sample standard deviation:

$$
s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}.
$$

Approximate 95% confidence interval for sufficiently regular samples:

$$
\bar x\pm t_{0.975,n-1}\frac{s}{\sqrt n}.
$$

Tail-sensitive systems should publish empirical percentiles even when the mean is reported.
