# Experiment protocol for observed results

An observed result should include all of the following.

## Hardware

- CPU exact model;
- motherboard and memory type, channel count, capacity, speed, and timings;
- GPU model, VRAM, driver, power limit, and clocks;
- NVMe model and filesystem;
- thermals and power mode.

## Software

- operating system and kernel;
- inference runtime repository and commit SHA;
- router patch SHA-256 and patched binary SHA-256;
- compiler, CMake, CUDA, and backend versions;
- model repository, file name, byte size, and SHA-256;
- exact command line and non-secret environment variables.

## Workload

- public prompt identifier or private prompt SHA-256 and byte count;
- coarse non-sensitive workload domain;
- context length;
- generated-token target;
- batch/parallel settings;
- reasoning effort and sampler configuration;
- cold-start and warm-up procedure;
- whether prefill routing was collected.

## Router-trace measurements

- route file and sidecar manifest SHA-256;
- event, token, and layer counts;
- static top-N per-selection hit rate;
- static top-N all-selected hit rate;
- consecutive-token overlap and exact-repeat rate;
- per-layer results, not only a global average;
- trace truncation or collection failures;
- privacy audit result.

## Performance measurements

Use a separate trace-disabled run for performance:

- time to first token;
- prompt processing throughput;
- decode P50/P95/P99 token latency;
- steady-state tokens/s;
- GPU/RAM/NVMe utilization;
- dynamic cache hit rates by layer and tier, when implemented;
- critical misses and page-fault counts;
- peak VRAM, RAM, and swap usage;
- output correctness comparison against a reference runtime.

At least three runs should be reported. Do not discard failed or slow runs without
stating the exclusion rule. Never combine trace-enabled route data with
trace-disabled timing and imply they came from one unmodified execution.
