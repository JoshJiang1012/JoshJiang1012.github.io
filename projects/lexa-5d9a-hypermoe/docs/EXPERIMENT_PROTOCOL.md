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
- compiler and CUDA versions;
- model repository, file name, byte size, and SHA-256;
- exact command line and environment variables.

## Workload

- prompt or public prompt hash;
- context length;
- generated token count;
- batch/parallel settings;
- reasoning effort and sampler configuration;
- cold-start and warm-up procedure.

## Measurements

- time to first token;
- prompt processing throughput;
- decode P50/P95/P99 token latency;
- steady-state tokens/s;
- GPU/RAM/NVMe utilization;
- cache hit rates by layer and tier;
- critical misses and page-fault counts;
- peak VRAM, RAM, and swap usage;
- output correctness comparison against a reference runtime.

At least three runs should be reported. Do not discard failed or slow runs without
stating the exclusion rule.
