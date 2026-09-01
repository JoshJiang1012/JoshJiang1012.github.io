# Suggested use cases

## Local AI and sparse inference

Use the GPU for mandatory dense kernels and hot operators, CPU/RAM for warm operators, and storage only for capacity overflow. Calibrate actual kernel efficiency before making throughput claims.

## Edge-cloud systems

Represent the local processor and remote accelerator as devices with an explicit network link. Include upload/download bytes and intermittent-link scenarios.

## Media pipelines

Model decode, transforms and inference as stages. A CPU may outperform a GPU for control-heavy decode while the GPU dominates tensor compute.

## Build and test orchestration

Treat workers as devices with different compile throughput and artifact-transfer links. Capacity can represent available workspace or cache.

## Scientific workflows

Separate mandatory reductions or serial coordination from divisible simulation work. Use energy weighting on battery- or budget-limited nodes.

## Defensive security analytics

In authorized environments, deterministic parsers, static analyzers and log normalization can run on CPU tiers while model-assisted correlation uses accelerator capacity. The project does not grant access to targets and contains no exploitation workflow.

## When not to use it

Do not treat the analytical model as a replacement for kernel profiling, network emulation, real-time verification or production load testing.
