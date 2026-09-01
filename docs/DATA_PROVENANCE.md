# Data provenance policy

Every published result must carry a classification.

## Observed measurement

Required fields:

- timestamp in UTC;
- environment identifier;
- hardware/runtime visibility;
- benchmark parameters;
- measured values;
- limitations.

## Observed synthetic harness

A real wall-clock measurement on generated input. It validates software throughput, memory use or correctness, but does not establish production workload behavior.

## Analytical estimate

The output of equations using declared assumptions. It must include input profiles and cannot use words such as “measured”, “achieved” or “benchmark” unless a separate observed run exists.

## Analytical proxy

A reduced quantity such as active-parameter ratio or weight-traffic reduction. A proxy is not interchangeable with power, joules, FLOPs or end-to-end speed.

## Forbidden evidence mixing

Do not:

- merge synthetic and real routes into one hit-rate headline;
- replace effective bandwidth with vendor peak bandwidth without changing the classification;
- cite CI success as proof of accelerator throughput;
- omit failed or infeasible placements from methodology;
- publish private prompts or generated text in routing traces.
