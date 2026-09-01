# Modelling guide

## 1. Choose the unit of work

A unit may be one inference token, one image, one video frame, one compilation target, one query batch or one simulation step. Keep all workload quantities consistent with that unit.

## 2. Measure or estimate device rates

Use representative kernels when possible. Put measured effective rates directly in `compute_tflops` and `memory_bandwidth_gbps` with efficiencies of 1.0, or use peak values with explicit efficiency factors.

## 3. Separate mandatory and divisible work

Mandatory work is fixed to a named tier. Divisible work is multiplied by placement share. This prevents the optimizer from moving an operation that a device cannot execute.

## 4. Add resident capacity

Resident bytes represent weights, state, buffers or working sets that must coexist during the stage. Capacity is enforced per device.

## 5. Add links and output bytes

When workers return data to an aggregator, declare both the link and output size. A missing link makes the placement infeasible.

## 6. Pick a grid step

For two or three devices, 0.01 is a useful first pass. Finer steps increase search cost. Validate the best region with finer resolution.

## 7. Publish evidence honestly

Store observed and analytical records separately. Include environment and parameters with every observed file.
