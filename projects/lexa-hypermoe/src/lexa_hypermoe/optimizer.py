from __future__ import annotations

from .model import Estimate, HardwareSpec, ModelSpec, Placement, estimate


def optimize(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    step: float = 0.0025,
    gpu_fraction_cap: float = 1.0,
) -> Estimate:
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be within (0, 1]")
    if not 0.0 <= gpu_fraction_cap <= 1.0:
        raise ValueError("gpu_fraction_cap must be within [0, 1]")

    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1 exactly")

    best: Estimate | None = None
    max_gpu_units = min(units, int(gpu_fraction_cap / step + 1e-9))
    for gpu_units in range(max_gpu_units + 1):
        for ram_units in range(units - gpu_units + 1):
            nvme_units = units - gpu_units - ram_units
            candidate = estimate(
                model,
                hardware,
                Placement(
                    gpu=gpu_units * step,
                    ram=ram_units * step,
                    nvme=nvme_units * step,
                ),
            )
            if best is None or candidate.total_ms < best.total_ms:
                best = candidate
    assert best is not None
    return best
