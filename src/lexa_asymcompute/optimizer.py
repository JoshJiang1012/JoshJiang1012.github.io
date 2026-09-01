from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

from .models import Estimate, FractionalPlacement, HardwareSystem, WorkloadSpec, estimate


@dataclass(frozen=True)
class ObjectiveWeights:
    latency: float = 1.0
    energy: float = 0.0

    def validate(self) -> None:
        if self.latency < 0 or self.energy < 0 or self.latency + self.energy <= 0:
            raise ValueError("objective weights must be non-negative and not both zero")


def _integer_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for cuts in combinations(range(total + parts - 1), parts - 1):
        points = (-1,) + cuts + (total + parts - 1,)
        yield tuple(points[i + 1] - points[i] - 1 for i in range(parts))


def placements(device_names: tuple[str, ...], step: float) -> Iterator[FractionalPlacement]:
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("step must evenly divide one, e.g. 0.1, 0.05, 0.02, 0.01")
    for composition in _integer_compositions(units, len(device_names)):
        yield FractionalPlacement(
            {name: count / units for name, count in zip(device_names, composition, strict=True)}
        )


def optimize(
    workload: WorkloadSpec,
    hardware: HardwareSystem,
    *,
    step: float = 0.01,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> Estimate:
    weights.validate()
    known = hardware.device_map()
    device_names = tuple(workload.eligible_devices or known.keys())
    if not device_names:
        raise ValueError("no eligible devices")

    best: Estimate | None = None
    best_score = float("inf")
    for placement in placements(device_names, step):
        try:
            candidate = estimate(workload, hardware, placement)
        except ValueError as exc:
            if "capacity" in str(exc):
                continue
            raise
        score = weights.latency * candidate.total_ms + weights.energy * candidate.energy_j
        if score < best_score - 1e-12:
            best = candidate
            best_score = score
    if best is None:
        raise ValueError("no feasible placement fits the declared device capacities")
    return best
