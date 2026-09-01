from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoEModelSpec:
    name: str
    total_parameters_b: float
    active_parameters_b: float
    layers: int
    experts_per_layer: int
    active_experts_per_token: int
    dense_traffic_gb_per_unit: float
    expert_traffic_gb_per_unit: float

    @classmethod
    def from_json(cls, path: str | Path) -> "MoEModelSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MoEHardwareSpec:
    name: str
    gpu_effective_gbps: float
    ram_effective_gbps: float
    nvme_effective_gbps: float
    fixed_overhead_ms: float
    expert_cache_gib: float

    @classmethod
    def from_json(cls, path: str | Path) -> "MoEHardwareSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MoEPlacement:
    gpu: float
    ram: float
    nvme: float

    def validate(self) -> None:
        values = (self.gpu, self.ram, self.nvme)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("placement fractions must be in [0, 1]")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("placement fractions must sum to one")


@dataclass(frozen=True)
class MoEEstimate:
    placement: MoEPlacement
    gpu_ms: float
    ram_ms: float
    nvme_ms: float
    critical_path_ms: float
    fixed_overhead_ms: float
    total_ms: float
    units_per_second: float
    classification: str = "analytical_estimate_not_observed_benchmark"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["placement"] = asdict(self.placement)
        return result


def estimate_moe(model: MoEModelSpec, hardware: MoEHardwareSpec, placement: MoEPlacement) -> MoEEstimate:
    placement.validate()
    bandwidths = (
        hardware.gpu_effective_gbps,
        hardware.ram_effective_gbps,
        hardware.nvme_effective_gbps,
    )
    if min(bandwidths) <= 0 or hardware.fixed_overhead_ms < 0:
        raise ValueError("bandwidths must be positive and overhead non-negative")
    gpu_ms = 1000.0 * (
        model.dense_traffic_gb_per_unit + placement.gpu * model.expert_traffic_gb_per_unit
    ) / hardware.gpu_effective_gbps
    ram_ms = 1000.0 * placement.ram * model.expert_traffic_gb_per_unit / hardware.ram_effective_gbps
    nvme_ms = 1000.0 * placement.nvme * model.expert_traffic_gb_per_unit / hardware.nvme_effective_gbps
    critical = max(gpu_ms, ram_ms, nvme_ms)
    total = critical + hardware.fixed_overhead_ms
    return MoEEstimate(
        placement=placement,
        gpu_ms=gpu_ms,
        ram_ms=ram_ms,
        nvme_ms=nvme_ms,
        critical_path_ms=critical,
        fixed_overhead_ms=hardware.fixed_overhead_ms,
        total_ms=total,
        units_per_second=1000.0 / total,
    )


def optimize_moe(model: MoEModelSpec, hardware: MoEHardwareSpec, step: float = 0.0025) -> MoEEstimate:
    units = round(1.0 / step)
    if not 0 < step <= 1 or abs(units * step - 1.0) > 1e-9:
        raise ValueError("step must evenly divide one")
    best: MoEEstimate | None = None
    for gpu_units in range(units + 1):
        for ram_units in range(units - gpu_units + 1):
            placement = MoEPlacement(
                gpu=gpu_units / units,
                ram=ram_units / units,
                nvme=(units - gpu_units - ram_units) / units,
            )
            candidate = estimate_moe(model, hardware, placement)
            if best is None or candidate.total_ms < best.total_ms:
                best = candidate
    assert best is not None
    return best


def moe_savings(model: MoEModelSpec) -> dict[str, float | str]:
    active_parameter_ratio = model.active_parameters_b / model.total_parameters_b
    expert_active_ratio = model.active_experts_per_token / model.experts_per_layer
    full_expert_traffic = (
        model.expert_traffic_gb_per_unit
        * model.experts_per_layer
        / model.active_experts_per_token
    )
    active_traffic = model.dense_traffic_gb_per_unit + model.expert_traffic_gb_per_unit
    full_traffic = model.dense_traffic_gb_per_unit + full_expert_traffic
    return {
        "classification": "analytical_proxy_not_measured_energy_or_flops",
        "active_parameter_ratio": active_parameter_ratio,
        "parameter_work_proxy_avoided": 1.0 - active_parameter_ratio,
        "expert_active_ratio": expert_active_ratio,
        "expert_work_proxy_avoided": 1.0 - expert_active_ratio,
        "active_weight_traffic_gb_per_unit": active_traffic,
        "all_expert_weight_traffic_gb_per_unit": full_traffic,
        "weight_traffic_proxy_reduction": 1.0 - active_traffic / full_traffic,
    }
