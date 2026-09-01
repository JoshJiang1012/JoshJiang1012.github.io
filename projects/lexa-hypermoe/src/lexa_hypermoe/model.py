from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_parameters_b: float
    active_parameters_b: float
    layers: int
    experts_per_layer: int
    active_experts_per_token: int
    dense_traffic_gb_per_token: float
    expert_traffic_gb_per_token: float

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    gpu_effective_gbps: float
    ram_effective_gbps: float
    nvme_effective_gbps: float
    fixed_overhead_ms: float
    expert_cache_gib: float

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Placement:
    gpu: float
    ram: float
    nvme: float

    def validate(self) -> None:
        values = (self.gpu, self.ram, self.nvme)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("placement fractions must be within [0, 1]")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("placement fractions must sum to 1")


@dataclass(frozen=True)
class Estimate:
    placement: Placement
    gpu_ms: float
    ram_ms: float
    nvme_ms: float
    critical_path_ms: float
    fixed_overhead_ms: float
    total_ms: float
    tokens_per_second: float
    classification: str = "analytical_estimate_not_observed_benchmark"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["placement"] = asdict(self.placement)
        return result


def estimate(model: ModelSpec, hardware: HardwareSpec, placement: Placement) -> Estimate:
    placement.validate()
    if min(
        hardware.gpu_effective_gbps,
        hardware.ram_effective_gbps,
        hardware.nvme_effective_gbps,
    ) <= 0:
        raise ValueError("effective bandwidth values must be positive")

    gpu_ms = 1000.0 * (
        model.dense_traffic_gb_per_token
        + placement.gpu * model.expert_traffic_gb_per_token
    ) / hardware.gpu_effective_gbps
    ram_ms = 1000.0 * (
        placement.ram * model.expert_traffic_gb_per_token
    ) / hardware.ram_effective_gbps
    nvme_ms = 1000.0 * (
        placement.nvme * model.expert_traffic_gb_per_token
    ) / hardware.nvme_effective_gbps
    critical = max(gpu_ms, ram_ms, nvme_ms)
    total = critical + hardware.fixed_overhead_ms
    return Estimate(
        placement=placement,
        gpu_ms=gpu_ms,
        ram_ms=ram_ms,
        nvme_ms=nvme_ms,
        critical_path_ms=critical,
        fixed_overhead_ms=hardware.fixed_overhead_ms,
        total_ms=total,
        tokens_per_second=1000.0 / total,
    )


def savings(model: ModelSpec) -> dict[str, float | str]:
    active_parameter_ratio = model.active_parameters_b / model.total_parameters_b
    expert_active_ratio = model.active_experts_per_token / model.experts_per_layer
    full_expert_traffic = (
        model.expert_traffic_gb_per_token
        * model.experts_per_layer
        / model.active_experts_per_token
    )
    active_traffic = model.dense_traffic_gb_per_token + model.expert_traffic_gb_per_token
    full_traffic = model.dense_traffic_gb_per_token + full_expert_traffic
    return {
        "classification": "analytical_proxy_not_measured_energy_or_flops",
        "active_parameter_ratio": active_parameter_ratio,
        "parameter_work_proxy_avoided": 1.0 - active_parameter_ratio,
        "expert_active_ratio": expert_active_ratio,
        "expert_work_proxy_avoided": 1.0 - expert_active_ratio,
        "active_weight_traffic_gb_per_token": active_traffic,
        "full_expert_weight_traffic_gb_per_token": full_traffic,
        "weight_traffic_proxy_reduction": 1.0 - active_traffic / full_traffic,
    }
