"""Analytical performance model for heterogeneous GPT-OSS MoE inference.

The equations in this module are deterministic engineering estimates. They do
not claim measured throughput unless callers provide measured bandwidths and
latencies. Units are explicit: storage sizes use bytes, bandwidth uses decimal
GB/s, and time uses seconds or milliseconds as named.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

_DECIMAL_GB = 1_000_000_000.0
_GIB = 1024.0**3
_MIB = 1024.0**2


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Minimal model constants needed by the analytical simulator."""

    name: str = "openai/gpt-oss-120b"
    total_parameters: float = 116_829_200_000.0
    active_parameters_per_token: float = 5_100_000_000.0
    layers: int = 36
    experts_per_layer: int = 128
    active_experts_per_token: int = 4
    hidden_size: int = 2880
    intermediate_size: int = 2880
    vocab_size: int = 201_088
    query_heads: int = 64
    key_value_heads: int = 8
    head_dim: int = 64
    sliding_window: int = 128
    full_attention_layers: int = 18
    sliding_attention_layers: int = 18
    dense_bytes_per_parameter: float = 2.0
    # GPT-OSS native MXFP4: 32 packed FP4 values in 16 bytes plus one
    # E8M0 scale byte per block => 17 / 32 bytes per scalar weight.
    expert_bytes_per_parameter: float = 17.0 / 32.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.layers,
            self.experts_per_layer,
            self.active_experts_per_token,
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("model dimensions must be positive")
        if self.active_experts_per_token > self.experts_per_layer:
            raise ValueError("active experts cannot exceed experts per layer")
        if self.active_parameters_per_token <= 0 or self.total_parameters <= 0:
            raise ValueError("parameter counts must be positive")
        if self.expert_bytes_per_parameter <= 0 or self.dense_bytes_per_parameter <= 0:
            raise ValueError("storage widths must be positive")

    @property
    def parameters_per_expert(self) -> int:
        # Gate, up, and down projections. Biases are intentionally omitted; they
        # are tiny relative to matrices and are included in the model uncertainty.
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def expert_size_bytes(self) -> float:
        return self.parameters_per_expert * self.expert_bytes_per_parameter

    @property
    def expert_size_mib(self) -> float:
        return self.expert_size_bytes / _MIB

    @property
    def active_expert_parameters_per_token(self) -> int:
        return self.layers * self.active_experts_per_token * self.parameters_per_expert

    @property
    def active_expert_bytes_per_token(self) -> float:
        return self.layers * self.active_experts_per_token * self.expert_size_bytes

    @property
    def estimated_dense_active_parameters_per_token(self) -> float:
        return max(
            0.0,
            self.active_parameters_per_token - self.active_expert_parameters_per_token,
        )

    @property
    def estimated_dense_bytes_per_token(self) -> float:
        return self.estimated_dense_active_parameters_per_token * self.dense_bytes_per_parameter

    @property
    def estimated_total_active_bytes_per_token(self) -> float:
        return self.estimated_dense_bytes_per_token + self.active_expert_bytes_per_token

    def kv_cache_bytes(self, context_tokens: int, bytes_per_element: float = 1.0) -> float:
        """Estimate KV storage using alternating full/sliding attention.

        This excludes allocator metadata and alignment. Callers should reserve
        additional headroom in real deployments.
        """
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")
        positions = (
            self.full_attention_layers * context_tokens
            + self.sliding_attention_layers * min(context_tokens, self.sliding_window)
        )
        return bytes_per_element * 2 * self.key_value_heads * self.head_dim * positions

    def to_dict(self) -> dict[str, float | int | str]:
        payload = asdict(self)
        payload.update(
            {
                "parameters_per_expert": self.parameters_per_expert,
                "expert_size_bytes": self.expert_size_bytes,
                "expert_size_mib": self.expert_size_mib,
                "active_expert_parameters_per_token": self.active_expert_parameters_per_token,
                "active_expert_bytes_per_token": self.active_expert_bytes_per_token,
                "estimated_dense_active_parameters_per_token": (
                    self.estimated_dense_active_parameters_per_token
                ),
                "estimated_dense_bytes_per_token": self.estimated_dense_bytes_per_token,
                "estimated_total_active_bytes_per_token": (
                    self.estimated_total_active_bytes_per_token
                ),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Effective hardware rates consumed by the model.

    The bandwidth fields are *effective* rates after kernel/dequantization
    efficiency, not necessarily vendor peak specifications.
    """

    name: str
    vram_gib: float
    ram_gib: float
    gpu_peak_bandwidth_gbps: float
    gpu_efficiency: float
    ram_effective_bandwidth_gbps: float
    nvme_effective_bandwidth_gbps: float
    pcie_effective_bandwidth_gbps: float
    reserved_vram_gib: float
    fixed_overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.vram_gib <= 0 or self.ram_gib <= 0:
            raise ValueError("memory capacities must be positive")
        if self.reserved_vram_gib < 0 or self.reserved_vram_gib >= self.vram_gib:
            raise ValueError("reserved VRAM must be within total VRAM")
        for value in (
            self.gpu_peak_bandwidth_gbps,
            self.ram_effective_bandwidth_gbps,
            self.nvme_effective_bandwidth_gbps,
            self.pcie_effective_bandwidth_gbps,
        ):
            if value <= 0:
                raise ValueError("bandwidths must be positive")
        if not 0 < self.gpu_efficiency <= 1:
            raise ValueError("gpu_efficiency must be in (0, 1]")
        if self.fixed_overhead_ms < 0:
            raise ValueError("fixed_overhead_ms cannot be negative")

    @property
    def gpu_effective_bandwidth_gbps(self) -> float:
        return self.gpu_peak_bandwidth_gbps * self.gpu_efficiency

    @property
    def expert_cache_bytes(self) -> float:
        return (self.vram_gib - self.reserved_vram_gib) * _GIB

    def expert_slots(self, spec: ModelSpec) -> int:
        return max(0, int(self.expert_cache_bytes // spec.expert_size_bytes))

    def to_dict(self, spec: ModelSpec | None = None) -> dict[str, float | int | str]:
        payload = asdict(self)
        payload["gpu_effective_bandwidth_gbps"] = self.gpu_effective_bandwidth_gbps
        if spec is not None:
            payload["expert_cache_bytes"] = self.expert_cache_bytes
            payload["expert_slots"] = self.expert_slots(spec)
            payload["expert_slots_per_layer_mean"] = self.expert_slots(spec) / spec.layers
        return payload


@dataclass(frozen=True, slots=True)
class Placement:
    """Fraction of active expert traffic assigned to each tier."""

    gpu: float
    ram: float
    nvme: float

    def __post_init__(self) -> None:
        values = (self.gpu, self.ram, self.nvme)
        if any(value < -1e-12 or value > 1.0 + 1e-12 for value in values):
            raise ValueError("placement fractions must be between 0 and 1")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError("placement fractions must sum to 1")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThroughputEstimate:
    placement: Placement
    gpu_time_ms: float
    ram_time_ms: float
    nvme_time_ms: float
    critical_path_ms: float
    fixed_overhead_ms: float
    expected_miss_penalty_ms: float
    total_time_ms: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "placement": self.placement.to_dict(),
            "gpu_time_ms": self.gpu_time_ms,
            "ram_time_ms": self.ram_time_ms,
            "nvme_time_ms": self.nvme_time_ms,
            "critical_path_ms": self.critical_path_ms,
            "fixed_overhead_ms": self.fixed_overhead_ms,
            "expected_miss_penalty_ms": self.expected_miss_penalty_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
        }


def estimate_throughput(
    spec: ModelSpec,
    hardware: HardwareProfile,
    placement: Placement,
    *,
    critical_miss_probability_per_layer: float = 0.0,
    critical_miss_penalty_ms: float = 0.0,
) -> ThroughputEstimate:
    """Estimate decode throughput for one-token, batch-one inference.

    The three expert paths are assumed to overlap. Dense traffic always uses the
    GPU path. NVMe bandwidth is treated as an already-calibrated effective path
    that includes staging and destination work; do not pass raw SSD peak speed
    unless intentionally constructing an optimistic upper bound.
    """
    if not 0 <= critical_miss_probability_per_layer <= 1:
        raise ValueError("critical miss probability must be in [0, 1]")
    if critical_miss_penalty_ms < 0:
        raise ValueError("critical miss penalty cannot be negative")

    expert_bytes = spec.active_expert_bytes_per_token
    gpu_bytes = spec.estimated_dense_bytes_per_token + placement.gpu * expert_bytes
    ram_bytes = placement.ram * expert_bytes
    nvme_bytes = placement.nvme * expert_bytes

    gpu_seconds = gpu_bytes / (hardware.gpu_effective_bandwidth_gbps * _DECIMAL_GB)
    ram_seconds = ram_bytes / (hardware.ram_effective_bandwidth_gbps * _DECIMAL_GB)
    nvme_seconds = nvme_bytes / (hardware.nvme_effective_bandwidth_gbps * _DECIMAL_GB)
    critical_seconds = max(gpu_seconds, ram_seconds, nvme_seconds)

    expected_miss_penalty_ms = (
        spec.layers * critical_miss_probability_per_layer * critical_miss_penalty_ms
    )
    total_ms = (
        critical_seconds * 1000.0
        + hardware.fixed_overhead_ms
        + expected_miss_penalty_ms
    )
    tokens_per_second = math.inf if total_ms == 0 else 1000.0 / total_ms
    return ThroughputEstimate(
        placement=placement,
        gpu_time_ms=gpu_seconds * 1000.0,
        ram_time_ms=ram_seconds * 1000.0,
        nvme_time_ms=nvme_seconds * 1000.0,
        critical_path_ms=critical_seconds * 1000.0,
        fixed_overhead_ms=hardware.fixed_overhead_ms,
        expected_miss_penalty_ms=expected_miss_penalty_ms,
        total_time_ms=total_ms,
        tokens_per_second=tokens_per_second,
    )


def optimize_placement(
    spec: ModelSpec,
    hardware: HardwareProfile,
    *,
    step: float = 0.0025,
    max_gpu_fraction: float = 1.0,
    critical_miss_probability_per_layer: float = 0.0,
    critical_miss_penalty_ms: float = 0.0,
) -> ThroughputEstimate:
    """Grid-search a robust placement under a configurable GPU fraction cap."""
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    if not 0 <= max_gpu_fraction <= 1:
        raise ValueError("max_gpu_fraction must be in [0, 1]")
    count = int(round(1.0 / step))
    best: ThroughputEstimate | None = None
    for gpu_i in range(count + 1):
        gpu = gpu_i / count
        if gpu > max_gpu_fraction + 1e-12:
            break
        for ram_i in range(count - gpu_i + 1):
            ram = ram_i / count
            nvme = 1.0 - gpu - ram
            estimate = estimate_throughput(
                spec,
                hardware,
                Placement(gpu=gpu, ram=ram, nvme=nvme),
                critical_miss_probability_per_layer=critical_miss_probability_per_layer,
                critical_miss_penalty_ms=critical_miss_penalty_ms,
            )
            if best is None or estimate.tokens_per_second > best.tokens_per_second:
                best = estimate
    assert best is not None
    return best


def closed_form_unconstrained_placement(
    spec: ModelSpec,
    hardware: HardwareProfile,
) -> Placement:
    """Solve the idealized three-path balance, then project onto the simplex.

    This ignores cache-capacity, synchronization, and miss constraints. It is an
    optimistic reference point rather than a deployment prescription.
    """
    dense = spec.estimated_dense_bytes_per_token
    expert = spec.active_expert_bytes_per_token
    g = hardware.gpu_effective_bandwidth_gbps * _DECIMAL_GB
    r = hardware.ram_effective_bandwidth_gbps * _DECIMAL_GB
    n = hardware.nvme_effective_bandwidth_gbps * _DECIMAL_GB
    t = (dense + expert) / (g + r + n)
    raw = [
        (g * t - dense) / expert,
        r * t / expert,
        n * t / expert,
    ]
    clipped = [max(0.0, value) for value in raw]
    total = sum(clipped)
    if total == 0:
        return Placement(1.0, 0.0, 0.0)
    normalized = [value / total for value in clipped]
    return Placement(*normalized)


def zipf_probability_mass(experts: int, cached: int, exponent: float) -> float:
    """Cumulative probability mass of the hottest ``cached`` Zipf experts."""
    if experts <= 0:
        raise ValueError("experts must be positive")
    if not 0 <= cached <= experts:
        raise ValueError("cached must be between zero and experts")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    denominator = sum(rank ** (-exponent) for rank in range(1, experts + 1))
    numerator = sum(rank ** (-exponent) for rank in range(1, cached + 1))
    return numerator / denominator if denominator else 0.0


def all_active_experts_hit_approx(per_expert_hit: float, top_k: int) -> float:
    """Independence approximation for all top-k routes hitting the cache."""
    if not 0 <= per_expert_hit <= 1:
        raise ValueError("hit probability must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return per_expert_hit**top_k


def layer_budget_ms(target_tokens_per_second: float, layers: int) -> float:
    if target_tokens_per_second <= 0 or layers <= 0:
        raise ValueError("target throughput and layers must be positive")
    return 1000.0 / (target_tokens_per_second * layers)


def prefetch_horizon_layers(load_latency_ms: float, layer_budget: float) -> int:
    if load_latency_ms < 0 or layer_budget <= 0:
        raise ValueError("latency must be nonnegative and layer budget positive")
    return int(math.ceil(load_latency_ms / layer_budget))


def max_layer_miss_probability(
    layers: int,
    target_probability_no_critical_miss: float,
) -> float:
    """Exact independent per-layer miss bound for a clean-token probability."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    if not 0 < target_probability_no_critical_miss <= 1:
        raise ValueError("target probability must be in (0, 1]")
    return 1.0 - target_probability_no_critical_miss ** (1.0 / layers)


def summarize_estimates(estimates: Iterable[ThroughputEstimate]) -> dict[str, float]:
    values = sorted(item.tokens_per_second for item in estimates)
    if not values:
        raise ValueError("at least one estimate is required")

    def percentile(p: float) -> float:
        index = (len(values) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return values[lower]
        fraction = index - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction

    return {
        "minimum": values[0],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "maximum": values[-1],
        "count": float(len(values)),
    }
