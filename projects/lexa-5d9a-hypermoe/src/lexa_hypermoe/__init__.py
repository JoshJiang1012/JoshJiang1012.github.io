"""Lexa 5D9A HyperMoE analytical and router-trace toolkit."""

from .cache_sim import CachePolicyStats, simulate_ema, simulate_lru, simulate_warmup_static
from .model import (
    HardwareProfile,
    ModelSpec,
    Placement,
    ThroughputEstimate,
    all_active_experts_hit_approx,
    closed_form_unconstrained_placement,
    estimate_throughput,
    layer_budget_ms,
    max_layer_miss_probability,
    optimize_placement,
    prefetch_horizon_layers,
    zipf_probability_mass,
)
from .trace import RouterEvent, TraceAudit, cache_sweep, trace_audit

__version__ = "1.0.0"

__all__ = [
    "CachePolicyStats",
    "HardwareProfile",
    "ModelSpec",
    "Placement",
    "RouterEvent",
    "ThroughputEstimate",
    "TraceAudit",
    "all_active_experts_hit_approx",
    "cache_sweep",
    "closed_form_unconstrained_placement",
    "estimate_throughput",
    "layer_budget_ms",
    "max_layer_miss_probability",
    "optimize_placement",
    "prefetch_horizon_layers",
    "simulate_ema",
    "simulate_lru",
    "simulate_warmup_static",
    "trace_audit",
    "zipf_probability_mass",
]
