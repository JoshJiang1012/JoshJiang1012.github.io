"""Lexa 5D9A HyperMoE analytical and router-trace toolkit."""

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

__version__ = "0.2.0"

__all__ = [
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
    "trace_audit",
    "zipf_probability_mass",
]
