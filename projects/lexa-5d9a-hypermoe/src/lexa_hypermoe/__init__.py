"""Lexa 5D9A HyperMoE analytical toolkit."""

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

__version__ = "0.1.0"

__all__ = [
    "HardwareProfile",
    "ModelSpec",
    "Placement",
    "ThroughputEstimate",
    "all_active_experts_hit_approx",
    "closed_form_unconstrained_placement",
    "estimate_throughput",
    "layer_budget_ms",
    "max_layer_miss_probability",
    "optimize_placement",
    "prefetch_horizon_layers",
    "zipf_probability_mass",
]
