"""Lexa AsymCompute: reproducible asymmetric-compute modelling."""

from .models import (
    DeviceSpec,
    Estimate,
    FractionalPlacement,
    HardwareSystem,
    LinkSpec,
    WorkloadSpec,
    estimate,
)
from .moe import MoEHardwareSpec, MoEModelSpec, MoEPlacement, estimate_moe, moe_savings

__all__ = [
    "DeviceSpec",
    "Estimate",
    "FractionalPlacement",
    "HardwareSystem",
    "LinkSpec",
    "WorkloadSpec",
    "estimate",
    "MoEHardwareSpec",
    "MoEModelSpec",
    "MoEPlacement",
    "estimate_moe",
    "moe_savings",
]

__version__ = "2.0.0"
