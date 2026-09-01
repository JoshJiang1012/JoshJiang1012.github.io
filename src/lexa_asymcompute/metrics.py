from __future__ import annotations

from collections.abc import Iterable
import math


def jain_fairness(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data or any(value < 0 for value in data):
        raise ValueError("values must be a non-empty non-negative sequence")
    denominator = len(data) * sum(value * value for value in data)
    if denominator == 0:
        return 1.0
    return sum(data) ** 2 / denominator


def asymmetry_index(values: Iterable[float]) -> float:
    """Normalized disparity: 0 means identical tiers, values approach 1 as skew grows."""

    return 1.0 - jain_fairness(values)


def speedup(baseline_ms: float, optimized_ms: float) -> float:
    if baseline_ms <= 0 or optimized_ms <= 0:
        raise ValueError("durations must be positive")
    return baseline_ms / optimized_ms


def latency_reduction(baseline_ms: float, optimized_ms: float) -> float:
    return 1.0 - 1.0 / speedup(baseline_ms, optimized_ms)


def geometric_mean(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data or any(value <= 0 for value in data):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in data) / len(data))
