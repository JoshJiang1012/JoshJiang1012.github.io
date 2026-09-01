from __future__ import annotations

import unittest

from lexa_asymcompute.moe import (
    MoEHardwareSpec,
    MoEModelSpec,
    MoEPlacement,
    estimate_moe,
    moe_savings,
    optimize_moe,
)


MODEL = MoEModelSpec(
    name="reference-sparse-model",
    total_parameters_b=116.8292,
    active_parameters_b=5.1,
    layers=36,
    experts_per_layer=128,
    active_experts_per_token=4,
    dense_traffic_gb_per_unit=3.0336,
    expert_traffic_gb_per_unit=1.9036,
)
HARDWARE = MoEHardwareSpec(
    name="reference-hardware",
    gpu_effective_gbps=394.24,
    ram_effective_gbps=50.0,
    nvme_effective_gbps=5.5,
    fixed_overhead_ms=1.5,
    expert_cache_gib=8.0,
)


class MoETests(unittest.TestCase):
    def test_reference_savings(self) -> None:
        result = moe_savings(MODEL)
        self.assertAlmostEqual(result["parameter_work_proxy_avoided"], 0.956346, places=5)
        self.assertAlmostEqual(result["expert_work_proxy_avoided"], 0.96875)

    def test_reference_placement(self) -> None:
        result = estimate_moe(MODEL, HARDWARE, MoEPlacement(0.6825, 0.2875, 0.03))
        self.assertAlmostEqual(result.units_per_second, 80.06, delta=0.2)

    def test_optimizer_reproduces_near_reference(self) -> None:
        result = optimize_moe(MODEL, HARDWARE, 0.0025)
        self.assertGreater(result.units_per_second, 79.0)
        self.assertAlmostEqual(result.placement.gpu, 0.6825, delta=0.01)


if __name__ == "__main__":
    unittest.main()
