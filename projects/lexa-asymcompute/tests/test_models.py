from __future__ import annotations

import unittest

from lexa_asymcompute.models import (
    DeviceSpec,
    FractionalPlacement,
    HardwareSystem,
    LinkSpec,
    WorkloadSpec,
    estimate,
)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = HardwareSystem(
            name="test-system",
            devices=(
                DeviceSpec("gpu", 40.0, 500.0, 16.0, 0.5, 0.8, power_watts=250),
                DeviceSpec("cpu", 1.0, 50.0, 32.0, 0.7, 0.8, power_watts=80),
            ),
            links=(LinkSpec("cpu", "gpu", 24.0, latency_ms=0.02),),
        )
        self.workload = WorkloadSpec(
            name="mixed",
            parallel_flops=2e10,
            parallel_bytes=4e9,
            parallel_resident_bytes=8 * 1024**3,
            output_bytes=1e8,
            aggregate_device="gpu",
            eligible_devices=("gpu", "cpu"),
            sync_overhead_ms=0.2,
        )

    def test_estimate_returns_positive_result(self) -> None:
        result = estimate(self.workload, self.hardware, FractionalPlacement({"gpu": 0.75, "cpu": 0.25}))
        self.assertGreater(result.units_per_second, 0)
        self.assertGreater(result.total_ms, result.sync_overhead_ms)
        self.assertEqual(len(result.devices), 2)

    def test_capacity_violation_rejected(self) -> None:
        workload = WorkloadSpec(
            name="too-large",
            parallel_flops=1,
            parallel_bytes=1,
            parallel_resident_bytes=100 * 1024**3,
            eligible_devices=("gpu",),
        )
        with self.assertRaisesRegex(ValueError, "capacity"):
            estimate(workload, self.hardware, FractionalPlacement({"gpu": 1.0}))

    def test_missing_link_rejected(self) -> None:
        no_links = HardwareSystem("no-links", self.hardware.devices)
        with self.assertRaisesRegex(ValueError, "missing link"):
            estimate(self.workload, no_links, FractionalPlacement({"gpu": 0.5, "cpu": 0.5}))


if __name__ == "__main__":
    unittest.main()
