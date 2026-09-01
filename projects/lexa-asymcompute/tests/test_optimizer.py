from __future__ import annotations

import unittest

from lexa_asymcompute.models import DeviceSpec, HardwareSystem, WorkloadSpec
from lexa_asymcompute.optimizer import optimize, placements


class OptimizerTests(unittest.TestCase):
    def test_placements_cover_simplex(self) -> None:
        values = list(placements(("a", "b", "c"), 0.5))
        self.assertEqual(len(values), 6)
        self.assertTrue(all(abs(sum(item.shares.values()) - 1.0) < 1e-9 for item in values))

    def test_optimizer_favors_fast_device(self) -> None:
        hardware = HardwareSystem(
            "two-tier",
            (
                DeviceSpec("fast", 10, 100, 16),
                DeviceSpec("slow", 1, 10, 64),
            ),
        )
        workload = WorkloadSpec(
            name="parallel",
            parallel_flops=1e12,
            parallel_bytes=1e10,
            eligible_devices=("fast", "slow"),
        )
        result = optimize(workload, hardware, step=0.1)
        self.assertGreater(result.placement.shares["fast"], result.placement.shares["slow"])


if __name__ == "__main__":
    unittest.main()
