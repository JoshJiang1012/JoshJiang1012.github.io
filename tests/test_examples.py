from __future__ import annotations

import json
from pathlib import Path
import unittest

from lexa_asymcompute.models import HardwareSystem, WorkloadSpec
from lexa_asymcompute.moe import MoEHardwareSpec, MoEModelSpec

ROOT = Path(__file__).resolve().parents[1]


class ExampleDataTests(unittest.TestCase):
    def test_generic_profiles_load(self) -> None:
        HardwareSystem.from_json(ROOT / "data/hardware/desktop_heterogeneous_example.json").validate()
        WorkloadSpec.from_json(ROOT / "data/workloads/edge_pipeline_example.json").validate()

    def test_sparse_profiles_load(self) -> None:
        model = MoEModelSpec.from_json(ROOT / "data/workloads/sparse_moe_reference.json")
        hardware = MoEHardwareSpec.from_json(ROOT / "data/hardware/moe_three_tier_assumed.json")
        self.assertGreater(model.total_parameters_b, model.active_parameters_b)
        self.assertGreater(hardware.gpu_effective_gbps, hardware.ram_effective_gbps)

    def test_observed_files_are_classified(self) -> None:
        for path in (ROOT / "data/observed").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["classification"].startswith("observed_"))
            self.assertTrue(payload["measured"])


if __name__ == "__main__":
    unittest.main()
