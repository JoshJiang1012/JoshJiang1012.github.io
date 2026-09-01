from pathlib import Path
import unittest

from lexa_hypermoe.model import HardwareSpec, ModelSpec, Placement, estimate, savings
from lexa_hypermoe.optimizer import optimize

ROOT = Path(__file__).resolve().parents[1]
MODEL = ModelSpec.from_json(ROOT / "data/model/gpt_oss_120b.json")
HARDWARE = HardwareSpec.from_json(ROOT / "data/hardware/rtx4080_i7_13700_32gb_assumed.json")


class ModelTests(unittest.TestCase):
    def test_cpu_moe_baseline(self) -> None:
        result = estimate(MODEL, HARDWARE, Placement(0.0, 1.0, 0.0))
        self.assertAlmostEqual(result.tokens_per_second, 25.27, places=2)

    def test_optimizer_reproduces_reference_point(self) -> None:
        result = optimize(MODEL, HARDWARE)
        self.assertAlmostEqual(result.placement.gpu, 0.6825, places=4)
        self.assertAlmostEqual(result.placement.ram, 0.2875, places=4)
        self.assertAlmostEqual(result.placement.nvme, 0.0300, places=4)
        self.assertAlmostEqual(result.tokens_per_second, 80.06, places=2)

    def test_savings_are_explicitly_analytical(self) -> None:
        result = savings(MODEL)
        self.assertEqual(result["classification"], "analytical_proxy_not_measured_energy_or_flops")
        self.assertAlmostEqual(result["parameter_work_proxy_avoided"], 0.95635, places=4)
        self.assertAlmostEqual(result["expert_work_proxy_avoided"], 0.96875, places=5)

    def test_invalid_placement_fails(self) -> None:
        with self.assertRaises(ValueError):
            estimate(MODEL, HARDWARE, Placement(0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
