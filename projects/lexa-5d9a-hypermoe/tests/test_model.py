from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from lexa_hypermoe.model import (
    HardwareProfile,
    ModelSpec,
    Placement,
    all_active_experts_hit_approx,
    estimate_throughput,
    layer_budget_ms,
    max_layer_miss_probability,
    optimize_placement,
    prefetch_horizon_layers,
    zipf_probability_mass,
)
from lexa_hypermoe.trace import aggregate_cache_stats, iter_jsonl, layer_cache_stats


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ModelSpec()
        self.hardware = HardwareProfile(
            name="test",
            vram_gib=16,
            ram_gib=32,
            gpu_peak_bandwidth_gbps=716.8,
            gpu_efficiency=0.55,
            ram_effective_bandwidth_gbps=50,
            nvme_effective_bandwidth_gbps=5.5,
            pcie_effective_bandwidth_gbps=24,
            reserved_vram_gib=8,
            fixed_overhead_ms=1.5,
        )

    def test_expert_geometry(self) -> None:
        self.assertEqual(self.spec.parameters_per_expert, 24_883_200)
        self.assertAlmostEqual(self.spec.expert_size_mib, 12.6068115234375)
        self.assertEqual(
            self.spec.active_expert_parameters_per_token,
            36 * 4 * 24_883_200,
        )
        self.assertGreater(self.spec.estimated_dense_bytes_per_token, 0)
        self.assertGreater(
            self.spec.estimated_total_active_bytes_per_token,
            self.spec.active_expert_bytes_per_token,
        )

    def test_kv_cache_estimate(self) -> None:
        q8 = self.spec.kv_cache_bytes(8192, 1.0)
        self.assertGreater(q8, 0)
        self.assertLess(q8, 512 * 1024 * 1024)
        self.assertEqual(self.spec.kv_cache_bytes(8192, 2.0), q8 * 2)

    def test_placement_validation(self) -> None:
        Placement(0.6, 0.3, 0.1)
        with self.assertRaises(ValueError):
            Placement(0.6, 0.6, 0.0)

    def test_throughput_finite(self) -> None:
        result = estimate_throughput(
            self.spec,
            self.hardware,
            Placement(0.68, 0.29, 0.03),
        )
        self.assertTrue(math.isfinite(result.tokens_per_second))
        self.assertGreater(result.tokens_per_second, 0)
        self.assertGreaterEqual(result.total_time_ms, result.critical_path_ms)

    def test_optimizer(self) -> None:
        result = optimize_placement(self.spec, self.hardware, step=0.02)
        self.assertAlmostEqual(
            result.placement.gpu + result.placement.ram + result.placement.nvme,
            1.0,
        )
        self.assertGreater(result.tokens_per_second, 0)

    def test_zipf_monotonicity(self) -> None:
        eight = zipf_probability_mass(128, 8, 1.2)
        sixteen = zipf_probability_mass(128, 16, 1.2)
        steeper = zipf_probability_mass(128, 16, 1.6)
        self.assertLess(eight, sixteen)
        self.assertLess(sixteen, steeper)
        self.assertAlmostEqual(all_active_experts_hit_approx(0.5, 4), 0.0625)

    def test_prefetch_and_miss_math(self) -> None:
        budget = layer_budget_ms(90, 36)
        self.assertAlmostEqual(budget, 1000 / (90 * 36))
        self.assertEqual(prefetch_horizon_layers(2.21, budget), 8)
        p = max_layer_miss_probability(36, 0.99)
        self.assertGreater(p, 0)
        self.assertLess(p, 0.001)
        self.assertAlmostEqual((1 - p) ** 36, 0.99, places=10)

    def test_expert_slots(self) -> None:
        slots = self.hardware.expert_slots(self.spec)
        self.assertGreater(slots, 600)
        self.assertLess(slots, 700)


class TraceTests(unittest.TestCase):
    def test_trace_analysis(self) -> None:
        rows = [
            {"token": 0, "layer": 0, "experts": [1, 2, 3, 4]},
            {"token": 1, "layer": 0, "experts": [1, 2, 5, 6]},
            {"token": 0, "layer": 1, "experts": [7, 8, 9, 10]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            stats = layer_cache_stats(
                iter_jsonl(path),
                cached_experts_per_layer=2,
            )
        aggregate = aggregate_cache_stats(stats)
        self.assertEqual(len(stats), 2)
        self.assertGreater(aggregate["per_selection_hit_rate"], 0)
        self.assertLessEqual(aggregate["per_selection_hit_rate"], 1)

    def test_invalid_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"token": 0, "layer": 0}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                tuple(iter_jsonl(path))


if __name__ == "__main__":
    unittest.main()
