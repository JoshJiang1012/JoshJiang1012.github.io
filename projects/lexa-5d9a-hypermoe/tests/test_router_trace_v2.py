from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lexa_hypermoe.trace import (
    RouterEvent,
    cache_sweep,
    iter_jsonl,
    layer_temporal_stats,
    trace_audit,
)


ROOT = Path(__file__).resolve().parents[1]


class RouterTraceV2Tests(unittest.TestCase):
    def write_rows(self, rows: list[dict[str, object]], directory: str) -> Path:
        path = Path(directory) / "trace.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_v2_sample_is_privacy_safe(self) -> None:
        sample = ROOT / "data/synthetic/router_trace_example.jsonl"
        audit = trace_audit(sample)
        self.assertTrue(audit.privacy_safe)
        self.assertEqual(audit.events, 4)
        self.assertEqual(audit.layers, 2)
        self.assertEqual(audit.tokens, 2)
        self.assertEqual(audit.max_expert_id, 122)
        self.assertEqual(audit.schema_versions, ("2.0",))
        self.assertEqual(
            audit.sha256,
            hashlib.sha256(sample.read_bytes()).hexdigest(),
        )

    def test_forbidden_prompt_field_fails_closed(self) -> None:
        row = {
            "schema_version": "2.0",
            "token": 0,
            "layer": 0,
            "experts": [1, 2, 3, 4],
            "domain": "test",
            "phase": "decode",
            "batch_size": 1,
            "source": "test",
            "prompt": "must not be stored",
        }
        with self.assertRaisesRegex(ValueError, "privacy-forbidden"):
            RouterEvent.from_mapping(row)

    def test_unknown_field_fails_closed(self) -> None:
        row = {
            "token": 0,
            "layer": 0,
            "experts": [1, 2, 3, 4],
            "unexpected": 1,
        }
        with self.assertRaisesRegex(ValueError, "unknown trace fields"):
            RouterEvent.from_mapping(row)

    def test_phase_filter_and_cache_sweep(self) -> None:
        rows = [
            {"token": 0, "layer": 0, "experts": [1, 2, 3, 4], "phase": "prefill"},
            {"token": 1, "layer": 0, "experts": [1, 2, 5, 6], "phase": "decode"},
            {"token": 2, "layer": 0, "experts": [1, 2, 5, 7], "phase": "decode"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(rows, directory)
            decode = tuple(iter_jsonl(path, phase="decode"))
        self.assertEqual(len(decode), 2)
        sweep = cache_sweep(decode, cache_sizes=(1, 2, 4))
        self.assertEqual([item["cached_experts_per_layer"] for item in sweep], [1, 2, 4])
        hit_rates = [item["aggregate"]["per_selection_hit_rate"] for item in sweep]
        self.assertEqual(hit_rates, sorted(hit_rates))

    def test_temporal_overlap(self) -> None:
        events = (
            RouterEvent(0, 0, (1, 2, 3, 4), phase="decode"),
            RouterEvent(1, 0, (1, 2, 3, 5), phase="decode"),
            RouterEvent(2, 0, (1, 2, 3, 5), phase="decode"),
        )
        stats = layer_temporal_stats(events)
        self.assertEqual(len(stats), 1)
        self.assertAlmostEqual(stats[0].mean_overlap_fraction, 0.875)
        self.assertAlmostEqual(stats[0].exact_repeat_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
