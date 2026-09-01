from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lexa_asymcompute.trace import load_trace, temporal_top_n_holdout, validate_geometry


class TraceTests(unittest.TestCase):
    def _trace(self, forbidden: bool = False) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        with handle:
            for token in range(10):
                for layer in range(2):
                    record = {"token": token, "layer": layer, "experts": [layer, layer + 2], "phase": "decode"}
                    if forbidden and token == 0 and layer == 0:
                        record["prompt"] = "secret"
                    handle.write(json.dumps(record) + "\n")
        return Path(handle.name)

    def test_privacy_fail_closed(self) -> None:
        path = self._trace(forbidden=True)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            load_trace(path)

    def test_geometry_and_holdout(self) -> None:
        events = load_trace(self._trace())
        geometry = validate_geometry(events, layers=2, top_k=2, experts_per_layer=8)
        self.assertTrue(geometry["valid"])
        result = temporal_top_n_holdout(events, cache_per_layer=2, calibration_fraction=0.7)
        self.assertEqual(result["selection_hit_rate"], 1.0)
        self.assertEqual(result["full_row_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
