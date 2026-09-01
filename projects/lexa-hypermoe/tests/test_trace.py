from pathlib import Path
import json
import tempfile
import unittest

from lexa_hypermoe.trace import holdout_topn, read_trace, validate_trace


def write_trace(path: Path, *, tokens: int = 10, layers: int = 36) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for token in range(tokens):
            for layer in range(layers):
                base = (token * 7 + layer * 11) % 128
                handle.write(json.dumps({
                    "schema_version": "1.0",
                    "token": token,
                    "layer": layer,
                    "experts": [base, (base + 17) % 128, (base + 41) % 128, (base + 73) % 128],
                    "domain": "synthetic",
                    "phase": "decode",
                    "source": "unit-test",
                }) + "\n")


class TraceTests(unittest.TestCase):
    def test_trace_geometry_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path)
            records = validate_trace(
                read_trace(path),
                layers=36,
                experts_per_layer=128,
                active_experts=4,
            )
            result = holdout_topn(records, cached_experts_per_layer=18)
            self.assertEqual(result["calibration_tokens"], 7)
            self.assertEqual(result["holdout_tokens"], 3)
            self.assertGreaterEqual(result["selection_hit_rate"], 0.0)
            self.assertLessEqual(result["selection_hit_rate"], 1.0)

    def test_unknown_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps({
                "token": 0, "layer": 0, "experts": [0, 1, 2, 3],
                "prompt": "must not be stored",
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_trace(path)

    def test_incomplete_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path)
            records = read_trace(path)[:-1]
            with self.assertRaises(ValueError):
                validate_trace(records, layers=36, experts_per_layer=128, active_experts=4)


if __name__ == "__main__":
    unittest.main()
