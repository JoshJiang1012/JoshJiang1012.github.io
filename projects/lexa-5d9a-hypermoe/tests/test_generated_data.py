from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GeneratedDataTests(unittest.TestCase):
    def test_model_metadata(self) -> None:
        payload = json.loads((ROOT / "data/model/gpt_oss_120b.json").read_text())
        self.assertEqual(payload["layers"], 36)
        self.assertEqual(payload["experts_per_layer"], 128)
        self.assertEqual(payload["active_experts_per_token"], 4)
        self.assertEqual(payload["source_class"], "official_config_plus_derived_estimates")

    def test_synthetic_csvs_have_classification(self) -> None:
        for path in sorted((ROOT / "data/synthetic").glob("*.csv")):
            with self.subTest(path=path.name):
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertTrue(all(row.get("data_class") for row in rows))


if __name__ == "__main__":
    unittest.main()
