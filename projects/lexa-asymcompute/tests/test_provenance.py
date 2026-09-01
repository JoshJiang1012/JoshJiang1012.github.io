from __future__ import annotations

import unittest

from lexa_asymcompute.provenance import validate_evidence_payload


class ProvenanceTests(unittest.TestCase):
    def test_observed_record(self) -> None:
        record = validate_evidence_payload({
            "classification": "observed_measurement",
            "environment": "vm",
            "measured": True,
        })
        self.assertTrue(record.measured)

    def test_analytical_cannot_claim_measured(self) -> None:
        with self.assertRaises(ValueError):
            validate_evidence_payload({
                "classification": "analytical_estimate",
                "environment": "formula",
                "measured": True,
            })


if __name__ == "__main__":
    unittest.main()
