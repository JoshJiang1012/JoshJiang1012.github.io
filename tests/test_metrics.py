from __future__ import annotations

import unittest

from lexa_asymcompute.metrics import asymmetry_index, jain_fairness, latency_reduction, speedup


class MetricTests(unittest.TestCase):
    def test_balanced_is_zero_asymmetry(self) -> None:
        self.assertAlmostEqual(jain_fairness([4, 4, 4]), 1.0)
        self.assertAlmostEqual(asymmetry_index([4, 4, 4]), 0.0)

    def test_skew_has_positive_asymmetry(self) -> None:
        self.assertGreater(asymmetry_index([100, 10, 1]), 0.0)

    def test_speedup(self) -> None:
        self.assertAlmostEqual(speedup(20, 10), 2.0)
        self.assertAlmostEqual(latency_reduction(20, 10), 0.5)


if __name__ == "__main__":
    unittest.main()
