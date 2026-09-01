from __future__ import annotations

import io
from contextlib import redirect_stdout
import unittest

from lexa_asymcompute.cli import main


class CliTests(unittest.TestCase):
    def test_formula_summary(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["formula-summary"])
        self.assertEqual(code, 0)
        self.assertIn("critical_path", output.getvalue())


if __name__ == "__main__":
    unittest.main()
