from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from evopolicygym.metric import measure


class MetricTest(unittest.TestCase):
    def test_measure_static_code_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.py").write_text(
                textwrap.dedent(
                    """
                    import math

                    class Policy:
                        def act(self, obs):
                            if obs:
                                return 1
                            return 0
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (root / "helpers").mkdir()
            (root / "helpers" / "logic.py").write_text(
                "def choose(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text("notes\n", encoding="utf-8")
            (root / "metrics.json").write_text("{}\n", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "ignored.py").write_text("BAD\n", encoding="utf-8")

            data = measure(root)

            self.assertEqual(data["schema_version"], "0.1")
            self.assertEqual(data["files"], 3)
            self.assertEqual(data["python_files"], 2)
            self.assertEqual(data["policy_bytes"], (root / "policy.py").stat().st_size)
            self.assertEqual(data["classes"], 1)
            self.assertEqual(data["functions"], 2)
            self.assertEqual(data["imports"], ["math"])
            self.assertEqual(data["cyclomatic_total"], 3)
            self.assertEqual(data["cyclomatic_max"], 2)
            self.assertEqual(data["parse_errors"], [])
            self.assertTrue(str(data["tree_hash"]).startswith("sha256:"))

    def test_measure_records_parse_errors_by_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("def broken(:\n", encoding="utf-8")

            data = measure(root)

            self.assertEqual(data["python_files"], 1)
            self.assertEqual(data["parse_errors"], ["bad.py"])
            self.assertEqual(data["cyclomatic_total"], 0)


if __name__ == "__main__":
    unittest.main()
