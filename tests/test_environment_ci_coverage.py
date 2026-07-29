from __future__ import annotations

import unittest
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_ENVIRONMENTS = _ROOT / "environments"
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"


class EnvironmentCICoverageTests(unittest.TestCase):
    def test_every_environment_distribution_is_named_by_ci(self) -> None:
        projects = {
            pyproject.parent.relative_to(_ROOT).as_posix()
            for pyproject in _ENVIRONMENTS.rglob("pyproject.toml")
            if ".venv" not in pyproject.parts
            and "vendor" not in pyproject.parts
        }
        workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

        missing = tuple(
            sorted(project for project in projects if project not in workflow)
        )

        self.assertEqual(missing, ())


if __name__ == "__main__":
    unittest.main()
