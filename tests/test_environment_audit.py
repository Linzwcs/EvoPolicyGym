from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_environments import (
    AuditCommand,
    EnvironmentProject,
    audit_project,
    commands_for,
    discover_projects,
    layout_issues,
    select_projects,
)

_ROOT = Path(__file__).parents[1]


class EnvironmentAuditTests(unittest.TestCase):
    def test_discovers_all_repository_environment_projects(self) -> None:
        expected = {
            pyproject.parent.resolve()
            for pyproject in (_ROOT / "environments").rglob("pyproject.toml")
            if ".venv" not in pyproject.parts
            and "vendor" not in pyproject.parts
        }

        discovered = discover_projects(_ROOT)

        self.assertEqual({project.path for project in discovered}, expected)
        self.assertGreater(len(discovered), 50)
        self.assertTrue(
            all(layout_issues(project) == () for project in discovered)
        )

    def test_discovery_prunes_virtual_and_vendored_projects(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            included = self._make_project(
                root,
                "environments/example/suite",
            )
            self._make_project(
                root,
                "environments/example/suite/vendor/upstream",
            )
            self._make_project(
                root,
                "environments/other/.venv/package",
            )

            discovered = discover_projects(root)

        self.assertEqual(
            tuple(project.relative_path for project in discovered),
            (included.relative_path,),
        )

    def test_smoke_and_full_plans_match_documented_contract(self) -> None:
        smoke = commands_for("smoke", skip_sync=False)
        full = commands_for("full", skip_sync=False)

        self.assertEqual(
            tuple(command.label for command in smoke),
            ("sync", "tests"),
        )
        self.assertEqual(
            tuple(command.label for command in full),
            ("sync", "ruff", "mypy", "tests", "build"),
        )
        self.assertEqual(full[2].arguments, ("uv", "run", "mypy", "--no-incremental"))
        self.assertEqual(
            tuple(
                command.label
                for command in commands_for("smoke", skip_sync=True)
            ),
            ("tests",),
        )

    def test_selects_projects_with_repository_relative_globs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            classic = self._make_project(
                root,
                "environments/gymnasium/classic/cartpole",
            )
            self._make_project(root, "environments/minigrid/empty")
            projects = discover_projects(root)

            selected = select_projects(
                projects,
                ("environments/gymnasium/*",),
            )

        self.assertEqual(selected, (classic,))

    def test_project_stops_after_first_failed_command(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = self._make_project(root, "environments/example")
            commands = (
                AuditCommand(
                    label="pass",
                    arguments=(sys.executable, "-c", "print('passed')"),
                ),
                AuditCommand(
                    label="fail",
                    arguments=(sys.executable, "-c", "raise SystemExit(3)"),
                ),
                AuditCommand(
                    label="not-run",
                    arguments=(sys.executable, "-c", "print('unexpected')"),
                ),
            )

            result = audit_project(
                project,
                commands,
                timeout_seconds=10,
            )

        self.assertFalse(result.passed)
        self.assertEqual(
            tuple(command.command.label for command in result.commands),
            ("pass", "fail"),
        )
        self.assertEqual(result.commands[-1].returncode, 3)

    def _make_project(
        self,
        root: Path,
        relative_path: str,
    ) -> EnvironmentProject:
        path = root / relative_path
        path.mkdir(parents=True)
        (path / "src").mkdir()
        (path / "tests").mkdir()
        (path / "pyproject.toml").write_text(
            "[project]\nname = 'example'\nversion = '0.1.0'\n",
            encoding="utf-8",
        )
        (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        return EnvironmentProject(repository_root=root.resolve(), path=path)


if __name__ == "__main__":
    unittest.main()
