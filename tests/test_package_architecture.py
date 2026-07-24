from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "evopolicygym"

PROTOCOL_FORBIDDEN_STDLIB = {
    "io",
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "tempfile",
    "threading",
}


def module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("evopolicygym", *parts))


def imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = module_name(path)
    package = (
        module.split(".")
        if path.name == "__init__.py"
        else module.split(".")[:-1]
    )
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    found.append(node.module)
                continue
            retained = len(package) - (node.level - 1)
            if retained < 0:
                raise AssertionError(f"invalid relative import in {path}")
            suffix = () if node.module is None else tuple(node.module.split("."))
            found.append(".".join((*package[:retained], *suffix)))
    return tuple(found)


class PackageArchitectureTests(unittest.TestCase):
    def test_policy_import_does_not_load_runtime_graph(self) -> None:
        environment = {
            **os.environ,
            "PYTHONPATH": str(PACKAGE.parent),
        }
        script = """\
import sys
import evopolicygym.policy

forbidden = (
    "evopolicygym.evaluation._service",
    "evopolicygym.run._service",
    "evopolicygym.execution.process",
    "evopolicygym._protocol",
)
assert not any(name in sys.modules for name in forbidden)
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_public_selections_do_not_load_private_runtime(self) -> None:
        environment = {
            **os.environ,
            "PYTHONPATH": str(PACKAGE.parent),
        }
        script = """\
import sys
from evopolicygym.agents.codex import Codex
from evopolicygym.evaluation import EvaluationConfig
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress, RunConfig, RunEvent, RunObserver

assert EvaluationConfig.__module__ == "evopolicygym.evaluation"
assert RunConfig.__module__ == "evopolicygym.run"
assert ConsoleProgress.__module__ == "evopolicygym.run.progress"
assert RunEvent.__module__ == "evopolicygym.run.progress"
assert RunObserver.__module__ == "evopolicygym.run.progress"
assert Codex.__module__ == "evopolicygym.agents.codex"
assert ProcessExecution.__module__ == "evopolicygym.execution"

forbidden = (
    "evopolicygym.evaluation._service",
    "evopolicygym.run._service",
    "evopolicygym.execution.process",
    "evopolicygym._protocol",
)
assert not any(name in sys.modules for name in forbidden)
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_protocol_codecs_have_no_io_owners(self) -> None:
        protocol = PACKAGE / "_protocol"
        for path in protocol.rglob("*.py"):
            with self.subTest(module=module_name(path)):
                imported = imports(path)
                roots = {name.partition(".")[0] for name in imported}
                self.assertTrue(roots.isdisjoint(PROTOCOL_FORBIDDEN_STDLIB))
                self.assertFalse(
                    any(
                        name.startswith(
                            (
                                "evopolicygym.evaluation",
                                "evopolicygym.run",
                                "evopolicygym.execution",
                                "evopolicygym.agents",
                            )
                        )
                        for name in imported
                    )
                )

    def test_evaluation_rules_do_not_select_execution(self) -> None:
        service = PACKAGE / "evaluation" / "_service.py"
        self.assertFalse(
            any(
                name.startswith(
                    (
                        "evopolicygym.execution",
                        "evopolicygym.agents",
                        "evopolicygym.run",
                    )
                )
                for name in imports(service)
            )
        )

    def test_submission_rules_do_not_select_execution_or_provider(self) -> None:
        session = PACKAGE / "run" / "_session.py"
        self.assertFalse(
            any(
                name.startswith(
                    (
                        "evopolicygym.execution",
                        "evopolicygym.agents",
                    )
                )
                for name in imports(session)
            )
        )

    def test_run_service_does_not_select_a_provider(self) -> None:
        service = PACKAGE / "run" / "_service.py"
        self.assertFalse(
            any(
                name.startswith("evopolicygym.agents.codex")
                for name in imports(service)
            )
        )

    def test_process_execution_has_no_provider_dependencies(self) -> None:
        for path in (PACKAGE / "execution" / "process").rglob("*.py"):
            with self.subTest(module=module_name(path)):
                self.assertFalse(
                    any(
                        name.startswith("evopolicygym.agents")
                        for name in imports(path)
                    )
                )

    def test_authoring_has_no_private_runtime_dependencies(self) -> None:
        for path in (PACKAGE / "authoring").rglob("*.py"):
            with self.subTest(module=module_name(path)):
                self.assertFalse(
                    any(
                        name.startswith(
                            (
                                "evopolicygym.evaluation._service",
                                "evopolicygym.run._",
                                "evopolicygym.execution.process",
                                "evopolicygym._protocol",
                            )
                        )
                        for name in imports(path)
                    )
                )

    def test_removed_shadow_and_role_namespaces_do_not_return(self) -> None:
        for removed in (
            "_evaluation",
            "_evolution",
            "_execution",
            "_local",
            "_engine",
            "_adapters",
            "_wire",
            "_wiring",
            "settings",
        ):
            with self.subTest(package=removed):
                self.assertEqual(
                    tuple((PACKAGE / removed).rglob("*.py")),
                    (),
                )
        self.assertFalse((PACKAGE / "_composition.py").exists())
        self.assertFalse((PACKAGE / "_composition").exists())

    def test_run_io_does_not_live_under_process_execution(self) -> None:
        process = PACKAGE / "execution" / "process"
        for misplaced in ("run_directory.py", "submissions.py"):
            with self.subTest(module=misplaced):
                self.assertFalse((process / misplaced).exists())
        self.assertFalse((process / "agent" / "session.py").exists())


if __name__ == "__main__":
    unittest.main()
