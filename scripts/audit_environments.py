#!/usr/bin/env python3
"""Audit every independently installable Environment distribution."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

type AuditLevel = Literal["smoke", "full"]

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_PRUNED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "dist",
        "vendor",
    }
)


@dataclass(frozen=True, slots=True)
class EnvironmentProject:
    """One leaf Environment distribution discovered in the repository."""

    repository_root: Path
    path: Path

    def __post_init__(self) -> None:
        root = self.repository_root.resolve()
        path = self.path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Environment project must be inside the repository") from error
        object.__setattr__(self, "repository_root", root)
        object.__setattr__(self, "path", path)

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(self.repository_root).as_posix()


@dataclass(frozen=True, slots=True)
class AuditCommand:
    """One subprocess check in an Environment audit plan."""

    label: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured outcome of one audit command."""

    command: AuditCommand
    returncode: int
    duration_seconds: float
    output: str


@dataclass(frozen=True, slots=True)
class ProjectResult:
    """Complete result for one Environment distribution."""

    project: EnvironmentProject
    layout_issues: tuple[str, ...]
    commands: tuple[CommandResult, ...]

    @property
    def passed(self) -> bool:
        return not self.layout_issues and all(
            result.returncode == 0 for result in self.commands
        )

    @property
    def duration_seconds(self) -> float:
        return sum(result.duration_seconds for result in self.commands)


def discover_projects(repository_root: Path) -> tuple[EnvironmentProject, ...]:
    """Discover leaf projects without traversing virtual or vendored trees."""

    root = repository_root.resolve()
    environments = root / "environments"
    projects: list[EnvironmentProject] = []
    if not environments.is_dir():
        return ()

    for directory, child_directories, filenames in os.walk(environments):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name not in _PRUNED_DIRECTORIES
        )
        if "pyproject.toml" not in filenames:
            continue
        projects.append(
            EnvironmentProject(repository_root=root, path=Path(directory))
        )
        child_directories.clear()

    return tuple(sorted(projects, key=lambda project: project.relative_path))


def layout_issues(project: EnvironmentProject) -> tuple[str, ...]:
    """Return missing files required by an independent distribution."""

    issues: list[str] = []
    for filename in ("pyproject.toml", "uv.lock"):
        if not (project.path / filename).is_file():
            issues.append(f"missing {filename}")
    for directory in ("src", "tests"):
        if not (project.path / directory).is_dir():
            issues.append(f"missing {directory}/")
    return tuple(issues)


def commands_for(
    level: AuditLevel,
    *,
    skip_sync: bool,
) -> tuple[AuditCommand, ...]:
    """Build the deterministic command plan for an audit level."""

    commands: list[AuditCommand] = []
    if not skip_sync:
        commands.append(
            AuditCommand(
                label="sync",
                arguments=("uv", "sync", "--locked", "--extra", "dev"),
            )
        )
    if level == "full":
        commands.extend(
            (
                AuditCommand(
                    label="ruff",
                    arguments=("uv", "run", "ruff", "check", "src", "tests"),
                ),
                AuditCommand(
                    label="mypy",
                    arguments=("uv", "run", "mypy", "--no-incremental"),
                ),
            )
        )
    commands.append(
        AuditCommand(
            label="tests",
            arguments=(
                "uv",
                "run",
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
            ),
        )
    )
    if level == "full":
        commands.append(
            AuditCommand(label="build", arguments=("uv", "build"))
        )
    return tuple(commands)


def select_projects(
    projects: Sequence[EnvironmentProject],
    patterns: Sequence[str],
) -> tuple[EnvironmentProject, ...]:
    """Select projects by repository-relative shell-style patterns."""

    if not patterns:
        return tuple(projects)
    normalized = tuple(pattern.removeprefix("./") for pattern in patterns)
    selected = tuple(
        project
        for project in projects
        if any(
            fnmatch.fnmatchcase(project.relative_path, pattern)
            for pattern in normalized
        )
    )
    if not selected:
        joined = ", ".join(patterns)
        raise ValueError(f"no Environment projects matched: {joined}")
    return selected


def audit_project(
    project: EnvironmentProject,
    commands: Sequence[AuditCommand],
    *,
    timeout_seconds: int,
) -> ProjectResult:
    """Run one project's commands, stopping that project at its first failure."""

    issues = layout_issues(project)
    if issues:
        return ProjectResult(project=project, layout_issues=issues, commands=())

    results: list[CommandResult] = []
    for command in commands:
        result = _run_command(
            project,
            command,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)
        if result.returncode != 0:
            break
    return ProjectResult(
        project=project,
        layout_issues=(),
        commands=tuple(results),
    )


def audit_projects(
    projects: Sequence[EnvironmentProject],
    commands: Sequence[AuditCommand],
    *,
    jobs: int,
    timeout_seconds: int,
    on_result: Callable[[ProjectResult, int, int], None] | None = None,
) -> tuple[ProjectResult, ...]:
    """Audit projects concurrently and return results in path order."""

    results: list[ProjectResult] = []
    total = len(projects)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(
                audit_project,
                project,
                commands,
                timeout_seconds=timeout_seconds,
            ): project
            for project in projects
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result, completed, total)
    return tuple(sorted(results, key=lambda result: result.project.relative_path))


def _run_command(
    project: EnvironmentProject,
    command: AuditCommand,
    *,
    timeout_seconds: int,
) -> CommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command.arguments,
            cwd=project.path,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired:
        returncode = 124
        output = f"timed out after {timeout_seconds} seconds"
    except OSError as error:
        returncode = 127
        output = str(error)
    return CommandResult(
        command=command,
        returncode=returncode,
        duration_seconds=time.perf_counter() - started,
        output=output,
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="repository root (defaults to the script's repository)",
    )
    parser.add_argument(
        "--level",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke runs locked sync and tests; full mirrors Environment CI",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        metavar="GLOB",
        help="repository-relative project glob; repeat to select multiple sets",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_integer,
        default=1,
        help="number of Environment projects to audit concurrently",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_integer,
        default=2700,
        help="timeout for each individual audit command",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="reuse existing project virtual environments",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected Environment projects without running commands",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command plan without running commands",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print captured output for successful commands too",
    )
    return parser


def _print_result(
    result: ProjectResult,
    completed: int,
    total: int,
    *,
    verbose: bool,
) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[{completed}/{total}] {status} {result.project.relative_path} "
        f"({result.duration_seconds:.1f}s)",
        flush=True,
    )
    for issue in result.layout_issues:
        print(f"  layout: {issue}", flush=True)
    for command in result.commands:
        if command.returncode == 0 and not verbose:
            continue
        rendered = shlex.join(command.command.arguments)
        print(
            f"  {command.command.label}: exit={command.returncode} "
            f"command={rendered}",
            flush=True,
        )
        if command.output:
            for line in command.output.rstrip().splitlines():
                print(f"    {line}", flush=True)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    repository_root = cast(Path, namespace.root)
    level = cast(AuditLevel, namespace.level)
    patterns = cast(list[str], namespace.project)
    jobs = cast(int, namespace.jobs)
    timeout_seconds = cast(int, namespace.timeout_seconds)
    skip_sync = cast(bool, namespace.skip_sync)
    list_only = cast(bool, namespace.list)
    dry_run = cast(bool, namespace.dry_run)
    verbose = cast(bool, namespace.verbose)

    projects = discover_projects(repository_root)
    if not projects:
        parser.error(f"no Environment projects found under {repository_root}")
    try:
        selected = select_projects(projects, patterns)
    except ValueError as error:
        parser.error(str(error))

    if list_only:
        for project in selected:
            print(project.relative_path)
        return 0

    commands = commands_for(level, skip_sync=skip_sync)
    if dry_run:
        dry_run_failed = False
        for project in selected:
            print(project.relative_path)
            for issue in layout_issues(project):
                dry_run_failed = True
                print(f"  layout: {issue}")
            for command in commands:
                print(f"  {command.label}: {shlex.join(command.arguments)}")
        return 1 if dry_run_failed else 0

    print(
        f"Auditing {len(selected)} Environment projects at level={level} "
        f"with jobs={jobs}",
        flush=True,
    )
    results = audit_projects(
        selected,
        commands,
        jobs=min(jobs, len(selected)),
        timeout_seconds=timeout_seconds,
        on_result=lambda result, completed, total: _print_result(
            result,
            completed,
            total,
            verbose=verbose,
        ),
    )
    passed = sum(result.passed for result in results)
    failed_count = len(results) - passed
    print(
        f"Summary: {passed} passed, {failed_count} failed, {len(results)} total",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
