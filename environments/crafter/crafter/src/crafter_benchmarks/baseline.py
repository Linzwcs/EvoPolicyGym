"""Packaged deterministic starting Program for Crafter development."""

from __future__ import annotations

import shutil
import tempfile
from importlib.resources import as_file, files
from pathlib import Path

from evopolicygym import Program


def baseline_program() -> Program:
    """Return the weak baseline with its ordinary gameplay reference."""

    package = files("crafter_benchmarks")
    policy_resource = package.joinpath(
        "programs",
        "baseline",
    )
    guide_resource = package.joinpath(
        "background",
        "PLAYER_GUIDE.md",
    )
    with as_file(policy_resource) as policy_directory:
        with tempfile.TemporaryDirectory(prefix="crafter-baseline-") as temporary:
            directory = Path(temporary) / "program"
            shutil.copytree(policy_directory, directory)
            (directory / "PLAYER_GUIDE.md").write_bytes(
                guide_resource.read_bytes()
            )
            return Program.from_directory(directory)


def local_symbolic_baseline_program() -> Program:
    """Return the weak baseline matching local-symbolic-v1 observations."""

    package = files("crafter_benchmarks")
    policy_resource = package.joinpath(
        "programs",
        "local_symbolic_baseline",
    )
    guide_resource = package.joinpath(
        "background",
        "PLAYER_GUIDE.md",
    )
    with as_file(policy_resource) as policy_directory:
        with tempfile.TemporaryDirectory(
            prefix="crafter-local-symbolic-baseline-"
        ) as temporary:
            directory = Path(temporary) / "program"
            shutil.copytree(policy_directory, directory)
            (directory / "PLAYER_GUIDE.md").write_bytes(
                guide_resource.read_bytes()
            )
            return Program.from_directory(directory)


__all__ = ["baseline_program", "local_symbolic_baseline_program"]
