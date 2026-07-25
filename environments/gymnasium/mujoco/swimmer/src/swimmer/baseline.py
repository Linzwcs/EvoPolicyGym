"""Packaged initial Program for Swimmer development Runs."""

from __future__ import annotations

from importlib.resources import as_file, files

from evopolicygym import Program


def baseline_program() -> Program:
    """Return a detached snapshot of the intentionally weak baseline."""

    resource = files("swimmer").joinpath(
        "programs",
        "baseline",
    )
    with as_file(resource) as directory:
        return Program.from_directory(directory)


__all__ = ["baseline_program"]
