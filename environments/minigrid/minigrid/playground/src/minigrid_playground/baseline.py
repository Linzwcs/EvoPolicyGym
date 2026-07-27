"""Packaged initial Program for MiniGrid Playground Runs."""

from __future__ import annotations

from importlib.resources import as_file, files

from evopolicygym import Program


def baseline_program() -> Program:
    """Return a detached snapshot of the public-observation baseline."""

    resource = files("minigrid_playground").joinpath("programs", "baseline")
    with as_file(resource) as directory:
        return Program.from_directory(directory)


__all__ = ["baseline_program"]

