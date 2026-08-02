"""Packaged observation-aware starting Program for NLE NetHack."""

from importlib.resources import as_file, files

from evopolicygym import Program


def baseline_program() -> Program:
    resource = files("nle_benchmarks").joinpath("programs", "baseline")
    with as_file(resource) as directory:
        return Program.from_directory(directory)


__all__ = ["baseline_program"]
