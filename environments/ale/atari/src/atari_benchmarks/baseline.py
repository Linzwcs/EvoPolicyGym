"""Packaged no-op starting Program for ALE Tetris."""

from importlib.resources import as_file, files

from evopolicygym import Program


def baseline_program() -> Program:
    resource = files("atari_benchmarks").joinpath("programs", "baseline")
    with as_file(resource) as directory:
        return Program.from_directory(directory)


__all__ = ["baseline_program"]
