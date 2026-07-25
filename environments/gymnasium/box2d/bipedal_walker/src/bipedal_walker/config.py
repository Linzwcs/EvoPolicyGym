"""Typed, public BipedalWalker-v3 environment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BipedalWalkerConfig:
    """Parameters that define one BipedalWalker Benchmark identity."""

    hardcore: bool = False

    def __post_init__(self) -> None:
        if type(self.hardcore) is not bool:
            raise TypeError("hardcore must be an exact bool")


__all__ = ["BipedalWalkerConfig"]
