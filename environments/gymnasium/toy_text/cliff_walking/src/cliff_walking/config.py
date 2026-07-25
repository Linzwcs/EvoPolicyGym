"""Typed, public CliffWalking-v1 environment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliffWalkingConfig:
    """Parameters that define one CliffWalking Benchmark identity."""

    is_slippery: bool = False

    def __post_init__(self) -> None:
        if type(self.is_slippery) is not bool:
            raise TypeError("is_slippery must be an exact bool")


__all__ = ["CliffWalkingConfig"]
