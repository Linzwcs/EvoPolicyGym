"""Typed, public Blackjack-v1 environment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlackjackConfig:
    """Parameters that define one Blackjack Benchmark identity."""

    natural: bool = False
    sab: bool = True

    def __post_init__(self) -> None:
        if type(self.natural) is not bool:
            raise TypeError("natural must be an exact bool")
        if type(self.sab) is not bool:
            raise TypeError("sab must be an exact bool")


__all__ = ["BlackjackConfig"]
