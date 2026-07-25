"""Typed public configuration for the supported Balatro profile."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BalatroConfig:
    """Environment parameters bound to one Balatro Benchmark instance."""

    deck: str = "b_red"
    stake: int = 1

    def __post_init__(self) -> None:
        if type(self.deck) is not str:
            raise TypeError("deck must be text")
        if self.deck != "b_red":
            raise ValueError("the supported Balatro profile requires deck='b_red'")
        if type(self.stake) is not int:
            raise TypeError("stake must be an integer")
        if self.stake != 1:
            raise ValueError("the supported Balatro profile requires stake=1")


__all__ = ["BalatroConfig"]
