"""Typed, public MiniGrid Empty environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, bool]] = {
    "5x5": ("MiniGrid-Empty-5x5-v0", 5, False),
    "5x5-random": ("MiniGrid-Empty-Random-5x5-v0", 5, True),
    "6x6": ("MiniGrid-Empty-6x6-v0", 6, False),
    "6x6-random": ("MiniGrid-Empty-Random-6x6-v0", 6, True),
    "8x8": ("MiniGrid-Empty-8x8-v0", 8, False),
    "16x16": ("MiniGrid-Empty-16x16-v0", 16, False),
}


@dataclass(frozen=True, slots=True)
class EmptyConfig:
    """Parameters defining one MiniGrid Empty Benchmark identity."""

    profile: str = "8x8"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile][0]

    @property
    def size(self) -> int:
        return _PROFILES[self.profile][1]

    @property
    def random_start(self) -> bool:
        return _PROFILES[self.profile][2]

    @property
    def max_episode_steps(self) -> int:
        return 4 * self.size**2


__all__ = ["EmptyConfig"]
