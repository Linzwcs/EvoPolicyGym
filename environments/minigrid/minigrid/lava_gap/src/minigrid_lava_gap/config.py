"""Typed, public MiniGrid LavaGap environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int]] = {
    "S5": ("MiniGrid-LavaGapS5-v0", 5),
    "S6": ("MiniGrid-LavaGapS6-v0", 6),
    "S7": ("MiniGrid-LavaGapS7-v0", 7),
}


@dataclass(frozen=True, slots=True)
class LavaGapConfig:
    """Parameters defining one LavaGap Benchmark identity."""

    profile: str = "S7"

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
    def max_episode_steps(self) -> int:
        return 4 * self.size**2


__all__ = ["LavaGapConfig"]
