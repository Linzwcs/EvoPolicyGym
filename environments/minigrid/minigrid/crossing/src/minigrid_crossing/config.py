"""Typed, public MiniGrid Crossing environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int, str]] = {
    "lava-S9-N1": ("MiniGrid-LavaCrossingS9N1-v0", 9, 1, "lava"),
    "lava-S9-N2": ("MiniGrid-LavaCrossingS9N2-v0", 9, 2, "lava"),
    "lava-S9-N3": ("MiniGrid-LavaCrossingS9N3-v0", 9, 3, "lava"),
    "lava-S11-N5": ("MiniGrid-LavaCrossingS11N5-v0", 11, 5, "lava"),
    "wall-S9-N1": ("MiniGrid-SimpleCrossingS9N1-v0", 9, 1, "wall"),
    "wall-S9-N2": ("MiniGrid-SimpleCrossingS9N2-v0", 9, 2, "wall"),
    "wall-S9-N3": ("MiniGrid-SimpleCrossingS9N3-v0", 9, 3, "wall"),
    "wall-S11-N5": ("MiniGrid-SimpleCrossingS11N5-v0", 11, 5, "wall"),
}


@dataclass(frozen=True, slots=True)
class CrossingConfig:
    """Parameters defining one Crossing Benchmark identity."""

    profile: str = "lava-S9-N3"

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
    def crossings(self) -> int:
        return _PROFILES[self.profile][2]

    @property
    def obstacle_type(self) -> str:
        return _PROFILES[self.profile][3]

    @property
    def max_episode_steps(self) -> int:
        return 4 * self.size**2


__all__ = ["CrossingConfig"]

