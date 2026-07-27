"""Typed, public MiniGrid ObstructedMaze configuration."""

from __future__ import annotations

from dataclasses import dataclass

type Profile = tuple[str, int, int, int, bool, bool]

_PROFILES: dict[str, Profile] = {
    "1Dl-v0": ("MiniGrid-ObstructedMaze-1Dl-v0", 1, 2, 2, False, False),
    "1Dlh-v0": ("MiniGrid-ObstructedMaze-1Dlh-v0", 1, 2, 2, True, False),
    "1Dlhb-v0": ("MiniGrid-ObstructedMaze-1Dlhb-v0", 1, 2, 2, True, True),
    "2Dl-v0": ("MiniGrid-ObstructedMaze-2Dl-v0", 3, 3, 4, False, False),
    "2Dlh-v0": ("MiniGrid-ObstructedMaze-2Dlh-v0", 3, 3, 4, True, False),
    "2Dlhb-v0": ("MiniGrid-ObstructedMaze-2Dlhb-v0", 3, 3, 4, True, True),
    "2Dlhb-v1": ("MiniGrid-ObstructedMaze-2Dlhb-v1", 3, 3, 4, True, True),
    "1Q-v0": ("MiniGrid-ObstructedMaze-1Q-v0", 3, 3, 5, True, True),
    "1Q-v1": ("MiniGrid-ObstructedMaze-1Q-v1", 3, 3, 5, True, True),
    "2Q-v0": ("MiniGrid-ObstructedMaze-2Q-v0", 3, 3, 11, True, True),
    "2Q-v1": ("MiniGrid-ObstructedMaze-2Q-v1", 3, 3, 11, True, True),
    "Full-v0": ("MiniGrid-ObstructedMaze-Full-v0", 3, 3, 25, True, True),
    "Full-v1": ("MiniGrid-ObstructedMaze-Full-v1", 3, 3, 25, True, True),
}


@dataclass(frozen=True, slots=True)
class ObstructedMazeConfig:
    """Parameters defining one ObstructedMaze Benchmark identity."""

    profile: str = "1Dlhb-v0"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile][0]

    @property
    def rows(self) -> int:
        return _PROFILES[self.profile][1]

    @property
    def columns(self) -> int:
        return _PROFILES[self.profile][2]

    @property
    def horizon_rooms(self) -> int:
        return _PROFILES[self.profile][3]

    @property
    def key_in_box(self) -> bool:
        return _PROFILES[self.profile][4]

    @property
    def blocked(self) -> bool:
        return _PROFILES[self.profile][5]

    @property
    def max_episode_steps(self) -> int:
        return 4 * self.horizon_rooms * 6**2


__all__ = ["ObstructedMazeConfig"]
