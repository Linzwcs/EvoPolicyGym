"""Typed, public MiniGrid ObstructedMaze configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    rows: int
    columns: int
    horizon_rooms: int
    key_in_box: bool
    blocked: bool
    quarters: int
    locked_doors: int
    unlocked_doors: int
    key_blocker_overlap_possible: bool = False


_PROFILES: dict[str, _Profile] = {
    "1Dl-v0": _Profile("MiniGrid-ObstructedMaze-1Dl-v0", 1, 2, 2, False, False, 0, 1, 0),
    "1Dlh-v0": _Profile("MiniGrid-ObstructedMaze-1Dlh-v0", 1, 2, 2, True, False, 0, 1, 0),
    "1Dlhb-v0": _Profile("MiniGrid-ObstructedMaze-1Dlhb-v0", 1, 2, 2, True, True, 0, 1, 0),
    "2Dl-v0": _Profile("MiniGrid-ObstructedMaze-2Dl-v0", 3, 3, 4, False, False, 1, 2, 1),
    "2Dlh-v0": _Profile("MiniGrid-ObstructedMaze-2Dlh-v0", 3, 3, 4, True, False, 1, 2, 1),
    "2Dlhb-v0": _Profile("MiniGrid-ObstructedMaze-2Dlhb-v0", 3, 3, 4, True, True, 1, 2, 1, True),
    "2Dlhb-v1": _Profile("MiniGrid-ObstructedMaze-2Dlhb-v1", 3, 3, 4, True, True, 1, 2, 1),
    "1Q-v0": _Profile("MiniGrid-ObstructedMaze-1Q-v0", 3, 3, 5, True, True, 1, 2, 1, True),
    "1Q-v1": _Profile("MiniGrid-ObstructedMaze-1Q-v1", 3, 3, 5, True, True, 1, 2, 1),
    "2Q-v0": _Profile("MiniGrid-ObstructedMaze-2Q-v0", 3, 3, 11, True, True, 2, 4, 2, True),
    "2Q-v1": _Profile("MiniGrid-ObstructedMaze-2Q-v1", 3, 3, 11, True, True, 2, 4, 2),
    "Full-v0": _Profile("MiniGrid-ObstructedMaze-Full-v0", 3, 3, 25, True, True, 4, 8, 4, True),
    "Full-v1": _Profile("MiniGrid-ObstructedMaze-Full-v1", 3, 3, 25, True, True, 4, 8, 4),
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
    def _selected(self) -> _Profile:
        return _PROFILES[self.profile]

    @property
    def environment_id(self) -> str:
        return self._selected.environment_id

    @property
    def rows(self) -> int:
        return self._selected.rows

    @property
    def columns(self) -> int:
        return self._selected.columns

    @property
    def horizon_rooms(self) -> int:
        return self._selected.horizon_rooms

    @property
    def key_in_box(self) -> bool:
        return self._selected.key_in_box

    @property
    def blocked(self) -> bool:
        return self._selected.blocked

    @property
    def quarters(self) -> int:
        return self._selected.quarters

    @property
    def locked_doors(self) -> int:
        return self._selected.locked_doors

    @property
    def unlocked_doors(self) -> int:
        return self._selected.unlocked_doors

    @property
    def key_blocker_overlap_possible(self) -> bool:
        return self._selected.key_blocker_overlap_possible

    @property
    def max_episode_steps(self) -> int:
        return 4 * self.horizon_rooms * 6**2


__all__ = ["ObstructedMazeConfig"]
