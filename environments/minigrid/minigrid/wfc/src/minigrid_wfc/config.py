"""Typed, public MiniGrid WFC environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

WFC_PROFILES = (
    "MazeSimple",
    "DungeonMazeScaled",
    "RoomsFabric",
    "ObstaclesBlackdots",
    "ObstaclesAngular",
    "ObstaclesHogs2",
    "ObstaclesHogs3",
    "MazeKnot",
    "MazeWall",
    "Maze",
    "MazeSpirals",
    "MazePaths",
    "Mazelike",
    "RoomsOffice",
    "RoomsMagicOffice",
    "Dungeon",
    "DungeonRooms",
    "DungeonLessRooms",
    "DungeonSpirals",
    "Skew2",
    "SkewCave",
    "SkewLake",
)
_PROFILE_SET = frozenset(WFC_PROFILES)
_SIZES = frozenset({15, 25})


@dataclass(frozen=True, slots=True)
class WFCConfig:
    """Parameters defining one generated WFC Benchmark identity."""

    profile: str = "MazeSimple"
    size: int = 25

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILE_SET:
            choices = "', '".join(WFC_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")
        if type(self.size) is not int or self.size not in _SIZES:
            raise ValueError("size must be 15 or 25")

    @property
    def environment_id(self) -> str:
        return f"MiniGrid-WFC-{self.profile}-v0"

    @property
    def max_episode_steps(self) -> int:
        return 4 * self.size**2


__all__ = ["WFC_PROFILES", "WFCConfig"]
