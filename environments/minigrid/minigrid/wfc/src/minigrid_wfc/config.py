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
_DEFAULT_PROFILES = frozenset(
    {
        "MazeSimple",
        "DungeonMazeScaled",
        "RoomsFabric",
        "ObstaclesBlackdots",
        "ObstaclesAngular",
        "ObstaclesHogs3",
    }
)
_INCONSISTENT_PROFILES = frozenset(
    {"ObstaclesHogs2", "MazeKnot", "MazeWall", "RoomsOffice", "Skew2"}
)


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

    @property
    def generation_class(self) -> str:
        """Return the upstream preset's expected generation-cost class."""

        if self.profile in _DEFAULT_PROFILES:
            return "default"
        if self.profile in _INCONSISTENT_PROFILES:
            return "inconsistent"
        return "slow"


__all__ = ["WFC_PROFILES", "WFCConfig"]
