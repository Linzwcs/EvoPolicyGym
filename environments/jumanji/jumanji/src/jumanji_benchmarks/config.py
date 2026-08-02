"""Typed Host-selected Jumanji task profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    category: str
    max_episode_steps: int
    action_num_values: tuple[int, ...]
    action_mask: bool = True

    @property
    def action_kind(self) -> str:
        return "discrete" if len(self.action_num_values) == 1 else "multi_discrete"


_PROFILES = {
    "game-2048": _Profile("Game2048-v1", "logic", 1_000, (4,)),
    "graph-coloring": _Profile("GraphColoring-v1", "logic", 20, (20,)),
    "minesweeper": _Profile("Minesweeper-v0", "logic", 100, (10, 10)),
    "rubiks-cube": _Profile("RubiksCube-v0", "logic", 200, (6, 1, 3), False),
    "rubiks-cube-partly-scrambled": _Profile(
        "RubiksCube-partly-scrambled-v0",
        "logic",
        20,
        (6, 1, 3),
        False,
    ),
    "sudoku": _Profile("Sudoku-v0", "logic", 81, (9, 9, 9)),
    "sudoku-very-easy": _Profile("Sudoku-very-easy-v0", "logic", 81, (9, 9, 9)),
    "sliding-tile-puzzle": _Profile("SlidingTilePuzzle-v0", "logic", 500, (4,)),
    "bin-pack": _Profile("BinPack-v2", "packing", 20, (40, 20)),
    "flat-pack": _Profile("FlatPack-v0", "packing", 25, (25, 4, 9, 9)),
    "job-shop": _Profile("JobShop-v0", "packing", 960, (21,) * 10),
    "knapsack": _Profile("Knapsack-v1", "packing", 50, (50,)),
    "tetris": _Profile("Tetris-v0", "packing", 400, (4, 10)),
    "cvrp": _Profile("CVRP-v1", "routing", 40, (21,)),
    "maze": _Profile("Maze-v0", "routing", 100, (4,)),
    "snake": _Profile("Snake-v1", "routing", 4_000, (4,)),
    "tsp": _Profile("TSP-v1", "routing", 20, (20,)),
    "pacman": _Profile("PacMan-v1", "routing", 1_000, (5,)),
}

JUMANJI_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class JumanjiConfig:
    """Configuration that fixes one Jumanji Benchmark identity."""

    profile: str = "maze"

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError("profile must be one of: " + ", ".join(JUMANJI_PROFILES))

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile].environment_id

    @property
    def category(self) -> str:
        return _PROFILES[self.profile].category

    @property
    def max_episode_steps(self) -> int:
        return _PROFILES[self.profile].max_episode_steps

    @property
    def action_kind(self) -> str:
        return _PROFILES[self.profile].action_kind

    @property
    def action_num_values(self) -> tuple[int, ...]:
        return _PROFILES[self.profile].action_num_values

    @property
    def has_action_mask(self) -> bool:
        return _PROFILES[self.profile].action_mask


__all__ = ["JUMANJI_PROFILES", "JumanjiConfig"]
