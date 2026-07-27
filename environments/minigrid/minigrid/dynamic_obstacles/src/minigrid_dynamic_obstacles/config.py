"""Typed, public MiniGrid DynamicObstacles environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int, bool]] = {
    "5x5-N2": ("MiniGrid-Dynamic-Obstacles-5x5-v0", 5, 2, False),
    "5x5-N2-random": (
        "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        5,
        2,
        True,
    ),
    "6x6-N3": ("MiniGrid-Dynamic-Obstacles-6x6-v0", 6, 3, False),
    "6x6-N3-random": (
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        6,
        3,
        True,
    ),
    "8x8-N4": ("MiniGrid-Dynamic-Obstacles-8x8-v0", 8, 4, False),
    "16x16-N8": ("MiniGrid-Dynamic-Obstacles-16x16-v0", 16, 8, False),
}


@dataclass(frozen=True, slots=True)
class DynamicObstaclesConfig:
    """Parameters defining one DynamicObstacles Benchmark identity."""

    profile: str = "8x8-N4"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        """Return the upstream Gymnasium registration."""

        return _PROFILES[self.profile][0]

    @property
    def size(self) -> int:
        """Return the square grid size."""

        return _PROFILES[self.profile][1]

    @property
    def obstacle_count(self) -> int:
        """Return the moving obstacle count."""

        return _PROFILES[self.profile][2]

    @property
    def random_start(self) -> bool:
        """Return whether the upstream agent start is randomized."""

        return _PROFILES[self.profile][3]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 4 * self.size**2


__all__ = ["DynamicObstaclesConfig"]

