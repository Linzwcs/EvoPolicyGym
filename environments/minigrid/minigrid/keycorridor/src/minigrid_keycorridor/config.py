"""Typed, public MiniGrid KeyCorridor environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int]] = {
    "S3R1": ("MiniGrid-KeyCorridorS3R1-v0", 3, 1),
    "S3R2": ("MiniGrid-KeyCorridorS3R2-v0", 3, 2),
    "S3R3": ("MiniGrid-KeyCorridorS3R3-v0", 3, 3),
    "S4R3": ("MiniGrid-KeyCorridorS4R3-v0", 4, 3),
    "S5R3": ("MiniGrid-KeyCorridorS5R3-v0", 5, 3),
    "S6R3": ("MiniGrid-KeyCorridorS6R3-v0", 6, 3),
}


@dataclass(frozen=True, slots=True)
class KeyCorridorConfig:
    """Parameters that define one MiniGrid KeyCorridor Benchmark identity."""

    profile: str = "S4R3"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        """Return the upstream Gymnasium registration."""

        return _PROFILES[self.profile][0]

    @property
    def room_size(self) -> int:
        """Return the upstream room-size parameter."""

        return _PROFILES[self.profile][1]

    @property
    def num_rows(self) -> int:
        """Return the number of room rows."""

        return _PROFILES[self.profile][2]

    @property
    def grid_width(self) -> int:
        """Return the complete three-column grid width."""

        return 3 * (self.room_size - 1) + 1

    @property
    def grid_height(self) -> int:
        """Return the complete grid height."""

        return self.num_rows * (self.room_size - 1) + 1

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 30 * self.room_size**2


__all__ = ["KeyCorridorConfig"]
