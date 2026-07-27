"""Typed, public MiniGrid DoorKey environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int]] = {
    "5x5": ("MiniGrid-DoorKey-5x5-v0", 5),
    "6x6": ("MiniGrid-DoorKey-6x6-v0", 6),
    "8x8": ("MiniGrid-DoorKey-8x8-v0", 8),
    "16x16": ("MiniGrid-DoorKey-16x16-v0", 16),
}


@dataclass(frozen=True, slots=True)
class DoorKeyConfig:
    """Parameters that define one MiniGrid DoorKey Benchmark identity."""

    profile: str = "8x8"

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
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 10 * self.size**2


__all__ = ["DoorKeyConfig"]
