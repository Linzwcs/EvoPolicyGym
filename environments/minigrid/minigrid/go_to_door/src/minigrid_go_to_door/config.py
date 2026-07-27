"""Typed, public MiniGrid GoToDoor environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int]] = {
    "5x5": ("MiniGrid-GoToDoor-5x5-v0", 5, 4),
    "6x6": ("MiniGrid-GoToDoor-6x6-v0", 6, 4),
    "8x8": ("MiniGrid-GoToDoor-8x8-v0", 8, 4),
}


@dataclass(frozen=True, slots=True)
class GoToDoorConfig:
    """Parameters that define one MiniGrid GoToDoor Benchmark identity."""

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
    def object_count(self) -> int:
        """Return the number of generated candidate doors."""

        return _PROFILES[self.profile][2]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 4 * self.size**2


__all__ = ["GoToDoorConfig"]
