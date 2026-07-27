"""Typed, public MiniGrid RedBlueDoors environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int]] = {
    "6x6": ("MiniGrid-RedBlueDoors-6x6-v0", 6),
    "8x8": ("MiniGrid-RedBlueDoors-8x8-v0", 8),
}


@dataclass(frozen=True, slots=True)
class RedBlueDoorsConfig:
    """Parameters defining one RedBlueDoors Benchmark identity."""

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
    def room_size(self) -> int:
        """Return the upstream room-size parameter."""

        return _PROFILES[self.profile][1]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 20 * self.room_size**2


__all__ = ["RedBlueDoorsConfig"]

