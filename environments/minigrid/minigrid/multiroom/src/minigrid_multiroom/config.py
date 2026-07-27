"""Typed, public MiniGrid MultiRoom environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int, int]] = {
    "N2-S4": ("MiniGrid-MultiRoom-N2-S4-v0", 2, 2, 4),
    "N4-S5": ("MiniGrid-MultiRoom-N4-S5-v1", 4, 4, 5),
    "N4-S5-v0-legacy-N6": (
        "MiniGrid-MultiRoom-N4-S5-v0",
        6,
        6,
        5,
    ),
    "N6-S10": ("MiniGrid-MultiRoom-N6-v0", 6, 6, 10),
}


@dataclass(frozen=True, slots=True)
class MultiRoomConfig:
    """Parameters that define one MiniGrid MultiRoom Benchmark identity."""

    profile: str = "N6-S10"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        """Return the upstream Gymnasium registration."""

        return _PROFILES[self.profile][0]

    @property
    def minimum_rooms(self) -> int:
        """Return the minimum generated room count."""

        return _PROFILES[self.profile][1]

    @property
    def maximum_rooms(self) -> int:
        """Return the maximum generated room count."""

        return _PROFILES[self.profile][2]

    @property
    def maximum_room_size(self) -> int:
        """Return the upstream maximum room dimension."""

        return _PROFILES[self.profile][3]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return self.maximum_rooms * 20


__all__ = ["MultiRoomConfig"]

