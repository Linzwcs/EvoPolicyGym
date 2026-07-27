"""Typed, public MiniGrid DistShift environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int]] = {
    "shift1": ("MiniGrid-DistShift1-v0", 2),
    "shift2": ("MiniGrid-DistShift2-v0", 5),
}


@dataclass(frozen=True, slots=True)
class DistShiftConfig:
    """Parameters defining one DistShift Benchmark identity."""

    profile: str = "shift1"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile][0]

    @property
    def strip2_row(self) -> int:
        return _PROFILES[self.profile][1]

    @property
    def max_episode_steps(self) -> int:
        return 4 * 9 * 7


__all__ = ["DistShiftConfig"]
