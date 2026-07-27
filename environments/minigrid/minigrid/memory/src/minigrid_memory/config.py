"""Typed, public MiniGrid Memory environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, bool]] = {
    "11x11": ("MiniGrid-MemoryS11-v0", 11, False),
    "13x13": ("MiniGrid-MemoryS13-v0", 13, False),
    "13x13-random": ("MiniGrid-MemoryS13Random-v0", 13, True),
    "17x17-random": ("MiniGrid-MemoryS17Random-v0", 17, True),
}


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Parameters that define one MiniGrid Memory Benchmark identity."""

    profile: str = "13x13-random"

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
        """Return the outer grid size."""

        return _PROFILES[self.profile][1]

    @property
    def random_length(self) -> bool:
        """Return whether the corridor length varies by Episode."""

        return _PROFILES[self.profile][2]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 5 * self.size**2


__all__ = ["MemoryConfig"]
