"""Typed configuration for the canonical Crafter reward profile."""

from dataclasses import dataclass

_OFFICIAL_MAX_EPISODE_STEPS = 10_000


@dataclass(frozen=True, slots=True)
class CrafterConfig:
    """Bind the Episode horizon for one Crafter Benchmark identity."""

    max_episode_steps: int = _OFFICIAL_MAX_EPISODE_STEPS

    def __post_init__(self) -> None:
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= _OFFICIAL_MAX_EPISODE_STEPS:
            raise ValueError("max_episode_steps must be between 1 and 10000")


__all__ = ["CrafterConfig"]
