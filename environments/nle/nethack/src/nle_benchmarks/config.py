"""Typed configuration for the deterministic NLE score profile."""

from dataclasses import dataclass

OFFICIAL_MAX_EPISODE_STEPS = 5_000


@dataclass(frozen=True, slots=True)
class NetHackConfig:
    """Bind the public Episode horizon for one Benchmark identity."""

    max_episode_steps: int = OFFICIAL_MAX_EPISODE_STEPS

    def __post_init__(self) -> None:
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= OFFICIAL_MAX_EPISODE_STEPS:
            raise ValueError("max_episode_steps must be between 1 and 5000")


__all__ = ["NetHackConfig"]
