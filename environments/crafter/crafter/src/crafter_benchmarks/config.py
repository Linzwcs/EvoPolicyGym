"""Typed configuration for the canonical Crafter reward profile."""

from dataclasses import dataclass

_OFFICIAL_MAX_EPISODE_STEPS = 10_000


@dataclass(frozen=True, slots=True)
class CrafterConfig:
    """Bind the Episode horizon and public replay presentation."""

    max_episode_steps: int = _OFFICIAL_MAX_EPISODE_STEPS
    replay_fps: int = 10
    replay_size: int = 256

    def __post_init__(self) -> None:
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= _OFFICIAL_MAX_EPISODE_STEPS:
            raise ValueError("max_episode_steps must be between 1 and 10000")
        if type(self.replay_fps) is not int:
            raise TypeError("replay_fps must be an exact integer")
        if not 1 <= self.replay_fps <= 60:
            raise ValueError("replay_fps must be between 1 and 60")
        if type(self.replay_size) is not int:
            raise TypeError("replay_size must be an exact integer")
        if (
            not 64 <= self.replay_size <= 512
            or self.replay_size % 16 != 0
        ):
            raise ValueError(
                "replay_size must be a multiple of 16 between 64 and 512"
            )


__all__ = ["CrafterConfig"]
