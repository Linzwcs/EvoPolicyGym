"""Typed configuration for Crafter reward and observation profiles."""

from dataclasses import dataclass
from typing import Literal

_OFFICIAL_MAX_EPISODE_STEPS = 10_000

type ObservationProfile = Literal["rgb", "local-symbolic-v1"]


@dataclass(frozen=True, slots=True)
class CrafterConfig:
    """Bind the Episode horizon, observation profile, and video feedback."""

    max_episode_steps: int = _OFFICIAL_MAX_EPISODE_STEPS
    include_mp4_feedback: bool = False
    observation_profile: ObservationProfile = "rgb"

    def __post_init__(self) -> None:
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= _OFFICIAL_MAX_EPISODE_STEPS:
            raise ValueError("max_episode_steps must be between 1 and 10000")
        if type(self.include_mp4_feedback) is not bool:
            raise TypeError("include_mp4_feedback must be bool")
        if type(self.observation_profile) is not str:
            raise TypeError("observation_profile must be an exact string")
        if self.observation_profile not in {"rgb", "local-symbolic-v1"}:
            raise ValueError(
                "observation_profile must be 'rgb' or 'local-symbolic-v1'"
            )
        if (
            self.observation_profile == "local-symbolic-v1"
            and self.include_mp4_feedback
        ):
            raise ValueError(
                "include_mp4_feedback is unavailable for local-symbolic-v1"
            )


__all__ = ["CrafterConfig", "ObservationProfile"]
