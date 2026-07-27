"""Typed Host-selected ViZDoom scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    max_episode_steps: int
    action_size: int
    game_variables: int
    audio: bool = False
    notifications: bool = False
    hybrid_action: bool = False


_PROFILES = {
    "basic": _Profile("VizdoomBasic-v1", 300, 4, 1),
    "basic-audio": _Profile(
        "VizdoomBasicAudio-v1", 300, 4, 1, audio=True
    ),
    "basic-notifications": _Profile(
        "VizdoomBasicNotifications-v1",
        300,
        4,
        1,
        notifications=True,
    ),
    "deadly-corridor": _Profile(
        "VizdoomDeadlyCorridor-v1", 2_100, 8, 1
    ),
    "deathmatch": _Profile(
        "VizdoomDeathmatch-v1",
        4_200,
        18,
        5,
        hybrid_action=True,
    ),
    "defend-center": _Profile(
        "VizdoomDefendCenter-v1", 2_100, 4, 2
    ),
    "defend-line": _Profile("VizdoomDefendLine-v1", 2_100, 4, 2),
    "health-gathering": _Profile(
        "VizdoomHealthGathering-v1", 2_100, 4, 1
    ),
    "health-gathering-supreme": _Profile(
        "VizdoomHealthGatheringSupreme-v1", 2_100, 4, 1
    ),
    "my-way-home": _Profile("VizdoomMyWayHome-v1", 2_100, 6, 1),
    "predict-position": _Profile(
        "VizdoomPredictPosition-v1", 300, 4, 0
    ),
    "take-cover": _Profile("VizdoomTakeCover-v1", 2_100, 3, 1),
}

VIZDOOM_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class ViZDoomConfig:
    """Configuration fixing one bundled ViZDoom scenario."""

    profile: str = "basic"

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError(
                "profile must be one of: " + ", ".join(VIZDOOM_PROFILES)
            )

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile].environment_id

    @property
    def max_episode_steps(self) -> int:
        return _PROFILES[self.profile].max_episode_steps

    @property
    def action_size(self) -> int:
        return _PROFILES[self.profile].action_size

    @property
    def game_variables(self) -> int:
        return _PROFILES[self.profile].game_variables

    @property
    def audio(self) -> bool:
        return _PROFILES[self.profile].audio

    @property
    def notifications(self) -> bool:
        return _PROFILES[self.profile].notifications

    @property
    def hybrid_action(self) -> bool:
        return _PROFILES[self.profile].hybrid_action


__all__ = ["VIZDOOM_PROFILES", "ViZDoomConfig"]
