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
    binary_buttons: tuple[str, ...] = ()
    continuous_controls: tuple[str, ...] = ()
    game_variable_names: tuple[str, ...] = ()


_PROFILES = {
    "basic": _Profile(
        "VizdoomBasic-v1",
        300,
        4,
        1,
        binary_buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        game_variable_names=("AMMO2",),
    ),
    "basic-audio": _Profile(
        "VizdoomBasicAudio-v1",
        300,
        4,
        1,
        audio=True,
        binary_buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        game_variable_names=("AMMO2",),
    ),
    "basic-notifications": _Profile(
        "VizdoomBasicNotifications-v1",
        300,
        4,
        1,
        notifications=True,
        binary_buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        game_variable_names=("AMMO2",),
    ),
    "deadly-corridor": _Profile(
        "VizdoomDeadlyCorridor-v1",
        2_100,
        8,
        1,
        binary_buttons=(
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "ATTACK",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
        ),
        game_variable_names=("HEALTH",),
    ),
    "deathmatch": _Profile(
        "VizdoomDeathmatch-v1",
        4_200,
        18,
        5,
        hybrid_action=True,
        binary_buttons=(
            "ATTACK",
            "SPEED",
            "STRAFE",
            "MOVE_RIGHT",
            "MOVE_LEFT",
            "MOVE_BACKWARD",
            "MOVE_FORWARD",
            "TURN_RIGHT",
            "TURN_LEFT",
            "SELECT_WEAPON1",
            "SELECT_WEAPON2",
            "SELECT_WEAPON3",
            "SELECT_WEAPON4",
            "SELECT_WEAPON5",
            "SELECT_WEAPON6",
            "SELECT_NEXT_WEAPON",
            "SELECT_PREV_WEAPON",
        ),
        continuous_controls=(
            "LOOK_UP_DOWN_DELTA",
            "TURN_LEFT_RIGHT_DELTA",
            "MOVE_LEFT_RIGHT_DELTA",
        ),
        game_variable_names=(
            "KILLCOUNT",
            "HEALTH",
            "ARMOR",
            "SELECTED_WEAPON",
            "SELECTED_WEAPON_AMMO",
        ),
    ),
    "defend-center": _Profile(
        "VizdoomDefendCenter-v1",
        2_100,
        4,
        2,
        binary_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
        game_variable_names=("AMMO2", "HEALTH"),
    ),
    "defend-line": _Profile(
        "VizdoomDefendLine-v1",
        2_100,
        4,
        2,
        binary_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
        game_variable_names=("AMMO2", "HEALTH"),
    ),
    "health-gathering": _Profile(
        "VizdoomHealthGathering-v1",
        2_100,
        4,
        1,
        binary_buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
        game_variable_names=("HEALTH",),
    ),
    "health-gathering-supreme": _Profile(
        "VizdoomHealthGatheringSupreme-v1",
        2_100,
        4,
        1,
        binary_buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
        game_variable_names=("HEALTH",),
    ),
    "my-way-home": _Profile(
        "VizdoomMyWayHome-v1",
        2_100,
        6,
        1,
        binary_buttons=(
            "TURN_LEFT",
            "TURN_RIGHT",
            "MOVE_FORWARD",
            "MOVE_LEFT",
            "MOVE_RIGHT",
        ),
        game_variable_names=("AMMO0",),
    ),
    "predict-position": _Profile(
        "VizdoomPredictPosition-v1",
        300,
        4,
        0,
        binary_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    ),
    "take-cover": _Profile(
        "VizdoomTakeCover-v1",
        2_100,
        3,
        1,
        binary_buttons=("MOVE_LEFT", "MOVE_RIGHT"),
        game_variable_names=("HEALTH",),
    ),
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

    @property
    def action_meanings(self) -> tuple[str, ...]:
        profile = _PROFILES[self.profile]
        return (
            "noop",
            *(button.lower() for button in reversed(profile.binary_buttons)),
        )

    @property
    def continuous_controls(self) -> tuple[str, ...]:
        return _PROFILES[self.profile].continuous_controls

    @property
    def game_variable_names(self) -> tuple[str, ...]:
        return _PROFILES[self.profile].game_variable_names


__all__ = ["VIZDOOM_PROFILES", "ViZDoomConfig"]
