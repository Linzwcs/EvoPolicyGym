"""One fresh Gymnasium Pusher-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import PusherConfig

_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "upper_arm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
)
_OBSERVATION_NAMES = (
    *(f"{name}_angle" for name in _JOINT_NAMES),
    *(f"{name}_angular_velocity" for name in _JOINT_NAMES),
    "fingertip_x",
    "fingertip_y",
    "fingertip_z",
    "object_x",
    "object_y",
    "object_z",
    "goal_x",
    "goal_y",
    "goal_z",
)


class PusherEnvironment:
    """The seeded strict adapter around configured Pusher-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: PusherConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not PusherConfig:
            raise TypeError("config must be PusherConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Pusher configuration belongs in PusherConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Pusher-v5",
                frame_skip=config.frame_skip,
                reward_near_weight=config.reward_near_weight,
                reward_dist_weight=config.reward_dist_weight,
                reward_control_weight=config.reward_control_weight,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        observation, reward, terminated, truncated, info = (
            self._environment.step(_action(action))
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Pusher returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
            metrics=_reward_metrics(info),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 7:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -2.0 <= item <= 2.0
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (23,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Pusher returned an invalid observation")
    return {
        name: _number(item, name=name)
        for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("Pusher returned invalid reward metrics")
    required = {"reward_dist", "reward_ctrl", "reward_near"}
    if not required.issubset(value):
        raise RuntimeError("Pusher omitted reward metrics")
    return {
        "reward_distance": _number(
            value["reward_dist"],
            name="reward distance",
        ),
        "reward_control": _number(
            value["reward_ctrl"],
            name="reward control",
        ),
        "reward_near": _number(
            value["reward_near"],
            name="reward near",
        ),
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Pusher returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Pusher returned a non-finite {name}")
    return number


__all__ = ["PusherEnvironment"]
