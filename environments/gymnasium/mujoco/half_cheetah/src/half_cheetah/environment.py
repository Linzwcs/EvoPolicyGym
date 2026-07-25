"""One fresh Gymnasium HalfCheetah-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import HalfCheetahConfig

_BODY_FIELDS = (
    "front_tip_z_position",
    "front_tip_angle",
    "back_thigh_angle",
    "back_shin_angle",
    "back_foot_angle",
    "front_thigh_angle",
    "front_shin_angle",
    "front_foot_angle",
    "front_tip_x_velocity",
    "front_tip_z_velocity",
    "front_tip_angular_velocity",
    "back_thigh_angular_velocity",
    "back_shin_angular_velocity",
    "back_foot_angular_velocity",
    "front_thigh_angular_velocity",
    "front_shin_angular_velocity",
    "front_foot_angular_velocity",
)


class HalfCheetahEnvironment:
    """The seeded strict adapter around configured HalfCheetah-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HalfCheetahConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HalfCheetahConfig:
            raise TypeError("config must be HalfCheetahConfig")
        if episode.scenario is not None:
            raise ValueError(
                "HalfCheetah configuration belongs in HalfCheetahConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._exclude_positions = (
            config.exclude_current_positions_from_observation
        )
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "HalfCheetah-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
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
        return _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )

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
            raise RuntimeError(
                "HalfCheetah returned invalid termination flags"
            )
        self._done = terminated or truncated
        return Step(
            observation=_observation(
                observation,
                exclude_positions=self._exclude_positions,
            ),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
            metrics=_metrics(info),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 6:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    exclude_positions: bool,
) -> dict[str, PolicyValue]:
    expected_shape = (17,) if exclude_positions else (18,)
    if (
        type(value) is not numpy.ndarray
        or value.shape != expected_shape
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("HalfCheetah returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["front_tip_x_position"] = _number(
            value[0],
            name="front tip x position",
        )
        offset = 1
    for name, item in zip(
        _BODY_FIELDS,
        value[offset:],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    return observation


def _metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("HalfCheetah returned invalid metrics")
    names = (
        "x_position",
        "x_velocity",
        "reward_forward",
        "reward_ctrl",
    )
    if not set(names).issubset(value):
        raise RuntimeError("HalfCheetah omitted public metrics")
    return {
        (
            "reward_control" if name == "reward_ctrl" else name
        ): _number(value[name], name=name.replace("_", " "))
        for name in names
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"HalfCheetah returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"HalfCheetah returned a non-finite {name}")
    return number


__all__ = ["HalfCheetahEnvironment"]
