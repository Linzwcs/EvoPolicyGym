"""One fresh Gymnasium Reacher-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import ReacherConfig

_OBSERVATION_NAMES = (
    "joint0_cos",
    "joint1_cos",
    "joint0_sin",
    "joint1_sin",
    "target_x",
    "target_y",
    "joint0_angular_velocity",
    "joint1_angular_velocity",
    "fingertip_target_x",
    "fingertip_target_y",
)


class ReacherEnvironment:
    """The seeded strict adapter around configured Reacher-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: ReacherConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not ReacherConfig:
            raise TypeError("config must be ReacherConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Reacher configuration belongs in ReacherConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Reacher-v5",
                frame_skip=config.frame_skip,
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
            raise RuntimeError("Reacher returned invalid termination flags")
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
    if type(value) is not list or len(value) != 2:
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


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (10,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Reacher returned an invalid observation")
    return {
        name: _number(item, name=name)
        for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("Reacher returned invalid reward metrics")
    if "reward_dist" not in value or "reward_ctrl" not in value:
        raise RuntimeError("Reacher omitted reward metrics")
    return {
        "reward_distance": _number(
            value["reward_dist"],
            name="reward distance",
        ),
        "reward_control": _number(
            value["reward_ctrl"],
            name="reward control",
        ),
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Reacher returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Reacher returned a non-finite {name}")
    return number


__all__ = ["ReacherEnvironment"]
