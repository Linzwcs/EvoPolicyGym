"""One fresh strict MetaWorld task instance per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import metaworld  # noqa: F401  # Registers the public MetaWorld entries.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import MetaWorldConfig

_STATE_SHAPE = (39,)
_ACTION_SHAPE = (4,)
_MAX_EPISODE_STEPS = 500
_PUBLIC_NUMERIC_METRICS = (
    "near_object",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
)


class MetaWorldEnvironment:
    """Seeded adapter around one task in a fixed MT collection."""

    def __init__(self, episode: EpisodeSpec, *, config: MetaWorldConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not MetaWorldConfig:
            raise TypeError("config must be MetaWorldConfig")
        self._config = config
        self._task_index = _task_index(
            episode.scenario,
            task_count=len(config.task_names),
        )
        task_name = config.task_names[self._task_index]
        self._environment = cast(
            gymnasium.Env[object, NDArray[numpy.float32]],
            gymnasium.make(
                "Meta-World/MT1",
                env_name=task_name,
                # MetaWorld 3.1.1 still routes this seed through NumPy's
                # legacy RandomState, whose public domain is uint32.
                seed=episode.environment_seed & 0xFFFF_FFFF,
                num_goals=1,
                reward_function_version="v2",
                terminate_on_success=False,
                disable_env_checker=True,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset()
        public = self._observation(observation)
        self._started = True
        return public

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
                "MetaWorld returned invalid termination flags"
            )
        self._steps += 1
        truncated = bool(
            truncated
            or (self._steps == _MAX_EPISODE_STEPS and not terminated)
        )
        self._done = terminated or truncated
        return Step(
            observation=self._observation(observation),
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

    def _observation(self, value: object) -> PolicyValue:
        state = _tensor(value)
        if len(self._config.task_names) == 1:
            return state
        task = bytes(
            1 if index == self._task_index else 0
            for index in range(len(self._config.task_names))
        )
        return {
            "state": state,
            "task": TensorValue(
                dtype="bool",
                shape=(len(self._config.task_names),),
                data=task,
            ),
        }


def _task_index(value: PolicyValue, *, task_count: int) -> int:
    if task_count == 1:
        if value is not None:
            raise ValueError("MT1 Episode scenario must be None")
        return 0
    if type(value) is not dict or set(value) != {"task_index"}:
        raise ValueError(
            "multi-task Episode scenario must contain only task_index"
        )
    index = value["task_index"]
    if type(index) is not int or not 0 <= index < task_count:
        raise ValueError("Episode task_index is out of range")
    return index


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != _ACTION_SHAPE[0]:
        raise InvalidAction()
    items: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
        ):
            raise InvalidAction()
        items.append(item)
    return numpy.asarray(items, dtype=numpy.float32)


def _tensor(value: object) -> TensorValue:
    if (
        type(value) is not numpy.ndarray
        or value.dtype != numpy.dtype("float64")
        or value.shape != _STATE_SHAPE
        or not numpy.isfinite(value).all()
    ):
        raise RuntimeError("MetaWorld returned an invalid observation")
    return TensorValue(
        dtype="float64",
        shape=_STATE_SHAPE,
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


def _metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("MetaWorld returned invalid metrics")
    if "success" not in value:
        raise RuntimeError("MetaWorld omitted success")
    metrics: dict[str, PolicyValue] = {
        "success": _flag(value["success"], name="success")
    }
    if "grasp_success" in value:
        metrics["grasp_success"] = _flag(
            value["grasp_success"],
            name="grasp_success",
        )
    for name in _PUBLIC_NUMERIC_METRICS:
        if name in value:
            metrics[name] = _number(value[name], name=name)
    return metrics


def _flag(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, numpy.bool_):
        return bool(value)
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"MetaWorld returned invalid {name} flag")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MetaWorld returned invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MetaWorld returned non-finite {name}")
    return number


__all__ = ["MetaWorldEnvironment"]
