"""One fresh strict Gymnasium-Robotics instance per Episode."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import gymnasium_robotics  # type: ignore[import-untyped]
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from gymnasium.spaces import Box
from numpy.typing import NDArray

from .config import RoboticsConfig

gymnasium.register_envs(gymnasium_robotics)

_TENSOR_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    }
)
_KITCHEN_TASK_COUNT = 7

type RoboticsAction = (
    NDArray[numpy.float32] | NDArray[numpy.float64]
)


class RoboticsEnvironment:
    """Seeded adapter around one Host-selected Robotics profile."""

    def __init__(self, episode: EpisodeSpec, *, config: RoboticsConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not RoboticsConfig:
            raise TypeError("config must be RoboticsConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Robotics configuration belongs in RoboticsConfig"
            )
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, RoboticsAction],
            gymnasium.make(config.environment_id),
        )
        action_space = self._environment.action_space
        expected_dtype = numpy.dtype(config.action_dtype)
        if (
            type(action_space) is not Box
            or action_space.shape != (config.action_size,)
            or action_space.dtype != expected_dtype
            or not numpy.all(action_space.low == -1.0)
            or not numpy.all(action_space.high == 1.0)
        ):
            self._environment.close()
            raise RuntimeError(
                "Gymnasium-Robotics action space changed incompatibly"
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
        public = _policy_value(observation, name="observation")
        _validate_observation(public, config=self._config)
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
            self._environment.step(_action(action, config=self._config))
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "Gymnasium-Robotics returned invalid termination flags"
            )
        public = _policy_value(observation, name="observation")
        _validate_observation(public, config=self._config)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
            metrics=_metrics(info, config=self._config),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(
    value: PolicyValue,
    *,
    config: RoboticsConfig,
) -> RoboticsAction:
    if type(value) is not list or len(value) != config.action_size:
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
    return cast(
        RoboticsAction,
        numpy.asarray(items, dtype=numpy.dtype(config.action_dtype)),
    )


def _validate_observation(
    value: PolicyValue,
    *,
    config: RoboticsConfig,
) -> None:
    if config.goal_size is None:
        _require_tensor(
            value,
            shape=(config.observation_size,),
            name="observation",
        )
        return
    if type(value) is not dict or set(value) != {
        "observation",
        "achieved_goal",
        "desired_goal",
    }:
        raise RuntimeError(
            "Gymnasium-Robotics returned an invalid goal observation"
        )
    _require_tensor(
        value["observation"],
        shape=(config.observation_size,),
        name="observation",
    )
    if config.goal_size == -1:
        _validate_kitchen_goals(value["achieved_goal"], name="achieved_goal")
        _validate_kitchen_goals(value["desired_goal"], name="desired_goal")
        return
    _require_tensor(
        value["achieved_goal"],
        shape=(config.goal_size,),
        name="achieved_goal",
    )
    _require_tensor(
        value["desired_goal"],
        shape=(config.goal_size,),
        name="desired_goal",
    )


def _validate_kitchen_goals(value: PolicyValue, *, name: str) -> None:
    expected = {
        "bottom burner": 2,
        "top burner": 2,
        "light switch": 2,
        "slide cabinet": 1,
        "hinge cabinet": 2,
        "microwave": 1,
        "kettle": 7,
    }
    if type(value) is not dict or set(value) != set(expected):
        raise RuntimeError(f"FrankaKitchen returned invalid {name}")
    for task, size in expected.items():
        _require_tensor(value[task], shape=(size,), name=f"{name}.{task}")


def _require_tensor(
    value: PolicyValue,
    *,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if (
        type(value) is not TensorValue
        or value.dtype != "float64"
        or value.shape != shape
    ):
        raise RuntimeError(
            f"Gymnasium-Robotics returned invalid {name} tensor"
        )


def _policy_value(value: object, *, name: str) -> PolicyValue:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        if type(value) is float and not math.isfinite(value):
            raise RuntimeError(
                f"Gymnasium-Robotics returned non-finite {name}"
            )
        return cast(PolicyValue, value)
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        try:
            return int(cast(SupportsIndex, value).__index__())
        except (OverflowError, ValueError) as error:
            raise RuntimeError(
                f"Gymnasium-Robotics returned out-of-range {name}"
            ) from error
    if isinstance(value, numpy.floating):
        return _number(value, name=name)
    if type(value) is numpy.ndarray:
        return _tensor(value, name=name)
    if isinstance(value, Mapping):
        public: dict[str, PolicyValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeError(
                    "Gymnasium-Robotics returned a non-string observation key"
                )
            public[key] = _policy_value(item, name=f"{name}.{key}")
        return public
    if type(value) is list:
        return [
            _policy_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            _policy_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise RuntimeError(
        f"Gymnasium-Robotics returned unsupported {name} carrier "
        f"{type(value).__name__}"
    )


def _tensor(value: object, *, name: str) -> TensorValue:
    if type(value) is not numpy.ndarray:
        raise RuntimeError(
            f"Gymnasium-Robotics returned invalid {name} tensor"
        )
    dtype = value.dtype.name
    if dtype not in _TENSOR_DTYPES:
        raise RuntimeError(
            f"Gymnasium-Robotics returned unsupported {name} dtype {dtype}"
        )
    if numpy.issubdtype(value.dtype, numpy.floating) and not numpy.isfinite(
        value
    ).all():
        raise RuntimeError(
            f"Gymnasium-Robotics returned non-finite {name}"
        )
    array = numpy.ascontiguousarray(value)
    if array.dtype.itemsize > 1 and (
        array.dtype.byteorder == ">"
        or (array.dtype.byteorder == "=" and sys.byteorder == "big")
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    return TensorValue(
        dtype=dtype,
        shape=tuple(int(size) for size in array.shape),
        data=array.tobytes(order="C"),
    )


def _metrics(
    value: object,
    *,
    config: RoboticsConfig,
) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("Gymnasium-Robotics returned invalid metrics")
    metrics: dict[str, PolicyValue] = {}
    success_name = (
        "is_success" if "is_success" in value else "success"
    )
    if success_name in value:
        metrics["success"] = _flag(
            value[success_name],
            name=success_name,
        )
    if config.family == "franka-kitchen":
        remaining = value.get("tasks_to_complete")
        completed = value.get("episode_task_completions")
        if type(remaining) is not list or type(completed) is not list:
            raise RuntimeError("FrankaKitchen omitted task completion metrics")
        if any(type(item) is not str for item in remaining + completed):
            raise RuntimeError(
                "FrankaKitchen returned invalid task completion metrics"
            )
        completed_count = len(completed)
        if not 0 <= completed_count <= _KITCHEN_TASK_COUNT:
            raise RuntimeError(
                "FrankaKitchen returned invalid completed task count"
            )
        metrics["completed_tasks"] = completed_count
        metrics["task_completion_fraction"] = (
            completed_count / _KITCHEN_TASK_COUNT
        )
        metrics["success"] = len(remaining) == 0
    return metrics


def _flag(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, numpy.bool_):
        return bool(value)
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(
            f"Gymnasium-Robotics returned invalid {name} flag"
        )
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"Gymnasium-Robotics returned non-finite {name}"
        )
    return number


__all__ = ["RoboticsEnvironment"]
