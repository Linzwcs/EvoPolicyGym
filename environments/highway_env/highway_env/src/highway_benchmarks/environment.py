"""One fresh strict HighwayEnv instance per Episode."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import highway_env  # type: ignore[import-untyped]  # noqa: F401
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import HighwayConfig

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


class HighwayEnvironment:
    """Seeded adapter around one Host-selected HighwayEnv profile."""

    def __init__(self, episode: EpisodeSpec, *, config: HighwayConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HighwayConfig:
            raise TypeError("config must be HighwayConfig")
        if episode.scenario is not None:
            raise ValueError("HighwayEnv configuration belongs in HighwayConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(config.environment_id),
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
        return _policy_value(observation, name="observation")

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        applied = (
            _continuous_action(action, size=self._config.action_size)
            if self._config.continuous
            else _discrete_action(action, size=self._config.action_size)
        )
        observation, reward, terminated, truncated, info = (
            self._environment.step(applied)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("HighwayEnv returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_policy_value(observation, name="observation"),
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


def _discrete_action(value: PolicyValue, *, size: int) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise InvalidAction()
    return value


def _continuous_action(
    value: PolicyValue,
    *,
    size: int,
) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != size:
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


def _policy_value(value: object, *, name: str) -> PolicyValue:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        if type(value) is float and not math.isfinite(value):
            raise RuntimeError(f"HighwayEnv returned non-finite {name}")
        return cast(PolicyValue, value)
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        try:
            return int(cast(SupportsIndex, value).__index__())
        except (OverflowError, ValueError) as error:
            raise RuntimeError(
                f"HighwayEnv returned an out-of-range {name}"
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
                    f"HighwayEnv returned a non-string {name} key"
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
        f"HighwayEnv returned unsupported {name} carrier "
        f"{type(value).__name__}"
    )


def _tensor(value: object, *, name: str) -> TensorValue:
    if type(value) is not numpy.ndarray:
        raise RuntimeError(f"HighwayEnv returned an invalid {name} tensor")
    dtype = value.dtype.name
    if dtype not in _TENSOR_DTYPES:
        raise RuntimeError(
            f"HighwayEnv returned unsupported {name} dtype {dtype}"
        )
    if numpy.issubdtype(value.dtype, numpy.floating) and not numpy.isfinite(
        value
    ).all():
        raise RuntimeError(f"HighwayEnv returned non-finite {name}")
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


def _metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("HighwayEnv returned invalid metrics")
    metrics: dict[str, PolicyValue] = {}
    for name in ("crashed", "is_success"):
        if name in value:
            flag = value[name]
            if type(flag) is bool:
                metrics[name] = flag
            elif isinstance(flag, numpy.bool_):
                metrics[name] = bool(flag)
            else:
                raise RuntimeError(f"HighwayEnv returned invalid {name}")
    if "speed" in value:
        metrics["speed"] = _number(value["speed"], name="speed")
    return metrics


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"HighwayEnv returned invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"HighwayEnv returned non-finite {name}")
    return number


__all__ = ["HighwayEnvironment"]
