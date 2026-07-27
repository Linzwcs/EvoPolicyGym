"""One fresh strict ViZDoom scenario per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
import vizdoom.gymnasium_wrapper  # noqa: F401  # Registers scenarios.
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import ViZDoomConfig

_SCREEN_SHAPE = (240, 320, 3)
_AUDIO_SHAPE = (1260, 2)
_CONTINUOUS_LIMIT = float(numpy.finfo(numpy.float32).max)


class ViZDoomEnvironment:
    """Seeded adapter around one bundled ViZDoom scenario."""

    def __init__(self, episode: EpisodeSpec, *, config: ViZDoomConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not ViZDoomConfig:
            raise TypeError("config must be ViZDoomConfig")
        if episode.scenario is not None:
            raise ValueError(
                "ViZDoom configuration belongs in ViZDoomConfig"
            )
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(config.environment_id),
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
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation, config=self._config)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        applied = (
            _hybrid_action(action, size=self._config.action_size)
            if self._config.hybrid_action
            else _discrete_action(action, size=self._config.action_size)
        )
        observation, reward, terminated, truncated, _ = (
            self._environment.step(applied)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("ViZDoom returned invalid termination flags")
        self._steps += 1
        truncated = bool(
            truncated
            or (
                self._steps == self._config.max_episode_steps
                and not terminated
            )
        )
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation, config=self._config),
            reward=_number(reward),
            terminated=terminated,
            truncated=truncated,
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


def _hybrid_action(value: PolicyValue, *, size: int) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"binary", "continuous"}:
        raise InvalidAction()
    binary = value["binary"]
    continuous = value["continuous"]
    if type(binary) is not int or not 0 <= binary < size:
        raise InvalidAction()
    if type(continuous) is not list or len(continuous) != 3:
        raise InvalidAction()
    controls: list[float] = []
    for item in continuous:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -_CONTINUOUS_LIMIT <= item <= _CONTINUOUS_LIMIT
        ):
            raise InvalidAction()
        controls.append(item)
    return {
        "binary": binary,
        "continuous": numpy.asarray(controls, dtype=numpy.float32),
    }


def _observation(
    value: object,
    *,
    config: ViZDoomConfig,
) -> dict[str, PolicyValue]:
    expected = {"screen"}
    if config.audio:
        expected.add("audio")
    if config.notifications:
        expected.add("notifications")
    if config.game_variables:
        expected.add("gamevariables")
    if type(value) is not dict or set(value) != expected:
        raise RuntimeError("ViZDoom returned an invalid observation")
    public: dict[str, PolicyValue] = {
        "screen": _tensor(
            value["screen"],
            dtype="uint8",
            shape=_SCREEN_SHAPE,
            name="screen",
        )
    }
    if config.audio:
        public["audio"] = _tensor(
            value["audio"],
            dtype="int16",
            shape=_AUDIO_SHAPE,
            name="audio",
        )
    if config.notifications:
        notifications = value["notifications"]
        if type(notifications) is not str:
            raise RuntimeError("ViZDoom returned invalid notifications")
        public["notifications"] = notifications
    if config.game_variables:
        public["gamevariables"] = _tensor(
            value["gamevariables"],
            dtype="float32",
            shape=(config.game_variables,),
            name="gamevariables",
        )
    return public


def _tensor(
    value: object,
    *,
    dtype: str,
    shape: tuple[int, ...],
    name: str,
) -> TensorValue:
    if (
        type(value) is not numpy.ndarray
        or value.dtype != numpy.dtype(dtype)
        or value.shape != shape
        or (
            numpy.issubdtype(value.dtype, numpy.floating)
            and not numpy.isfinite(value).all()
        )
    ):
        raise RuntimeError(f"ViZDoom returned invalid {name}")
    return TensorValue(
        dtype=dtype,
        shape=shape,
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("ViZDoom returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("ViZDoom returned non-finite reward")
    return number


__all__ = ["ViZDoomEnvironment"]
