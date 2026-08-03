"""One fresh strict functional Jumanji environment per Episode."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping
from typing import Protocol, SupportsFloat, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jumanji
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import JumanjiConfig

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
_MISSING = object()


class _FunctionalEnvironment(Protocol):
    @property
    def action_spec(self) -> object: ...

    def reset(self, key: object) -> tuple[object, object]: ...

    def step(self, state: object, action: object) -> tuple[object, object]: ...

    def close(self) -> None: ...


class JumanjiEnvironment:
    """Seeded adapter around one Host-selected Jumanji profile."""

    def __init__(self, episode: EpisodeSpec, *, config: JumanjiConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not JumanjiConfig:
            raise TypeError("config must be JumanjiConfig")
        if episode.scenario is not None:
            raise ValueError("Jumanji configuration belongs in JumanjiConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            _FunctionalEnvironment,
            jumanji.make(config.environment_id),  # type: ignore[attr-defined]
        )
        self._reset_environment: Callable[[object], tuple[object, object]] = jax.jit(
            self._environment.reset
        )
        self._step_environment: Callable[
            [object, object], tuple[object, object]
        ] = jax.jit(self._environment.step)
        try:
            _check_action_spec(self._environment.action_spec, config=config)
        except Exception:
            self._environment.close()
            raise
        self._state: object = _MISSING
        self._raw_observation: object = _MISSING
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        key = jax.random.PRNGKey(self._seed & 0xFFFF_FFFF)
        key = jax.random.fold_in(key, (self._seed >> 32) & 0xFFFF_FFFF)
        state, timestep = self._reset_environment(key)
        observation = _timestep_observation(timestep)
        self._state = state
        self._raw_observation = observation
        self._started = True
        return _policy_observation(
            observation,
            state=state,
            config=self._config,
        )

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        applied = _action(action, config=self._config)
        _check_action_mask(
            self._raw_observation,
            applied,
            config=self._config,
        )
        state, timestep = self._step_environment(self._state, applied)
        observation = _timestep_observation(timestep)
        upstream_done = _timestep_last(timestep)
        self._steps += 1
        no_legal_actions = (
            not upstream_done
            and self._config.has_action_mask
            and not _has_legal_action(observation, config=self._config)
        )
        terminated = upstream_done or no_legal_actions
        truncated = (
            self._steps == self._config.max_episode_steps and not terminated
        )
        reward = _number(getattr(timestep, "reward", _MISSING), name="reward")
        metrics = _metrics(getattr(timestep, "extras", _MISSING))
        if no_legal_actions:
            metrics["no_legal_actions"] = True
        if self._config.profile == "tetris":
            metrics["lines_cleared"] = _tetris_lines_cleared(reward)
        if self._config.profile == "pacman" and upstream_done:
            metrics["terminal_reason"] = _pacman_terminal_reason(
                state,
                time_limit=self._config.max_episode_steps,
            )
        if self._config.profile.startswith("rubiks-cube") and upstream_done:
            metrics["terminal_reason"] = _rubik_terminal_reason(
                observation,
                step_count=self._steps,
                time_limit=self._config.max_episode_steps,
            )
        self._state = state
        self._raw_observation = observation
        self._done = terminated or truncated
        return Step(
            observation=_policy_observation(
                observation,
                state=state,
                config=self._config,
            ),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _check_action_spec(value: object, *, config: JumanjiConfig) -> None:
    expected_type = "DiscreteArray" if config.action_kind == "discrete" else "MultiDiscreteArray"
    if type(value).__name__ != expected_type:
        raise RuntimeError("Jumanji action specification drifted")
    raw_num_values = getattr(value, "num_values", _MISSING)
    try:
        array = numpy.asarray(raw_num_values)
    except (TypeError, ValueError):
        raise RuntimeError("Jumanji returned an invalid action specification") from None
    actual: tuple[int, ...]
    if config.action_kind == "discrete":
        actual = (int(array.item()),) if array.shape == () else ()
    else:
        actual = tuple(int(item) for item in array.tolist()) if array.ndim == 1 else ()
    dtype = numpy.dtype(getattr(value, "dtype", object)).name
    if actual != config.action_num_values or dtype != "int32":
        raise RuntimeError("Jumanji action specification drifted")


def _action(value: PolicyValue, *, config: JumanjiConfig) -> object:
    if config.action_kind == "discrete":
        if type(value) is not int or not 0 <= value < config.action_num_values[0]:
            raise InvalidAction()
        return jnp.asarray(value, dtype=jnp.int32)
    if type(value) is not list or len(value) != len(config.action_num_values):
        raise InvalidAction()
    items: list[int] = []
    for item, size in zip(value, config.action_num_values, strict=True):
        if type(item) is not int or not 0 <= item < size:
            raise InvalidAction()
        items.append(item)
    return jnp.asarray(items, dtype=jnp.int32)


def _check_action_mask(
    observation: object,
    action: object,
    *,
    config: JumanjiConfig,
) -> None:
    if not config.has_action_mask:
        return
    raw_mask = _field(observation, "action_mask")
    try:
        mask = numpy.asarray(raw_mask)
        indices = numpy.asarray(action, dtype=numpy.int64)
    except (TypeError, ValueError):
        raise RuntimeError("Jumanji returned an invalid action mask") from None
    if mask.dtype != numpy.dtype(bool):
        raise RuntimeError("Jumanji returned an invalid action mask")
    if config.action_kind == "discrete":
        expected = (config.action_num_values[0],)
        if mask.shape != expected or indices.shape != ():
            raise RuntimeError("Jumanji returned an invalid action mask")
        valid = bool(mask[int(indices.item())])
    elif mask.shape == config.action_num_values:
        if indices.shape != (len(config.action_num_values),):
            raise RuntimeError("Jumanji returned an invalid action")
        valid = bool(mask[tuple(int(item) for item in indices.tolist())])
    elif (
        len(set(config.action_num_values)) == 1
        and mask.shape == (len(config.action_num_values), config.action_num_values[0])
    ):
        if indices.shape != (len(config.action_num_values),):
            raise RuntimeError("Jumanji returned an invalid action")
        valid = all(bool(mask[index, int(item)]) for index, item in enumerate(indices.tolist()))
    else:
        raise RuntimeError("Jumanji returned an invalid action mask")
    if not valid:
        raise InvalidAction()


def _has_legal_action(observation: object, *, config: JumanjiConfig) -> bool:
    raw_mask = _field(observation, "action_mask")
    try:
        mask = numpy.asarray(raw_mask)
    except (TypeError, ValueError):
        raise RuntimeError("Jumanji returned an invalid action mask") from None
    if mask.dtype != numpy.dtype(bool):
        raise RuntimeError("Jumanji returned an invalid action mask")
    if config.action_kind == "discrete" or mask.shape == config.action_num_values:
        expected = config.action_num_values
        if mask.shape != expected:
            raise RuntimeError("Jumanji returned an invalid action mask")
        return bool(numpy.any(mask))
    expected = (len(config.action_num_values), config.action_num_values[0])
    if mask.shape != expected or len(set(config.action_num_values)) != 1:
        raise RuntimeError("Jumanji returned an invalid action mask")
    return bool(numpy.all(numpy.any(mask, axis=1)))


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise RuntimeError(f"Jumanji omitted observation {name}")
        return value[name]
    field = getattr(value, name, _MISSING)
    if field is _MISSING:
        raise RuntimeError(f"Jumanji omitted observation {name}")
    return field


def _timestep_observation(value: object) -> object:
    observation = getattr(value, "observation", _MISSING)
    if observation is _MISSING:
        raise RuntimeError("Jumanji returned an invalid TimeStep")
    return observation


def _timestep_last(value: object) -> bool:
    last = getattr(value, "last", _MISSING)
    if not callable(last):
        raise RuntimeError("Jumanji returned an invalid TimeStep")
    result = numpy.asarray(last())
    if result.shape != () or result.dtype != numpy.dtype(bool):
        raise RuntimeError("Jumanji returned an invalid termination flag")
    return bool(result.item())


def _policy_observation(
    observation: object,
    *,
    state: object,
    config: JumanjiConfig,
) -> PolicyValue:
    public = _policy_value(observation, name="observation")
    if config.profile != "tetris":
        return public
    if type(public) is not dict:
        raise RuntimeError("Jumanji returned an invalid Tetris observation")
    step_count = _policy_value(_field(state, "step_count"), name="state.step_count")
    if type(step_count) is not int or not 0 <= step_count <= config.max_episode_steps:
        raise RuntimeError("Jumanji returned an invalid Tetris step count")
    # Jumanji 1.1.1 writes a constant zero into Observation.step_count even though
    # State.step_count is maintained correctly. Publish the intended live value.
    public["step_count"] = step_count
    return public


def _tetris_lines_cleared(reward: float) -> int:
    rewards = (0.0, 40.0, 100.0, 300.0, 1_200.0)
    try:
        return rewards.index(reward)
    except ValueError:
        raise RuntimeError("Jumanji returned an unknown Tetris line reward") from None


def _pacman_terminal_reason(state: object, *, time_limit: int) -> str:
    dead = _policy_value(_field(state, "dead"), name="state.dead")
    pellets = _policy_value(_field(state, "pellets"), name="state.pellets")
    step_count = _policy_value(
        _field(state, "step_count"),
        name="state.step_count",
    )
    if type(dead) is not bool or type(pellets) is not int or type(step_count) is not int:
        raise RuntimeError("Jumanji returned invalid PacMan terminal state")
    reasons: list[str] = []
    if dead:
        reasons.append("ghost_collision")
    if pellets == 0:
        reasons.append("all_pellets_collected")
    if step_count >= time_limit:
        reasons.append("time_limit")
    if not reasons:
        raise RuntimeError("Jumanji PacMan terminated without a public reason")
    return "+".join(reasons)


def _rubik_terminal_reason(
    observation: object,
    *,
    step_count: int,
    time_limit: int,
) -> str:
    try:
        cube = numpy.asarray(_field(observation, "cube"))
    except (TypeError, ValueError):
        raise RuntimeError("Jumanji returned invalid Rubik terminal state") from None
    if cube.shape != (6, 3, 3) or not numpy.issubdtype(cube.dtype, numpy.integer):
        raise RuntimeError("Jumanji returned invalid Rubik terminal state")
    reasons: list[str] = []
    if all(bool(numpy.all(face == face.reshape(-1)[0])) for face in cube):
        reasons.append("solved")
    if step_count >= time_limit:
        reasons.append("time_limit")
    if not reasons:
        raise RuntimeError("Jumanji Rubik terminated without a public reason")
    return "+".join(reasons)


def _policy_value(value: object, *, name: str) -> PolicyValue:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        if type(value) is float and not math.isfinite(value):
            raise RuntimeError(f"Jumanji returned non-finite {name}")
        return cast(PolicyValue, value)
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        try:
            return int(cast(SupportsIndex, value).__index__())
        except (OverflowError, ValueError) as error:
            raise RuntimeError(f"Jumanji returned an out-of-range {name}") from error
    if isinstance(value, numpy.floating):
        return _number(value, name=name)
    if isinstance(value, (numpy.ndarray, jax.Array)):
        return _array_value(numpy.asarray(value), name=name)
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        return _policy_value(as_dict(), name=name)
    if isinstance(value, Mapping):
        public: dict[str, PolicyValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeError(f"Jumanji returned a non-string {name} key")
            public[key] = _policy_value(item, name=f"{name}.{key}")
        return public
    if type(value) is list:
        return [_policy_value(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if type(value) is tuple:
        return tuple(
            _policy_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise RuntimeError(f"Jumanji returned unsupported {name} carrier {type(value).__name__}")


def _array_value(value: NDArray[numpy.generic], *, name: str) -> PolicyValue:
    if value.shape == ():
        return _policy_value(value.item(), name=name)
    return _tensor(value, name=name)


def _tensor(value: NDArray[numpy.generic], *, name: str) -> TensorValue:
    dtype = value.dtype.name
    if dtype not in _TENSOR_DTYPES:
        raise RuntimeError(f"Jumanji returned unsupported {name} dtype {dtype}")
    if numpy.issubdtype(value.dtype, numpy.floating) and not numpy.isfinite(value).all():
        raise RuntimeError(f"Jumanji returned non-finite {name}")
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
    if not isinstance(value, Mapping):
        raise RuntimeError("Jumanji returned invalid metrics")
    public: dict[str, PolicyValue] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise RuntimeError("Jumanji returned a non-string metric key")
        metric = _policy_value(item, name=f"metric.{key}")
        if type(metric) not in {bool, int, float, str}:
            raise RuntimeError("Jumanji returned a non-scalar metric")
        public[key] = metric
    return public


def _number(value: object, *, name: str) -> float:
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Jumanji returned invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Jumanji returned non-finite {name}")
    return number


__all__ = ["JumanjiEnvironment"]
