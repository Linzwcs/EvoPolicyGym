"""One fresh strict DeepMind Control Suite task instance per Episode."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
from dm_control import suite  # type: ignore[import-untyped]
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import DmControlConfig, ObservationField
from .video import (
    VIDEO_CAPTURE_FAILED_METRIC,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_METRIC,
    VIDEO_FRAME_SHAPE,
    VIDEO_FRAME_WIDTH,
    VIDEO_INITIAL_FRAME_METRIC,
    video_capture_interval,
)


class _ActionSpec(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype[np.float64]
    minimum: object
    maximum: object


class _TimeStep(Protocol):
    observation: Mapping[str, object]
    reward: object
    discount: object

    def last(self) -> bool: ...


class _UpstreamEnvironment(Protocol):
    physics: _Physics

    def action_spec(self) -> _ActionSpec: ...

    def reset(self) -> _TimeStep: ...

    def step(
        self,
        action: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> _TimeStep: ...

    def close(self) -> None: ...


class _Physics(Protocol):
    model: _WrappedModel
    data: _WrappedData


class _WrappedModel(Protocol):
    _model: mujoco.MjModel


class _WrappedData(Protocol):
    _data: mujoco.MjData


class DmControlEnvironment:
    """Seeded state-observation adapter around one Control Suite task."""

    def __init__(self, episode: EpisodeSpec, *, config: DmControlConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not DmControlConfig:
            raise TypeError("config must be DmControlConfig")
        if episode.scenario is not None:
            raise ValueError("dm_control does not accept Episode scenarios")

        upstream_seed = episode.environment_seed & 0xFFFF_FFFF
        self._config = config
        self._environment = cast(
            _UpstreamEnvironment,
            suite.load(
                config.domain,
                config.task,
                task_kwargs={"random": np.random.RandomState(upstream_seed)},
            ),
        )
        action_spec = self._environment.action_spec()
        if not _valid_action_spec(action_spec, size=config.action_size):
            self._environment.close()
            raise RuntimeError("dm_control action specification drifted")

        self._closed = False
        self._reset = False
        self._done = False
        self._step_count = 0
        self._return = 0.0
        self._best_reward = -math.inf
        self._action_l2_sum = 0.0
        self._saturated_action_component_count = 0
        self._zero_action_count = 0
        self._state_motion_sum = 0.0
        self._no_state_change_count = 0
        self._previous_state: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        self._video_capture_failed = False
        self._video_capture_interval = video_capture_interval(config.max_episode_steps)
        self._video_renderer: mujoco.Renderer | None = None

    def reset(self) -> PolicyValue:
        self._require_open()
        if self._reset:
            raise RuntimeError("dm_control Environment may be reset only once")
        time_step = self._environment.reset()
        if time_step.reward is not None or time_step.discount is not None:
            raise RuntimeError("dm_control returned an invalid initial TimeStep")
        observation, state = _observation(
            time_step.observation,
            fields=self._config.observation_fields,
        )
        self._previous_state = state
        self._reset = True
        return observation

    def step(self, action: PolicyValue) -> Step:
        self._require_open()
        if not self._reset:
            raise RuntimeError("dm_control Environment must be reset before step")
        if self._done:
            raise RuntimeError("dm_control Environment cannot step after termination")
        control = _action(action, size=self._config.action_size)
        initial_video_frame = self._capture_video_frame() if self._step_count == 0 else None
        time_step = self._environment.step(control)
        reward = _finite_number(time_step.reward, name="reward")
        discount = _finite_number(time_step.discount, name="discount")
        if not 0.0 <= discount <= 1.0:
            raise RuntimeError("dm_control returned an invalid discount")
        upstream_last = time_step.last()
        if type(upstream_last) is not bool:
            raise RuntimeError("dm_control returned an invalid step type")
        observation, state = _observation(
            time_step.observation,
            fields=self._config.observation_fields,
        )

        self._step_count += 1
        self._return += reward
        self._best_reward = max(self._best_reward, reward)
        action_l2_norm = float(np.linalg.norm(control))
        saturated_components = int(np.count_nonzero(np.abs(control) == 1.0))
        self._action_l2_sum += action_l2_norm
        self._saturated_action_component_count += saturated_components
        if not np.any(control):
            self._zero_action_count += 1

        previous_state = self._previous_state
        if previous_state is None:
            raise RuntimeError("dm_control previous observation is unavailable")
        state_motion_l2 = float(np.linalg.norm(state - previous_state))
        self._state_motion_sum += state_motion_l2
        if state_motion_l2 == 0.0:
            self._no_state_change_count += 1
        self._previous_state = state

        configured_limit = self._step_count >= self._config.max_episode_steps
        terminated = upstream_last and discount == 0.0
        truncated = not terminated and (upstream_last or configured_limit)
        self._done = terminated or truncated
        video_frame = None
        if (
            self._step_count == 1
            or self._step_count % self._video_capture_interval == 0
            or self._done
        ):
            video_frame = self._capture_video_frame()
        metrics: dict[str, PolicyValue] = {
            "step_count": self._step_count,
            "discount": discount,
            "reward": reward,
            "return": self._return,
            "mean_reward": self._return / self._step_count,
            "best_reward": self._best_reward,
            "action_l2_norm": action_l2_norm,
            "mean_action_l2_norm": self._action_l2_sum / self._step_count,
            "saturated_action_component_count": saturated_components,
            "cumulative_saturated_action_component_count": (
                self._saturated_action_component_count
            ),
            "zero_action_count": self._zero_action_count,
            "state_motion_l2": state_motion_l2,
            "mean_state_motion_l2": self._state_motion_sum / self._step_count,
            "no_state_change_count": self._no_state_change_count,
            "upstream_last": upstream_last,
            VIDEO_CAPTURE_FAILED_METRIC: self._video_capture_failed,
        }
        if initial_video_frame is not None:
            metrics[VIDEO_INITIAL_FRAME_METRIC] = initial_video_frame
        if video_frame is not None:
            metrics[VIDEO_FRAME_METRIC] = video_frame
        return Step(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close_video_renderer()
            self._environment.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("dm_control Environment is closed")

    def _capture_video_frame(self) -> TensorValue | None:
        if self._video_capture_failed:
            return None
        try:
            if self._video_renderer is None:
                self._video_renderer = mujoco.Renderer(
                    self._environment.physics.model._model,
                    height=VIDEO_FRAME_HEIGHT,
                    width=VIDEO_FRAME_WIDTH,
                )
            self._video_renderer.update_scene(
                self._environment.physics.data._data,
                camera=0,
            )
            raw = self._video_renderer.render()
            frame = np.asarray(raw)
            if frame.shape != VIDEO_FRAME_SHAPE or frame.dtype != np.dtype(np.uint8):
                raise RuntimeError("dm_control camera frame shape or dtype drifted")
            contiguous = np.ascontiguousarray(frame)
            return TensorValue(
                dtype="uint8",
                shape=VIDEO_FRAME_SHAPE,
                data=contiguous.tobytes(order="C"),
            )
        except Exception:
            self._video_capture_failed = True
            self._close_video_renderer()
            return None

    def _close_video_renderer(self) -> None:
        renderer = self._video_renderer
        self._video_renderer = None
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception:
            self._video_capture_failed = True


def _valid_action_spec(specification: _ActionSpec, *, size: int) -> bool:
    try:
        minimum = np.asarray(specification.minimum, dtype=np.float64)
        maximum = np.asarray(specification.maximum, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        specification.shape == (size,)
        and specification.dtype == np.dtype(np.float64)
        and minimum.shape == (size,)
        and maximum.shape == (size,)
        and np.all(minimum == -1.0)
        and np.all(maximum == 1.0)
    )


def _action(
    value: PolicyValue,
    *,
    size: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if type(value) is not list or len(value) != size:
        raise InvalidAction()
    if any(type(component) is not float for component in value):
        raise InvalidAction()
    control = np.asarray(value, dtype=np.float64)
    if not np.isfinite(control).all() or np.any(control < -1.0) or np.any(control > 1.0):
        raise InvalidAction()
    return control


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise RuntimeError(f"dm_control returned an invalid {name}")
    try:
        number = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"dm_control returned an invalid {name}") from error
    if not math.isfinite(number):
        raise RuntimeError(f"dm_control returned a non-finite {name}")
    return number


def _observation(
    raw: Mapping[str, object],
    *,
    fields: tuple[ObservationField, ...],
) -> tuple[
    dict[str, PolicyValue],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    expected_names = tuple(name for name, _shape in fields)
    if set(raw) != set(expected_names):
        raise RuntimeError("dm_control observation fields drifted")
    observation: dict[str, PolicyValue] = {}
    flattened: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for name, expected_shape in fields:
        try:
            value = np.asarray(raw[name], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"dm_control returned invalid {name}") from error
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise RuntimeError(f"dm_control returned invalid {name}")
        contiguous = np.ascontiguousarray(value, dtype="<f8")
        observation[name] = TensorValue(
            dtype="float64",
            shape=expected_shape,
            data=contiguous.tobytes(order="C"),
        )
        flattened.append(contiguous.reshape(-1))
    return observation, np.concatenate(flattened)


__all__ = ["DmControlEnvironment"]
