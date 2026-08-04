"""One fresh strict robosuite task instance per Episode."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol, cast

import numpy as np
import robosuite  # type: ignore[import-untyped]
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import RobosuiteConfig


class _UpstreamEnvironment(Protocol):
    action_dim: int

    @property
    def action_spec(self) -> tuple[object, object]: ...

    def reset(self) -> Mapping[str, object]: ...

    def step(
        self,
        action: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[Mapping[str, object], object, object, object]: ...

    def _check_success(self) -> object: ...

    def close(self) -> None: ...


class RobosuiteEnvironment:
    """Seeded Panda/BASIC-controller adapter around one manipulation task."""

    def __init__(self, episode: EpisodeSpec, *, config: RobosuiteConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not RobosuiteConfig:
            raise TypeError("config must be RobosuiteConfig")
        if episode.scenario is not None:
            raise ValueError("robosuite does not accept Episode scenarios")

        robots: str | list[str]
        environment_configuration: str
        if config.robot_count == 1:
            robots = "Panda"
            environment_configuration = "default"
        else:
            robots = ["Panda", "Panda"]
            environment_configuration = "opposed"

        self._config = config
        self._environment = cast(
            _UpstreamEnvironment,
            robosuite.make(
                config.environment_id,
                robots=robots,
                env_configuration=environment_configuration,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=True,
                reward_shaping=True,
                reward_scale=1.0,
                horizon=config.max_episode_steps,
                ignore_done=False,
                hard_reset=False,
                seed=episode.environment_seed,
            ),
        )
        if self._environment.action_dim != config.action_size:
            self._environment.close()
            raise RuntimeError("robosuite action dimension drifted")
        low, high = self._environment.action_spec
        if not _valid_action_bounds(low, high, size=config.action_size):
            self._environment.close()
            raise RuntimeError("robosuite action bounds drifted")

        self._closed = False
        self._reset = False
        self._done = False
        self._step_count = 0
        self._success_ever = False
        self._first_success_step = -1
        self._success_lost_count = 0
        self._best_reward = -math.inf
        self._action_l2_sum = 0.0
        self._saturated_action_component_count = 0
        self._zero_action_count = 0
        self._proprio_motion_sum = 0.0
        self._object_motion_sum = 0.0
        self._no_state_change_count = 0
        self._previous_proprioception: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        self._previous_objects: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None

    def reset(self) -> PolicyValue:
        self._require_open()
        if self._reset:
            raise RuntimeError("robosuite Environment may be reset only once")
        raw = self._environment.reset()
        observation, proprioception, objects = _observation(raw, config=self._config)
        self._previous_proprioception = proprioception
        self._previous_objects = objects
        self._reset = True
        return observation

    def step(self, action: PolicyValue) -> Step:
        self._require_open()
        if not self._reset:
            raise RuntimeError("robosuite Environment must be reset before step")
        if self._done:
            raise RuntimeError("robosuite Environment cannot step after termination")
        control = _action(action, size=self._config.action_size)
        raw_observation, raw_reward, raw_done, raw_info = self._environment.step(control)
        if type(raw_done) is not bool:
            raise RuntimeError("robosuite returned an invalid done flag")
        if not isinstance(raw_info, Mapping):
            raise RuntimeError("robosuite returned invalid info")
        if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
            raise RuntimeError("robosuite returned an invalid reward")
        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise RuntimeError("robosuite returned a non-finite reward")
        observation, proprioception, objects = _observation(
            raw_observation,
            config=self._config,
        )

        success = bool(self._environment._check_success())
        success_before = self._success_ever
        self._step_count += 1
        if success and not self._success_ever:
            self._success_ever = True
            self._first_success_step = self._step_count
        if success_before and not success:
            self._success_lost_count += 1
        self._best_reward = max(self._best_reward, reward)

        action_l2_norm = float(np.linalg.norm(control))
        saturated_components = int(np.count_nonzero(np.abs(control) == 1.0))
        self._action_l2_sum += action_l2_norm
        self._saturated_action_component_count += saturated_components
        if not np.any(control):
            self._zero_action_count += 1

        previous_proprioception = self._previous_proprioception
        previous_objects = self._previous_objects
        if previous_proprioception is None or previous_objects is None:
            raise RuntimeError("robosuite previous observation is unavailable")
        proprioception_motion_l2 = float(
            np.linalg.norm(proprioception - previous_proprioception)
        )
        object_motion_l2 = float(np.linalg.norm(objects - previous_objects))
        self._proprio_motion_sum += proprioception_motion_l2
        self._object_motion_sum += object_motion_l2
        if proprioception_motion_l2 == 0.0 and object_motion_l2 == 0.0:
            self._no_state_change_count += 1
        self._previous_proprioception = proprioception
        self._previous_objects = objects

        truncated = raw_done and self._step_count >= self._config.max_episode_steps
        terminated = raw_done and not truncated
        self._done = raw_done
        return Step(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "step_count": self._step_count,
                "success": success,
                "success_ever": self._success_ever,
                "success_first_reached_this_step": success and not success_before,
                "first_success_step": self._first_success_step,
                "success_lost_this_step": success_before and not success,
                "success_lost_count": self._success_lost_count,
                "dense_reward": reward,
                "best_dense_reward": self._best_reward,
                "action_l2_norm": action_l2_norm,
                "mean_action_l2_norm": self._action_l2_sum / self._step_count,
                "saturated_action_component_count": saturated_components,
                "cumulative_saturated_action_component_count": (
                    self._saturated_action_component_count
                ),
                "zero_action_count": self._zero_action_count,
                "proprioception_motion_l2": proprioception_motion_l2,
                "object_motion_l2": object_motion_l2,
                "mean_proprioception_motion_l2": (
                    self._proprio_motion_sum / self._step_count
                ),
                "mean_object_motion_l2": self._object_motion_sum / self._step_count,
                "no_state_change_count": self._no_state_change_count,
            },
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._environment.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("robosuite Environment is closed")


def _valid_action_bounds(low: object, high: object, *, size: int) -> bool:
    try:
        minimum = np.asarray(low, dtype=np.float64)
        maximum = np.asarray(high, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        minimum.shape == (size,)
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


def _observation(
    raw: Mapping[str, object],
    *,
    config: RobosuiteConfig,
) -> tuple[
    dict[str, PolicyValue],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    proprioception_parts = tuple(
        _vector(raw, f"robot{index}_proprio-state")
        for index in range(config.robot_count)
    )
    proprioception = np.concatenate(proprioception_parts)
    objects = _vector(raw, "object-state")
    if proprioception.shape != (config.proprioception_size,):
        raise RuntimeError("robosuite proprioception shape drifted")
    if objects.shape != (config.object_state_size,):
        raise RuntimeError("robosuite object-state shape drifted")
    return (
        {
            "proprioception": _tensor(proprioception),
            "objects": _tensor(objects),
        },
        proprioception,
        objects,
    )


def _vector(
    raw: Mapping[str, object],
    name: str,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if name not in raw:
        raise RuntimeError(f"robosuite omitted {name}")
    try:
        value = np.asarray(raw[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"robosuite returned invalid {name}") from error
    if value.ndim != 1 or not np.isfinite(value).all():
        raise RuntimeError(f"robosuite returned invalid {name}")
    return np.ascontiguousarray(value, dtype="<f8")


def _tensor(value: np.ndarray[tuple[int], np.dtype[np.float64]]) -> TensorValue:
    return TensorValue(
        dtype="float64",
        shape=(value.size,),
        data=value.tobytes(order="C"),
    )


__all__ = ["RobosuiteEnvironment"]
