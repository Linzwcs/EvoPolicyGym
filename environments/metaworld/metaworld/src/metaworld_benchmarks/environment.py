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
        self._task_name = task_name
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
        self._previous_state: NDArray[numpy.float64] | None = None
        self._previous_upstream: dict[str, PolicyValue] | None = None
        self._success_ever = False
        self._first_success_step = -1
        self._successful_steps = 0
        self._success_lost_count = 0
        self._previous_success = False
        self._grasp_success_ever = False
        self._first_grasp_success_step = -1
        self._grasp_success_lost_count = 0
        self._previous_grasp_success = False
        self._best_reward = -math.inf
        self._best_near_object = -math.inf
        self._best_grasp_reward = -math.inf
        self._best_in_place_reward = -math.inf
        self._best_obj_to_target = math.inf
        self._cumulative_return = 0.0
        self._cumulative_action_l2 = 0.0
        self._saturated_action_components = 0
        self._zero_action_count = 0
        self._cumulative_state_motion = 0.0
        self._no_state_change_count = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset()
        public = self._observation(observation)
        self._previous_state = _state_array(public)
        self._started = True
        return public

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        previous_state = self._previous_state
        if previous_state is None:
            raise RuntimeError("MetaWorld state history is unavailable")
        control = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(control)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MetaWorld returned invalid termination flags")
        self._steps += 1
        truncated = bool(truncated or (self._steps == _MAX_EPISODE_STEPS and not terminated))
        public = self._observation(observation)
        current_state = _state_array(public)
        number = _number(reward, name="reward")
        upstream = _upstream_metrics(info)
        if not math.isclose(
            number,
            _required_float(upstream, "unscaled_reward"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("MetaWorld reward semantics drifted")
        previous_upstream = self._previous_upstream
        success = _required_bool(upstream, "success")
        grasp_success = _required_bool(upstream, "grasp_success")
        success_first_reached = success and not self._success_ever
        success_lost = self._previous_success and not success
        if success_first_reached:
            self._first_success_step = self._steps
        self._success_ever = self._success_ever or success
        self._successful_steps += int(success)
        self._success_lost_count += int(success_lost)
        self._previous_success = success

        grasp_success_first_reached = grasp_success and not self._grasp_success_ever
        grasp_success_lost = self._previous_grasp_success and not grasp_success
        if grasp_success_first_reached:
            self._first_grasp_success_step = self._steps
        self._grasp_success_ever = self._grasp_success_ever or grasp_success
        self._grasp_success_lost_count += int(grasp_success_lost)
        self._previous_grasp_success = grasp_success

        near_object = _required_float(upstream, "near_object")
        grasp_reward = _required_float(upstream, "grasp_reward")
        in_place_reward = _required_float(upstream, "in_place_reward")
        obj_to_target = _required_float(upstream, "obj_to_target")
        self._best_reward = max(self._best_reward, number)
        self._best_near_object = max(self._best_near_object, near_object)
        self._best_grasp_reward = max(self._best_grasp_reward, grasp_reward)
        self._best_in_place_reward = max(
            self._best_in_place_reward,
            in_place_reward,
        )
        self._best_obj_to_target = min(
            self._best_obj_to_target,
            obj_to_target,
        )
        self._cumulative_return += number

        action_l2 = float(numpy.linalg.norm(control))
        action_max_abs = float(numpy.max(numpy.abs(control)))
        saturated_components = int(numpy.count_nonzero(numpy.abs(control) >= 1.0))
        zero_action = action_max_abs == 0.0
        self._cumulative_action_l2 += action_l2
        self._saturated_action_components += saturated_components
        self._zero_action_count += int(zero_action)
        state_motion = float(numpy.linalg.norm(current_state - previous_state))
        no_state_change = state_motion <= 1e-12
        self._cumulative_state_motion += state_motion
        self._no_state_change_count += int(no_state_change)

        terminal_reason = (
            "success_termination"
            if terminated and success
            else "upstream_termination"
            if terminated
            else "time_limit"
            if truncated
            else "in_progress"
        )
        self._done = terminated or truncated
        self._previous_state = current_state
        self._previous_upstream = upstream
        metrics: dict[str, PolicyValue] = {
            **upstream,
            "task_name": self._task_name,
            "task_index": self._task_index,
            "step_count": self._steps,
            "remaining_steps": max(_MAX_EPISODE_STEPS - self._steps, 0),
            "success_ever": self._success_ever,
            "success_first_reached_this_step": success_first_reached,
            "first_success_step": self._first_success_step,
            "successful_step_count": self._successful_steps,
            "successful_step_fraction": self._successful_steps / self._steps,
            "success_lost_this_step": success_lost,
            "success_lost_count": self._success_lost_count,
            "grasp_success_ever": self._grasp_success_ever,
            "grasp_success_first_reached_this_step": (grasp_success_first_reached),
            "first_grasp_success_step": self._first_grasp_success_step,
            "grasp_success_lost_this_step": grasp_success_lost,
            "grasp_success_lost_count": self._grasp_success_lost_count,
            "reward_improvement_this_step": _increase(
                previous_upstream,
                "unscaled_reward",
                number,
            ),
            "near_object_improvement_this_step": _increase(
                previous_upstream,
                "near_object",
                near_object,
            ),
            "grasp_reward_improvement_this_step": _increase(
                previous_upstream,
                "grasp_reward",
                grasp_reward,
            ),
            "in_place_reward_improvement_this_step": _increase(
                previous_upstream,
                "in_place_reward",
                in_place_reward,
            ),
            "obj_to_target_improvement_this_step": _decrease(
                previous_upstream,
                "obj_to_target",
                obj_to_target,
            ),
            "best_reward": self._best_reward,
            "best_near_object": self._best_near_object,
            "best_grasp_reward": self._best_grasp_reward,
            "best_in_place_reward": self._best_in_place_reward,
            "best_obj_to_target": self._best_obj_to_target,
            "cumulative_return": self._cumulative_return,
            "action_l2_norm": action_l2,
            "action_max_abs": action_max_abs,
            "saturated_action_component_count": saturated_components,
            "saturated_action_component_fraction": (saturated_components / _ACTION_SHAPE[0]),
            "zero_action": zero_action,
            "mean_action_l2_norm": self._cumulative_action_l2 / self._steps,
            "cumulative_saturated_action_component_count": (self._saturated_action_components),
            "zero_action_count": self._zero_action_count,
            "state_motion_l2": state_motion,
            "no_state_change": no_state_change,
            "mean_state_motion_l2": self._cumulative_state_motion / self._steps,
            "no_state_change_count": self._no_state_change_count,
            "task_stage": _task_stage(
                success=success,
                success_ever=self._success_ever,
                grasp_success=grasp_success,
                truncated=truncated,
            ),
            "terminal_reason": terminal_reason,
        }
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
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
            1 if index == self._task_index else 0 for index in range(len(self._config.task_names))
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
        raise ValueError("multi-task Episode scenario must contain only task_index")
    index = value["task_index"]
    if type(index) is not int or not 0 <= index < task_count:
        raise ValueError("Episode task_index is out of range")
    return index


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != _ACTION_SHAPE[0]:
        raise InvalidAction()
    items: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -1.0 <= item <= 1.0:
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


def _upstream_metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("MetaWorld returned invalid metrics")
    if "success" not in value:
        raise RuntimeError("MetaWorld omitted success")
    required = {"success", "grasp_success", *_PUBLIC_NUMERIC_METRICS}
    if not required <= set(value):
        raise RuntimeError("MetaWorld omitted public reward diagnostics")
    metrics: dict[str, PolicyValue] = {
        "success": _flag(value["success"], name="success"),
        "grasp_success": _flag(
            value["grasp_success"],
            name="grasp_success",
        ),
    }
    for name in _PUBLIC_NUMERIC_METRICS:
        if name in value:
            metrics[name] = _number(value[name], name=name)
    return metrics


def _state_array(value: PolicyValue) -> NDArray[numpy.float64]:
    state = value["state"] if type(value) is dict else value
    if type(state) is not TensorValue:
        raise RuntimeError("MetaWorld public state is invalid")
    return numpy.frombuffer(state.data, dtype="<f8").copy()


def _required_bool(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics[name]
    if type(value) is not bool:
        raise RuntimeError(f"MetaWorld {name} metric is invalid")
    return value


def _required_float(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    if type(value) is not float:
        raise RuntimeError(f"MetaWorld {name} metric is invalid")
    return value


def _increase(
    previous: dict[str, PolicyValue] | None,
    name: str,
    current: float,
) -> float | None:
    return None if previous is None else current - _required_float(previous, name)


def _decrease(
    previous: dict[str, PolicyValue] | None,
    name: str,
    current: float,
) -> float | None:
    return None if previous is None else _required_float(previous, name) - current


def _task_stage(
    *,
    success: bool,
    success_ever: bool,
    grasp_success: bool,
    truncated: bool,
) -> str:
    if success:
        return "success"
    if truncated:
        return "time_limit"
    if success_ever:
        return "success_lost"
    if grasp_success:
        return "grasp_or_contact_achieved"
    return "reaching_and_positioning"


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
