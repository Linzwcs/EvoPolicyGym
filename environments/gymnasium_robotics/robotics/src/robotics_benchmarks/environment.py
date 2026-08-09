"""One fresh strict Gymnasium-Robotics instance per Episode."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, SupportsIndex, cast

import gymnasium
import gymnasium_robotics  # type: ignore[import-untyped]
import mujoco  # type: ignore[import-untyped]
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from gymnasium.spaces import Box
from gymnasium_robotics.utils import rotations  # type: ignore[import-untyped]
from numpy.typing import NDArray

from .config import RoboticsConfig
from .video import (
    VIDEO_CAPTURE_FAILED_METRIC,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_METRIC,
    VIDEO_FRAME_SHAPE,
    VIDEO_FRAME_WIDTH,
    VIDEO_INITIAL_FRAME_METRIC,
    FreeCamera,
    VideoCamera,
    video_camera,
    video_capture_interval,
)

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

type RoboticsAction = NDArray[numpy.float32] | NDArray[numpy.float64]


class _MujocoEnvironment(Protocol):
    model: mujoco.MjModel
    data: mujoco.MjData


@dataclass(frozen=True, slots=True)
class _GoalFacts:
    distance: float | None = None
    position_distance: float | None = None
    rotation_distance: float | None = None
    orientation_similarity: float | None = None
    task_progress: float | None = None


class RoboticsEnvironment:
    """Seeded adapter with goal, control, motion, and task diagnostics."""

    def __init__(self, episode: EpisodeSpec, *, config: RoboticsConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not RoboticsConfig:
            raise TypeError("config must be RoboticsConfig")
        if episode.scenario is not None:
            raise ValueError("Robotics configuration belongs in RoboticsConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, RoboticsAction],
            gymnasium.make(config.environment_id),
        )
        upstream = cast(_MujocoEnvironment, self._environment.unwrapped)
        self._model = upstream.model
        self._data = upstream.data
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
            raise RuntimeError("Gymnasium-Robotics action space changed incompatibly")
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._previous_state: NDArray[numpy.float64] | None = None
        self._initial_goal = _GoalFacts()
        self._previous_goal = _GoalFacts()
        self._best_goal_distance: float | None = None
        self._success_ever = False
        self._first_success_step = -1
        self._successful_steps = 0
        self._success_lost_count = 0
        self._previous_success = False
        self._cumulative_return = 0.0
        self._best_reward = -math.inf
        self._cumulative_action_l2 = 0.0
        self._cumulative_action_max_abs = 0.0
        self._saturated_action_components = 0
        self._zero_action_count = 0
        self._cumulative_state_motion = 0.0
        self._no_state_change_count = 0
        self._completed_tasks: set[str] = set()
        self._video_camera = _renderer_camera(video_camera(config.profile))
        self._video_capture_failed = False
        self._video_capture_interval = video_capture_interval(config.max_episode_steps)
        self._video_renderer: mujoco.Renderer | None = None

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, info = self._environment.reset(seed=self._seed)
        public = _policy_value(observation, name="observation")
        _validate_observation(public, config=self._config)
        self._previous_state = _state_array(public)
        self._initial_goal = _goal_facts(public, config=self._config)
        self._previous_goal = self._initial_goal
        self._best_goal_distance = self._initial_goal.distance
        if self._config.family == "franka-kitchen":
            _, completed, _ = _kitchen_tasks(info)
            self._completed_tasks.update(completed)
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
            raise RuntimeError("Robotics state history is unavailable")
        control = _action(action, config=self._config)
        initial_video_frame = self._capture_video_frame() if self._steps == 0 else None
        observation, reward, terminated, truncated, info = self._environment.step(control)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Gymnasium-Robotics returned invalid termination flags")
        public = _policy_value(observation, name="observation")
        _validate_observation(public, config=self._config)
        number = _number(reward, name="reward")
        self._steps += 1

        upstream = _upstream_metrics(info, config=self._config)
        success = upstream["success"]
        if type(success) is not bool:
            raise RuntimeError("Gymnasium-Robotics success metric is invalid")
        current_state = _state_array(public)
        state_motion = float(numpy.linalg.norm(current_state - previous_state))
        no_state_change = state_motion <= 1e-12
        self._cumulative_state_motion += state_motion
        self._no_state_change_count += int(no_state_change)

        action_l2 = float(numpy.linalg.norm(control))
        action_max_abs = float(numpy.max(numpy.abs(control)))
        saturated_components = int(numpy.count_nonzero(numpy.abs(control) >= 1.0))
        zero_action = action_max_abs == 0.0
        self._cumulative_action_l2 += action_l2
        self._cumulative_action_max_abs += action_max_abs
        self._saturated_action_components += saturated_components
        self._zero_action_count += int(zero_action)

        goal = _goal_facts(public, config=self._config)
        distance_improvement = _difference(self._previous_goal.distance, goal.distance)
        distance_improvement_from_initial = _difference(
            self._initial_goal.distance,
            goal.distance,
        )
        previous_best_goal_distance = self._best_goal_distance
        new_best_goal_distance = bool(
            goal.distance is not None
            and (
                previous_best_goal_distance is None
                or goal.distance < previous_best_goal_distance - 1e-15
            )
        )
        if new_best_goal_distance:
            self._best_goal_distance = goal.distance

        success_first_reached = success and not self._success_ever
        success_lost = self._previous_success and not success
        if success_first_reached:
            self._first_success_step = self._steps
        self._success_ever = self._success_ever or success
        self._successful_steps += int(success)
        self._success_lost_count += int(success_lost)
        self._previous_success = success
        self._cumulative_return += number
        self._best_reward = max(self._best_reward, number)

        remaining_tasks, completed_tasks, step_completions = _optional_kitchen_tasks(
            info,
            config=self._config,
        )
        self._completed_tasks.update(completed_tasks)
        task_completion_fraction = (
            len(self._completed_tasks) / _KITCHEN_TASK_COUNT
            if self._config.family == "franka-kitchen"
            else None
        )
        if self._config.family == "franka-kitchen" and not math.isclose(
            number,
            float(len(step_completions)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("FrankaKitchen reward semantics drifted")
        _validate_sparse_reward(number, success=success, config=self._config)

        self._done = terminated or truncated
        self._previous_state = current_state
        self._previous_goal = goal
        terminal_reason = (
            "success_termination"
            if terminated and success
            else "upstream_termination"
            if terminated
            else "time_limit"
            if truncated
            else "in_progress"
        )
        completed_task_names = list[PolicyValue](sorted(self._completed_tasks))
        newly_completed_task_names = list[PolicyValue](sorted(step_completions))
        remaining_task_names = list[PolicyValue](sorted(remaining_tasks))
        video_frame = None
        if (
            self._steps == 1
            or self._steps % self._video_capture_interval == 0
            or self._done
        ):
            video_frame = self._capture_video_frame()
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._config.max_episode_steps - self._steps, 0),
            "success": success,
            "success_ever": self._success_ever,
            "success_first_reached_this_step": success_first_reached,
            "first_success_step": self._first_success_step,
            "successful_step_count": self._successful_steps,
            "successful_step_fraction": self._successful_steps / self._steps,
            "success_lost_this_step": success_lost,
            "success_lost_count": self._success_lost_count,
            "goal_distance": goal.distance,
            "initial_goal_distance": self._initial_goal.distance,
            "best_goal_distance": self._best_goal_distance,
            "goal_distance_improvement_this_step": distance_improvement,
            "goal_distance_improvement_from_initial": distance_improvement_from_initial,
            "new_best_goal_distance": new_best_goal_distance,
            "goal_position_distance": goal.position_distance,
            "goal_rotation_distance": goal.rotation_distance,
            "goal_orientation_similarity": goal.orientation_similarity,
            "task_progress": (
                task_completion_fraction
                if task_completion_fraction is not None
                else goal.task_progress
            ),
            "action_l2_norm": action_l2,
            "action_max_abs": action_max_abs,
            "saturated_action_component_count": saturated_components,
            "saturated_action_component_fraction": (
                saturated_components / self._config.action_size
            ),
            "zero_action": zero_action,
            "mean_action_l2_norm": self._cumulative_action_l2 / self._steps,
            "mean_action_max_abs": self._cumulative_action_max_abs / self._steps,
            "cumulative_saturated_action_component_count": (self._saturated_action_components),
            "zero_action_count": self._zero_action_count,
            "state_motion_l2": state_motion,
            "no_state_change": no_state_change,
            "mean_state_motion_l2": self._cumulative_state_motion / self._steps,
            "no_state_change_count": self._no_state_change_count,
            "reward_this_step": number,
            "best_reward": self._best_reward,
            "cumulative_return": self._cumulative_return,
            "completed_task_names": completed_task_names,
            "newly_completed_task_names": newly_completed_task_names,
            "remaining_task_names": remaining_task_names,
            "completed_tasks": (
                len(self._completed_tasks) if self._config.family == "franka-kitchen" else None
            ),
            "task_completion_fraction": task_completion_fraction,
            "task_stage": _task_stage(
                success=success,
                success_ever=self._success_ever,
                terminated=terminated,
                truncated=truncated,
                task_completion_fraction=task_completion_fraction,
            ),
            "terminal_reason": terminal_reason,
            VIDEO_CAPTURE_FAILED_METRIC: self._video_capture_failed,
        }
        if initial_video_frame is not None:
            metrics[VIDEO_INITIAL_FRAME_METRIC] = initial_video_frame
        if video_frame is not None:
            metrics[VIDEO_FRAME_METRIC] = video_frame
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
        self._close_video_renderer()
        self._environment.close()
        self._closed = True

    def _capture_video_frame(self) -> TensorValue | None:
        if self._video_capture_failed:
            return None
        try:
            if self._video_renderer is None:
                self._video_renderer = mujoco.Renderer(
                    self._model,
                    height=VIDEO_FRAME_HEIGHT,
                    width=VIDEO_FRAME_WIDTH,
                )
            self._video_renderer.update_scene(
                self._data,
                camera=self._video_camera,
            )
            raw = self._video_renderer.render()
            frame = numpy.asarray(raw)
            if frame.shape != VIDEO_FRAME_SHAPE or frame.dtype != numpy.dtype(numpy.uint8):
                raise RuntimeError("Gymnasium-Robotics camera frame shape or dtype drifted")
            contiguous = numpy.ascontiguousarray(frame)
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


def _renderer_camera(camera: VideoCamera) -> str | int | mujoco.MjvCamera:
    if type(camera) is not FreeCamera:
        return camera
    selected = mujoco.MjvCamera()
    selected.type = mujoco.mjtCamera.mjCAMERA_FREE
    selected.fixedcamid = -1
    selected.trackbodyid = -1
    selected.lookat[:] = camera.lookat
    selected.distance = camera.distance
    selected.azimuth = camera.azimuth
    selected.elevation = camera.elevation
    return selected


def _action(value: PolicyValue, *, config: RoboticsConfig) -> RoboticsAction:
    if type(value) is not list or len(value) != config.action_size:
        raise InvalidAction()
    items: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -1.0 <= item <= 1.0:
            raise InvalidAction()
        items.append(item)
    return cast(
        RoboticsAction,
        numpy.asarray(items, dtype=numpy.dtype(config.action_dtype)),
    )


def _validate_observation(value: PolicyValue, *, config: RoboticsConfig) -> None:
    if config.goal_size is None:
        _require_tensor(value, shape=(config.observation_size,), name="observation")
        return
    if type(value) is not dict or set(value) != {
        "observation",
        "achieved_goal",
        "desired_goal",
    }:
        raise RuntimeError("Gymnasium-Robotics returned an invalid goal observation")
    _require_tensor(
        value["observation"],
        shape=(config.observation_size,),
        name="observation",
    )
    if config.goal_size == -1:
        _validate_kitchen_goals(value["achieved_goal"], name="achieved_goal")
        _validate_kitchen_goals(value["desired_goal"], name="desired_goal")
        return
    _require_tensor(value["achieved_goal"], shape=(config.goal_size,), name="achieved_goal")
    _require_tensor(value["desired_goal"], shape=(config.goal_size,), name="desired_goal")


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


def _require_tensor(value: PolicyValue, *, shape: tuple[int, ...], name: str) -> None:
    if type(value) is not TensorValue or value.dtype != "float64" or value.shape != shape:
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name} tensor")


def _state_array(value: PolicyValue) -> NDArray[numpy.float64]:
    state = value["observation"] if type(value) is dict else value
    return _tensor_array(state, name="observation")


def _goal_facts(value: PolicyValue, *, config: RoboticsConfig) -> _GoalFacts:
    if config.goal_size == -1:
        return _GoalFacts()
    if config.goal_size is not None:
        if type(value) is not dict:
            raise RuntimeError("Robotics goal observation is invalid")
        achieved = _tensor_array(value["achieved_goal"], name="achieved_goal")
        desired = _tensor_array(value["desired_goal"], name="desired_goal")
        if config.goal_size == 7:
            position_distance = float(numpy.linalg.norm(achieved[:3] - desired[:3]))
            achieved_quaternion = achieved[3:].copy()
            desired_quaternion = desired[3:].copy()
            if config.profile.startswith("hand-manipulate-pen"):
                achieved_euler = rotations.quat2euler(achieved_quaternion)
                desired_euler = rotations.quat2euler(desired_quaternion)
                achieved_euler[2] = desired_euler[2]
                achieved_quaternion = rotations.euler2quat(achieved_euler)
            difference = rotations.quat_mul(
                achieved_quaternion,
                rotations.quat_conjugate(desired_quaternion),
            )
            rotation_distance = 2.0 * math.acos(max(-1.0, min(1.0, float(difference[0]))))
            return _GoalFacts(
                distance=10.0 * position_distance + rotation_distance,
                position_distance=position_distance,
                rotation_distance=rotation_distance,
            )
        distance = float(numpy.linalg.norm(achieved - desired))
        return _GoalFacts(distance=distance, position_distance=distance)

    state = _state_array(value)
    if config.profile == "adroit-hand-door":
        door_angle = float(state[28])
        return _GoalFacts(
            distance=max(1.35 - door_angle, 0.0),
            task_progress=door_angle,
        )
    if config.profile == "adroit-hand-hammer":
        return _GoalFacts(task_progress=float(state[-1]))
    if config.profile == "adroit-hand-pen":
        position_distance = float(numpy.linalg.norm(state[39:42]))
        orientation_similarity = float(numpy.dot(state[33:36], state[36:39]))
        return _GoalFacts(
            distance=position_distance + (1.0 - orientation_similarity),
            position_distance=position_distance,
            rotation_distance=1.0 - orientation_similarity,
            orientation_similarity=orientation_similarity,
        )
    if config.profile == "adroit-hand-relocate":
        distance = float(numpy.linalg.norm(state[-3:]))
        return _GoalFacts(distance=distance, position_distance=distance)
    return _GoalFacts()


def _difference(previous: float | None, current: float | None) -> float | None:
    return None if previous is None or current is None else previous - current


def _upstream_metrics(value: object, *, config: RoboticsConfig) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("Gymnasium-Robotics returned invalid metrics")
    success_name = "is_success" if "is_success" in value else "success"
    if config.family == "franka-kitchen":
        remaining, completed, _ = _kitchen_tasks(value)
        return {"success": not remaining and len(completed) == _KITCHEN_TASK_COUNT}
    if success_name not in value:
        raise RuntimeError("Gymnasium-Robotics omitted its success metric")
    return {"success": _flag(value[success_name], name=success_name)}


def _kitchen_tasks(value: object) -> tuple[set[str], set[str], set[str]]:
    if type(value) is not dict:
        raise RuntimeError("FrankaKitchen returned invalid metrics")
    remaining = value.get("tasks_to_complete")
    completed = value.get("episode_task_completions")
    step = value.get("step_task_completions", [])
    if type(remaining) is not list or type(completed) is not list or type(step) is not list:
        raise RuntimeError("FrankaKitchen omitted task completion metrics")
    if any(type(item) is not str for item in remaining + completed + step):
        raise RuntimeError("FrankaKitchen returned invalid task completion metrics")
    return set(remaining), set(completed), set(step)


def _optional_kitchen_tasks(
    value: object,
    *,
    config: RoboticsConfig,
) -> tuple[set[str], set[str], set[str]]:
    return _kitchen_tasks(value) if config.family == "franka-kitchen" else (set(), set(), set())


def _validate_sparse_reward(number: float, *, success: bool, config: RoboticsConfig) -> None:
    if config.family in {"fetch", "shadow-hand", "shadow-hand-touch"}:
        expected = 0.0 if success else -1.0
    elif config.family == "maze":
        expected = 1.0 if success else 0.0
    else:
        return
    if number != expected:
        raise RuntimeError("Gymnasium-Robotics sparse reward semantics drifted")


def _task_stage(
    *,
    success: bool,
    success_ever: bool,
    terminated: bool,
    truncated: bool,
    task_completion_fraction: float | None,
) -> str:
    if success:
        return "goal_achieved"
    if terminated:
        return "terminated_without_success"
    if truncated:
        return "time_limit"
    if task_completion_fraction is not None and task_completion_fraction > 0.0:
        return "partial_task_completion"
    if success_ever:
        return "goal_lost_after_achievement"
    return "approaching_goal"


def _policy_value(value: object, *, name: str) -> PolicyValue:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        if type(value) is float and not math.isfinite(value):
            raise RuntimeError(f"Gymnasium-Robotics returned non-finite {name}")
        return cast(PolicyValue, value)
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        try:
            return int(cast(SupportsIndex, value).__index__())
        except (OverflowError, ValueError) as error:
            raise RuntimeError(f"Gymnasium-Robotics returned out-of-range {name}") from error
    if isinstance(value, numpy.floating):
        return _number(value, name=name)
    if type(value) is numpy.ndarray:
        return _tensor(value, name=name)
    if isinstance(value, Mapping):
        public: dict[str, PolicyValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeError("Gymnasium-Robotics returned a non-string observation key")
            public[key] = _policy_value(item, name=f"{name}.{key}")
        return public
    if type(value) is list:
        return [_policy_value(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if type(value) is tuple:
        return tuple(
            _policy_value(item, name=f"{name}[{index}]") for index, item in enumerate(value)
        )
    raise RuntimeError(
        f"Gymnasium-Robotics returned unsupported {name} carrier {type(value).__name__}"
    )


def _tensor(value: object, *, name: str) -> TensorValue:
    if type(value) is not numpy.ndarray:
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name} tensor")
    dtype = value.dtype.name
    if dtype not in _TENSOR_DTYPES:
        raise RuntimeError(f"Gymnasium-Robotics returned unsupported {name} dtype {dtype}")
    if numpy.issubdtype(value.dtype, numpy.floating) and not numpy.isfinite(value).all():
        raise RuntimeError(f"Gymnasium-Robotics returned non-finite {name}")
    array = numpy.ascontiguousarray(value)
    if array.dtype.itemsize > 1 and (
        array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big")
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    return TensorValue(
        dtype=dtype,
        shape=tuple(int(size) for size in array.shape),
        data=array.tobytes(order="C"),
    )


def _tensor_array(value: PolicyValue, *, name: str) -> NDArray[numpy.float64]:
    if type(value) is not TensorValue or value.dtype != "float64" or len(value.shape) != 1:
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name} tensor")
    return numpy.frombuffer(value.data, dtype="<f8").copy()


def _flag(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, numpy.bool_):
        return bool(value)
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name} flag")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Gymnasium-Robotics returned invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Gymnasium-Robotics returned non-finite {name}")
    return number


__all__ = ["RoboticsEnvironment"]
