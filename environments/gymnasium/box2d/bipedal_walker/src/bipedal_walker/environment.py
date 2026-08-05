"""One fresh Gymnasium BipedalWalker-v3 Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import BipedalWalkerConfig
from .visual import (
    VISUAL_CAPTURE_FAILED_METRIC,
    VISUAL_FRAME_METRIC,
    VISUAL_FRAME_SHAPE,
    VISUAL_INITIAL_FRAME_METRIC,
    visual_capture_interval,
)

_SCALAR_OBSERVATION_NAMES = (
    "hull_angle",
    "hull_angular_velocity",
    "horizontal_velocity",
    "vertical_velocity",
    "left_hip_angle",
    "left_hip_angular_velocity",
    "left_knee_angle",
    "left_knee_angular_velocity",
    "left_foot_contact",
    "right_hip_angle",
    "right_hip_angular_velocity",
    "right_knee_angle",
    "right_knee_angular_velocity",
    "right_foot_contact",
)
_OBSERVATION_SIZE = 24
_ACTION_SIZE = 4
_ACTION_COMPONENTS = (
    "left_hip",
    "left_knee",
    "right_hip",
    "right_knee",
)
_MAX_EPISODE_STEPS = 1_600
_FRAMES_PER_SECOND = 50.0
_SCALE = 30.0
_VIEWPORT_WIDTH = 600.0
_VIEWPORT_HEIGHT = 400.0
_MOTOR_TORQUE = 80.0
_HIP_MOTOR_SPEED = 4.0
_KNEE_MOTOR_SPEED = 6.0
_MOTOR_ENERGY_COEFFICIENT = 0.00035 * _MOTOR_TORQUE
_FORWARD_SHAPING_COEFFICIENT = 130.0
_HULL_ANGLE_SHAPING_COEFFICIENT = 5.0
_LIDAR_RANGE = 160.0 / _SCALE


class BipedalWalkerEnvironment:
    """The seeded strict adapter around configured BipedalWalker-v3."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: BipedalWalkerConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BipedalWalkerConfig:
            raise TypeError("config must be BipedalWalkerConfig")
        if episode.scenario is not None:
            raise ValueError(
                "BipedalWalker configuration belongs in BipedalWalkerConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "BipedalWalker-v3",
                hardcore=config.hardcore,
                render_mode="rgb_array",
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._observation: dict[str, PolicyValue] | None = None
        self._steps = 0
        self._cumulative_requested_motor_penalty = 0.0
        self._cumulative_charged_motor_penalty = 0.0
        self._cumulative_forward_shaping = 0.0
        self._cumulative_posture_shaping = 0.0
        self._cumulative_terminal_override = 0.0
        self._cumulative_return = 0.0
        self._relative_progress_coordinate = 0.0
        self._maximum_relative_progress_coordinate = 0.0
        self._visual_capture_failed = False
        self._visual_capture_interval = visual_capture_interval(
            _MAX_EPISODE_STEPS
        )

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        self._observation = public_observation
        self._started = True
        return public_observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        previous_observation = self._observation
        if previous_observation is None:
            raise RuntimeError("BipedalWalker observation is unavailable")
        public_action = _action(action)
        initial_visual_frame = (
            self._capture_visual_frame() if self._steps == 0 else None
        )
        observation, reward, terminated, truncated, _ = self._environment.step(public_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "BipedalWalker returned invalid termination flags"
            )
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        self._steps += 1
        (
            requested_motor_penalty,
            charged_motor_penalty,
            forward_shaping,
            posture_shaping,
            terminal_override,
            reward_was_overridden,
        ) = _reward_components(
            previous_observation,
            public_observation,
            public_action,
            reward=public_reward,
            terminated=terminated,
        )
        if truncated != (self._steps == _MAX_EPISODE_STEPS):
            raise RuntimeError("BipedalWalker time-limit semantics drifted")
        self._cumulative_requested_motor_penalty += requested_motor_penalty
        self._cumulative_charged_motor_penalty += charged_motor_penalty
        self._cumulative_forward_shaping += forward_shaping
        self._cumulative_posture_shaping += posture_shaping
        self._cumulative_terminal_override += terminal_override
        self._cumulative_return += public_reward
        progress_delta = forward_shaping * _SCALE / _FORWARD_SHAPING_COEFFICIENT
        self._relative_progress_coordinate += progress_delta
        self._maximum_relative_progress_coordinate = max(
            self._maximum_relative_progress_coordinate,
            self._relative_progress_coordinate,
        )
        metrics = _transition_metrics(
            previous_observation,
            public_observation,
            public_action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            requested_motor_penalty=requested_motor_penalty,
            charged_motor_penalty=charged_motor_penalty,
            forward_shaping=forward_shaping,
            posture_shaping=posture_shaping,
            terminal_override=terminal_override,
            reward_was_overridden=reward_was_overridden,
            progress_delta=progress_delta,
            cumulative_requested_motor_penalty=(
                self._cumulative_requested_motor_penalty
            ),
            cumulative_charged_motor_penalty=self._cumulative_charged_motor_penalty,
            cumulative_forward_shaping=self._cumulative_forward_shaping,
            cumulative_posture_shaping=self._cumulative_posture_shaping,
            cumulative_terminal_override=self._cumulative_terminal_override,
            cumulative_return=self._cumulative_return,
            relative_progress_coordinate=self._relative_progress_coordinate,
            maximum_relative_progress_coordinate=(
                self._maximum_relative_progress_coordinate
            ),
        )
        self._observation = public_observation
        self._done = terminated or truncated
        visual_frame = None
        if (
            self._steps == 1
            or self._steps % self._visual_capture_interval == 0
            or self._done
        ):
            visual_frame = self._capture_visual_frame()
        metrics[VISUAL_CAPTURE_FAILED_METRIC] = self._visual_capture_failed
        if initial_visual_frame is not None:
            metrics[VISUAL_INITIAL_FRAME_METRIC] = initial_visual_frame
        if visual_frame is not None:
            metrics[VISUAL_FRAME_METRIC] = visual_frame
        return Step(
            observation=public_observation,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _capture_visual_frame(self) -> TensorValue | None:
        if self._visual_capture_failed:
            return None
        try:
            raw = self._environment.render()
            if (
                type(raw) is not numpy.ndarray
                or raw.dtype != numpy.dtype(numpy.uint8)
                or raw.shape != VISUAL_FRAME_SHAPE
            ):
                raise RuntimeError("BipedalWalker RGB frame shape or dtype drifted")
            contiguous = numpy.ascontiguousarray(raw)
            return TensorValue(
                dtype="uint8",
                shape=VISUAL_FRAME_SHAPE,
                data=contiguous.tobytes(order="C"),
            )
        except Exception:
            self._visual_capture_failed = True
            return None


def _action(value: PolicyValue) -> list[float]:
    if type(value) is not list or len(value) != _ACTION_SIZE:
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
    return action


def _observation(value: object) -> dict[str, PolicyValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError("BipedalWalker returned an invalid observation")
    items = tuple(value)
    if len(items) != _OBSERVATION_SIZE:
        raise RuntimeError(
            "BipedalWalker returned an invalid observation shape"
        )

    observation: dict[str, PolicyValue] = {}
    for index, name in enumerate(_SCALAR_OBSERVATION_NAMES):
        if name in {"left_foot_contact", "right_foot_contact"}:
            observation[name] = _contact(items[index], name=name)
        else:
            observation[name] = _number(items[index], name=name)
    observation["lidar_ranges"] = [
        _number(item, name=f"lidar range {index}")
        for index, item in enumerate(items[14:])
    ]
    return observation


def _contact(value: object, *, name: str) -> bool:
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"BipedalWalker returned an invalid {name}")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"BipedalWalker returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"BipedalWalker returned a non-finite {name}")
    return number


def _reward_components(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    action: list[float],
    *,
    reward: float,
    terminated: bool,
) -> tuple[float, float, float, float, float, bool]:
    requested_motor_penalty = _MOTOR_ENERGY_COEFFICIENT * math.fsum(
        abs(value) for value in action
    )
    reward_was_overridden = terminated and reward == -100.0
    geometric_posture_shaping = -_HULL_ANGLE_SHAPING_COEFFICIENT * (
        abs(_float_field(current, "hull_angle"))
        - abs(_float_field(previous, "hull_angle"))
    )
    if reward_was_overridden:
        charged_motor_penalty = 0.0
        forward_shaping = 0.0
        posture_shaping = 0.0
        terminal_override = -100.0
    else:
        charged_motor_penalty = requested_motor_penalty
        posture_shaping = geometric_posture_shaping
        forward_shaping = reward + charged_motor_penalty - posture_shaping
        terminal_override = 0.0
    reconstructed_reward = (
        forward_shaping
        + posture_shaping
        - charged_motor_penalty
        + terminal_override
    )
    if not math.isclose(reward, reconstructed_reward, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("BipedalWalker reward decomposition drifted")
    return (
        requested_motor_penalty,
        charged_motor_penalty,
        forward_shaping,
        posture_shaping,
        terminal_override,
        reward_was_overridden,
    )


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    action: list[float],
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    requested_motor_penalty: float,
    charged_motor_penalty: float,
    forward_shaping: float,
    posture_shaping: float,
    terminal_override: float,
    reward_was_overridden: bool,
    progress_delta: float,
    cumulative_requested_motor_penalty: float,
    cumulative_charged_motor_penalty: float,
    cumulative_forward_shaping: float,
    cumulative_posture_shaping: float,
    cumulative_terminal_override: float,
    cumulative_return: float,
    relative_progress_coordinate: float,
    maximum_relative_progress_coordinate: float,
) -> dict[str, PolicyValue]:
    action_by_component: dict[str, PolicyValue] = {}
    target_speeds: dict[str, PolicyValue] = {}
    maximum_torques: dict[str, PolicyValue] = {}
    for component, value in zip(_ACTION_COMPONENTS, action, strict=True):
        action_by_component[component] = value
        target_speeds[component] = _signed_unit(value) * (
            _HIP_MOTOR_SPEED if "hip" in component else _KNEE_MOTOR_SPEED
        )
        maximum_torques[component] = _MOTOR_TORQUE * abs(value)
    previous_left_contact = _bool_field(previous, "left_foot_contact")
    previous_right_contact = _bool_field(previous, "right_foot_contact")
    left_contact = _bool_field(current, "left_foot_contact")
    right_contact = _bool_field(current, "right_foot_contact")
    lidar = _lidar(current)
    closest_lidar_fraction = min(lidar)
    closest_lidar_index = lidar.index(closest_lidar_fraction)
    if terminated and reward_was_overridden:
        terminal_reason = "fall_or_behind_start"
    elif terminated:
        terminal_reason = "course_complete"
    elif truncated:
        terminal_reason = "time_limit"
    else:
        terminal_reason = "none"
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "simulated_seconds": step_count / _FRAMES_PER_SECOND,
        "requested_action": action_by_component,
        "target_motor_speeds_radians_per_second": target_speeds,
        "maximum_motor_torques": maximum_torques,
        "total_absolute_motor_command": math.fsum(abs(value) for value in action),
        "requested_motor_energy_penalty": requested_motor_penalty,
        "charged_motor_energy_penalty": charged_motor_penalty,
        "reward_was_terminal_override": reward_was_overridden,
        "forward_progress_shaping_reward": forward_shaping,
        "hull_posture_shaping_reward": posture_shaping,
        "terminal_override_reward": terminal_override,
        "reward_from_public_terms": (
            forward_shaping
            + posture_shaping
            - charged_motor_penalty
            + terminal_override
        ),
        "relative_progress_coordinate_delta": progress_delta,
        "relative_progress_coordinate": relative_progress_coordinate,
        "maximum_relative_progress_coordinate": maximum_relative_progress_coordinate,
        "cumulative_requested_motor_energy_penalty": (
            cumulative_requested_motor_penalty
        ),
        "cumulative_charged_motor_energy_penalty": cumulative_charged_motor_penalty,
        "cumulative_forward_progress_shaping_reward": cumulative_forward_shaping,
        "cumulative_hull_posture_shaping_reward": cumulative_posture_shaping,
        "cumulative_terminal_override_reward": cumulative_terminal_override,
        "cumulative_return": cumulative_return,
        "hull_angle_radians": _float_field(current, "hull_angle"),
        "hull_angle_degrees": math.degrees(_float_field(current, "hull_angle")),
        "absolute_hull_angle_radians": abs(_float_field(current, "hull_angle")),
        "hull_angular_velocity_radians_per_second": (
            _float_field(current, "hull_angular_velocity") * _FRAMES_PER_SECOND / 2.0
        ),
        "horizontal_velocity_world_units_per_second": (
            _float_field(current, "horizontal_velocity")
            * _FRAMES_PER_SECOND
            / (0.3 * (_VIEWPORT_WIDTH / _SCALE))
        ),
        "vertical_velocity_world_units_per_second": (
            _float_field(current, "vertical_velocity")
            * _FRAMES_PER_SECOND
            / (0.3 * (_VIEWPORT_HEIGHT / _SCALE))
        ),
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
        "left_foot_contact_started": left_contact and not previous_left_contact,
        "left_foot_contact_ended": previous_left_contact and not left_contact,
        "right_foot_contact_started": right_contact and not previous_right_contact,
        "right_foot_contact_ended": previous_right_contact and not right_contact,
        "support_state": _support_state(left_contact, right_contact),
        "closest_lidar_fraction": closest_lidar_fraction,
        "closest_lidar_ray_index": closest_lidar_index,
        "closest_lidar_ray_angle_from_down_radians": 1.5 * closest_lidar_index / 10.0,
        "closest_lidar_distance_world_units": closest_lidar_fraction * _LIDAR_RANGE,
        "terminal_reason": terminal_reason,
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"BipedalWalker returned invalid {name}")
    return value


def _bool_field(observation: dict[str, PolicyValue], name: str) -> bool:
    value = observation.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"BipedalWalker returned invalid {name}")
    return value


def _lidar(observation: dict[str, PolicyValue]) -> list[float]:
    value = observation.get("lidar_ranges")
    if type(value) is not list or len(value) != 10:
        raise RuntimeError("BipedalWalker returned invalid lidar ranges")
    lidar: list[float] = []
    for item in value:
        if type(item) is not float or not 0.0 <= item <= 1.0:
            raise RuntimeError("BipedalWalker returned invalid lidar ranges")
        lidar.append(item)
    return lidar


def _signed_unit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _support_state(left_contact: bool, right_contact: bool) -> str:
    if left_contact and right_contact:
        return "double_support"
    if left_contact:
        return "left_support"
    if right_contact:
        return "right_support"
    return "airborne"


__all__ = ["BipedalWalkerEnvironment"]
