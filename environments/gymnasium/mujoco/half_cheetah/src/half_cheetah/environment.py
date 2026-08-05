"""One fresh Gymnasium HalfCheetah-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import HalfCheetahConfig
from .visual import (
    VISUAL_CAPTURE_FAILED_METRIC,
    VISUAL_FRAME_METRIC,
    VISUAL_FRAME_SHAPE,
    VISUAL_INITIAL_FRAME_METRIC,
    visual_capture_interval,
)

_BODY_FIELDS = (
    "torso_z_position",
    "torso_pitch_angle",
    "back_thigh_angle",
    "back_shin_angle",
    "back_foot_angle",
    "front_thigh_angle",
    "front_shin_angle",
    "front_foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_pitch_angular_velocity",
    "back_thigh_angular_velocity",
    "back_shin_angular_velocity",
    "back_foot_angular_velocity",
    "front_thigh_angular_velocity",
    "front_shin_angular_velocity",
    "front_foot_angular_velocity",
)
_ACTION_COMPONENTS = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)
_ACTUATOR_GEARS = (120.0, 90.0, 60.0, 120.0, 60.0, 30.0)
_MAX_EPISODE_STEPS = 1_000
_MODEL_TIMESTEP_SECONDS = 0.01


class HalfCheetahEnvironment:
    """The seeded strict adapter around configured HalfCheetah-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HalfCheetahConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HalfCheetahConfig:
            raise TypeError("config must be HalfCheetahConfig")
        if episode.scenario is not None:
            raise ValueError(
                "HalfCheetah configuration belongs in HalfCheetahConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._exclude_positions = (
            config.exclude_current_positions_from_observation
        )
        self._frame_skip = config.frame_skip
        self._forward_reward_weight = config.forward_reward_weight
        self._control_cost_weight = config.ctrl_cost_weight
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "HalfCheetah-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
                render_mode="rgb_array",
                width=VISUAL_FRAME_SHAPE[1],
                height=VISUAL_FRAME_SHAPE[0],
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._initial_x_position: float | None = None
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._minimum_x_velocity = math.inf
        self._maximum_x_velocity = -math.inf
        self._minimum_torso_z_position = math.inf
        self._maximum_torso_z_position = -math.inf
        self._maximum_absolute_torso_pitch = 0.0
        self._forward_steps = 0
        self._cumulative_forward_reward = 0.0
        self._cumulative_control_reward = 0.0
        self._cumulative_return = 0.0
        self._cumulative_absolute_action = 0.0
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
        self._started = True
        return _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        applied_action = _action(action)
        initial_visual_frame = (
            self._capture_visual_frame() if self._steps == 0 else None
        )
        observation, reward, terminated, truncated, info = self._environment.step(
            applied_action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "HalfCheetah returned invalid termination flags"
            )
        public_observation = _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        self._steps += 1
        seconds_per_step = self._frame_skip * _MODEL_TIMESTEP_SECONDS
        if self._initial_x_position is None:
            self._initial_x_position = (
                provider_metrics["x_position"]
                - provider_metrics["x_velocity"] * seconds_per_step
            )
            self._minimum_x_position = self._initial_x_position
            self._maximum_x_position = self._initial_x_position
        self._minimum_x_position = min(
            self._minimum_x_position,
            provider_metrics["x_position"],
        )
        self._maximum_x_position = max(
            self._maximum_x_position,
            provider_metrics["x_position"],
        )
        self._minimum_x_velocity = min(
            self._minimum_x_velocity,
            provider_metrics["x_velocity"],
        )
        self._maximum_x_velocity = max(
            self._maximum_x_velocity,
            provider_metrics["x_velocity"],
        )
        torso_z = _float_field(public_observation, "torso_z_position")
        torso_pitch = _float_field(public_observation, "torso_pitch_angle")
        self._minimum_torso_z_position = min(self._minimum_torso_z_position, torso_z)
        self._maximum_torso_z_position = max(self._maximum_torso_z_position, torso_z)
        self._maximum_absolute_torso_pitch = max(
            self._maximum_absolute_torso_pitch,
            abs(torso_pitch),
        )
        self._forward_steps += int(provider_metrics["x_velocity"] > 0.0)
        self._cumulative_forward_reward += provider_metrics["reward_forward"]
        self._cumulative_control_reward += provider_metrics["reward_control"]
        self._cumulative_return += public_reward
        self._cumulative_absolute_action += float(
            numpy.sum(numpy.abs(applied_action))
        )
        metrics = _transition_metrics(
            public_observation,
            applied_action,
            provider_metrics=provider_metrics,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            frame_skip=self._frame_skip,
            forward_reward_weight=self._forward_reward_weight,
            control_cost_weight=self._control_cost_weight,
            initial_x_position=self._initial_x_position,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            minimum_x_velocity=self._minimum_x_velocity,
            maximum_x_velocity=self._maximum_x_velocity,
            minimum_torso_z_position=self._minimum_torso_z_position,
            maximum_torso_z_position=self._maximum_torso_z_position,
            maximum_absolute_torso_pitch=self._maximum_absolute_torso_pitch,
            forward_steps=self._forward_steps,
            cumulative_forward_reward=self._cumulative_forward_reward,
            cumulative_control_reward=self._cumulative_control_reward,
            cumulative_return=self._cumulative_return,
            cumulative_absolute_action=self._cumulative_absolute_action,
        )
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
                raise RuntimeError("HalfCheetah RGB frame shape or dtype drifted")
            contiguous = numpy.ascontiguousarray(raw)
            return TensorValue(
                dtype="uint8",
                shape=VISUAL_FRAME_SHAPE,
                data=contiguous.tobytes(order="C"),
            )
        except Exception:
            self._visual_capture_failed = True
            return None


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 6:
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
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    exclude_positions: bool,
) -> dict[str, PolicyValue]:
    expected_shape = (17,) if exclude_positions else (18,)
    if (
        type(value) is not numpy.ndarray
        or value.shape != expected_shape
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("HalfCheetah returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["torso_x_position"] = _number(
            value[0],
            name="torso x position",
        )
        offset = 1
    for name, item in zip(
        _BODY_FIELDS,
        value[offset:],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    return observation


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("HalfCheetah returned invalid metrics")
    names = (
        "x_position",
        "x_velocity",
        "reward_forward",
        "reward_ctrl",
    )
    if not set(names).issubset(value):
        raise RuntimeError("HalfCheetah omitted public metrics")
    return {
        (
            "reward_control" if name == "reward_ctrl" else name
        ): _number(value[name], name=name.replace("_", " "))
        for name in names
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, float],
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    frame_skip: int,
    forward_reward_weight: float,
    control_cost_weight: float,
    initial_x_position: float,
    minimum_x_position: float,
    maximum_x_position: float,
    minimum_x_velocity: float,
    maximum_x_velocity: float,
    minimum_torso_z_position: float,
    maximum_torso_z_position: float,
    maximum_absolute_torso_pitch: float,
    forward_steps: int,
    cumulative_forward_reward: float,
    cumulative_control_reward: float,
    cumulative_return: float,
    cumulative_absolute_action: float,
) -> dict[str, PolicyValue]:
    reward_forward = provider_metrics["reward_forward"]
    reward_control = provider_metrics["reward_control"]
    if not math.isclose(
        reward,
        reward_forward + reward_control,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HalfCheetah reward decomposition drifted")
    expected_control_reward = -control_cost_weight * float(
        numpy.sum(numpy.square(action))
    )
    if not math.isclose(
        reward_control,
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("HalfCheetah control-cost semantics drifted")
    if not math.isclose(
        reward_forward,
        forward_reward_weight * provider_metrics["x_velocity"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HalfCheetah forward-reward semantics drifted")
    if terminated:
        raise RuntimeError("HalfCheetah unexpectedly terminated")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("HalfCheetah time-limit semantics drifted")
    if "torso_x_position" in observation and not math.isclose(
        _float_field(observation, "torso_x_position"),
        provider_metrics["x_position"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HalfCheetah x-position observation drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_forward_reward + cumulative_control_reward,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("HalfCheetah cumulative reward decomposition drifted")
    action_by_joint: dict[str, PolicyValue] = {}
    gear_scaled_controls: dict[str, PolicyValue] = {}
    for component, gear, item in zip(
        _ACTION_COMPONENTS,
        _ACTUATOR_GEARS,
        action,
        strict=True,
    ):
        value = float(item)
        action_by_joint[component] = value
        gear_scaled_controls[component] = value * gear
    seconds_per_step = frame_skip * _MODEL_TIMESTEP_SECONDS
    x_position = provider_metrics["x_position"]
    net_displacement = x_position - initial_x_position
    torso_pitch = _float_field(observation, "torso_pitch_angle")
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_action_by_joint": action_by_joint,
        "actuator_gear_scaled_controls": gear_scaled_controls,
        "sum_squared_action": float(numpy.sum(numpy.square(action))),
        "sum_absolute_action": float(numpy.sum(numpy.abs(action))),
        "cumulative_absolute_action": cumulative_absolute_action,
        "initial_x_position": initial_x_position,
        "x_position": x_position,
        "net_x_displacement": net_displacement,
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "x_velocity": provider_metrics["x_velocity"],
        "minimum_x_velocity": minimum_x_velocity,
        "maximum_x_velocity": maximum_x_velocity,
        "mean_x_velocity_from_displacement": (
            net_displacement / (step_count * seconds_per_step)
        ),
        "forward_step_fraction": forward_steps / step_count,
        "backward_or_stationary_step_fraction": 1.0 - forward_steps / step_count,
        "torso_z_position": _float_field(observation, "torso_z_position"),
        "minimum_torso_z_position": minimum_torso_z_position,
        "maximum_torso_z_position": maximum_torso_z_position,
        "torso_pitch_radians": torso_pitch,
        "torso_pitch_degrees": math.degrees(torso_pitch),
        "maximum_absolute_torso_pitch_radians": maximum_absolute_torso_pitch,
        "torso_x_velocity": _float_field(observation, "torso_x_velocity"),
        "torso_z_velocity": _float_field(observation, "torso_z_velocity"),
        "torso_pitch_angular_velocity": _float_field(
            observation,
            "torso_pitch_angular_velocity",
        ),
        "reward_forward": reward_forward,
        "reward_control": reward_control,
        "reward_from_public_terms": reward_forward + reward_control,
        "cumulative_reward_forward": cumulative_forward_reward,
        "cumulative_reward_control": cumulative_control_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"HalfCheetah returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"HalfCheetah returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"HalfCheetah returned a non-finite {name}")
    return number


__all__ = ["HalfCheetahEnvironment"]
