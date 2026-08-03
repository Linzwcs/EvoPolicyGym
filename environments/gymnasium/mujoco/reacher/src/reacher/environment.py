"""One fresh Gymnasium Reacher-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import ReacherConfig

_OBSERVATION_NAMES = (
    "joint0_cos",
    "joint1_cos",
    "joint0_sin",
    "joint1_sin",
    "target_x",
    "target_y",
    "joint0_angular_velocity",
    "joint1_angular_velocity",
    "fingertip_target_x",
    "fingertip_target_y",
)
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 200.0
_MAX_EPISODE_STEPS = 50


class ReacherEnvironment:
    """The seeded strict adapter around configured Reacher-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: ReacherConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not ReacherConfig:
            raise TypeError("config must be ReacherConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Reacher configuration belongs in ReacherConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Reacher-v5",
                frame_skip=config.frame_skip,
                reward_dist_weight=config.reward_dist_weight,
                reward_control_weight=config.reward_control_weight,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._initial_fingertip_target_distance: float | None = None
        self._minimum_fingertip_target_distance = math.inf
        self._maximum_fingertip_target_distance = -math.inf
        self._closest_approach_step = 0
        self._maximum_absolute_joint0_angular_velocity = 0.0
        self._maximum_absolute_joint1_angular_velocity = 0.0
        self._cumulative_action_squared_norm = 0.0
        self._cumulative_reward_distance = 0.0
        self._cumulative_reward_control = 0.0
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        initial_distance = _fingertip_target_distance(public_observation)
        self._initial_fingertip_target_distance = initial_distance
        self._minimum_fingertip_target_distance = initial_distance
        self._maximum_fingertip_target_distance = initial_distance
        self._started = True
        return public_observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(applied_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Reacher returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        reward_terms = _reward_metrics(info)
        initial_distance = self._initial_fingertip_target_distance
        if initial_distance is None:
            raise RuntimeError("Reacher initial diagnostics are unavailable")
        distance = _fingertip_target_distance(public_observation)
        joint0_velocity = _float_field(
            public_observation,
            "joint0_angular_velocity",
        )
        joint1_velocity = _float_field(
            public_observation,
            "joint1_angular_velocity",
        )
        self._steps += 1
        if distance < self._minimum_fingertip_target_distance:
            self._minimum_fingertip_target_distance = distance
            self._closest_approach_step = self._steps
        self._maximum_fingertip_target_distance = max(
            self._maximum_fingertip_target_distance,
            distance,
        )
        self._maximum_absolute_joint0_angular_velocity = max(
            self._maximum_absolute_joint0_angular_velocity,
            abs(joint0_velocity),
        )
        self._maximum_absolute_joint1_angular_velocity = max(
            self._maximum_absolute_joint1_angular_velocity,
            abs(joint1_velocity),
        )
        action_squared_norm = float(numpy.square(applied_action).sum())
        self._cumulative_action_squared_norm += action_squared_norm
        self._cumulative_reward_distance += reward_terms["reward_distance"]
        self._cumulative_reward_control += reward_terms["reward_control"]
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            action=applied_action,
            action_squared_norm=action_squared_norm,
            reward=public_reward,
            reward_terms=reward_terms,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            initial_distance=initial_distance,
            minimum_distance=self._minimum_fingertip_target_distance,
            maximum_distance=self._maximum_fingertip_target_distance,
            closest_approach_step=self._closest_approach_step,
            maximum_absolute_joint0_angular_velocity=(
                self._maximum_absolute_joint0_angular_velocity
            ),
            maximum_absolute_joint1_angular_velocity=(
                self._maximum_absolute_joint1_angular_velocity
            ),
            cumulative_action_squared_norm=(self._cumulative_action_squared_norm),
            cumulative_reward_distance=self._cumulative_reward_distance,
            cumulative_reward_control=self._cumulative_reward_control,
            cumulative_return=self._cumulative_return,
        )
        self._done = terminated or truncated
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


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 2:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -1.0 <= item <= 1.0:
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (10,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Reacher returned an invalid observation")
    return {
        name: _number(item, name=name) for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Reacher returned invalid reward metrics")
    if "reward_dist" not in value or "reward_ctrl" not in value:
        raise RuntimeError("Reacher omitted reward metrics")
    return {
        "reward_distance": _number(
            value["reward_dist"],
            name="reward distance",
        ),
        "reward_control": _number(
            value["reward_ctrl"],
            name="reward control",
        ),
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    *,
    action: NDArray[numpy.float32],
    action_squared_norm: float,
    reward: float,
    reward_terms: dict[str, float],
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: ReacherConfig,
    initial_distance: float,
    minimum_distance: float,
    maximum_distance: float,
    closest_approach_step: int,
    maximum_absolute_joint0_angular_velocity: float,
    maximum_absolute_joint1_angular_velocity: float,
    cumulative_action_squared_norm: float,
    cumulative_reward_distance: float,
    cumulative_reward_control: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    distance = _fingertip_target_distance(observation)
    expected_distance_reward = -config.reward_dist_weight * distance
    expected_control_reward = -config.reward_control_weight * action_squared_norm
    if not math.isclose(
        reward_terms["reward_distance"],
        expected_distance_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Reacher distance-reward semantics drifted")
    if not math.isclose(
        reward_terms["reward_control"],
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Reacher control-reward semantics drifted")
    reconstructed_reward = sum(reward_terms.values())
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Reacher reward decomposition drifted")
    if terminated:
        raise RuntimeError("Reacher natural-termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Reacher time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_reward_distance + cumulative_reward_control,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Reacher cumulative reward decomposition drifted")
    joint0_cos = _float_field(observation, "joint0_cos")
    joint1_cos = _float_field(observation, "joint1_cos")
    joint0_sin = _float_field(observation, "joint0_sin")
    joint1_sin = _float_field(observation, "joint1_sin")
    target_x = _float_field(observation, "target_x")
    target_y = _float_field(observation, "target_y")
    delta_x = _float_field(observation, "fingertip_target_x")
    delta_y = _float_field(observation, "fingertip_target_y")
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    requested_joint0 = float(action[0])
    requested_joint1 = float(action[1])
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_joint0_control": requested_joint0,
        "requested_joint1_control": requested_joint1,
        "gear_scaled_joint0_torque": requested_joint0 * _ACTUATOR_GEAR,
        "gear_scaled_joint1_torque": requested_joint1 * _ACTUATOR_GEAR,
        "action_squared_norm": action_squared_norm,
        "cumulative_action_squared_norm": cumulative_action_squared_norm,
        "mean_action_squared_norm": cumulative_action_squared_norm / step_count,
        "joint0_angle_radians": math.atan2(joint0_sin, joint0_cos),
        "joint1_relative_angle_radians": math.atan2(joint1_sin, joint1_cos),
        "joint0_unit_circle_error": abs(joint0_sin**2 + joint0_cos**2 - 1.0),
        "joint1_unit_circle_error": abs(joint1_sin**2 + joint1_cos**2 - 1.0),
        "joint0_angular_velocity": _float_field(
            observation,
            "joint0_angular_velocity",
        ),
        "joint1_angular_velocity": _float_field(
            observation,
            "joint1_angular_velocity",
        ),
        "maximum_absolute_joint0_angular_velocity": (maximum_absolute_joint0_angular_velocity),
        "maximum_absolute_joint1_angular_velocity": (maximum_absolute_joint1_angular_velocity),
        "target_x": target_x,
        "target_y": target_y,
        "target_radius": math.hypot(target_x, target_y),
        "fingertip_x": target_x + delta_x,
        "fingertip_y": target_y + delta_y,
        "fingertip_target_delta_x": delta_x,
        "fingertip_target_delta_y": delta_y,
        "initial_fingertip_target_distance": initial_distance,
        "fingertip_target_distance": distance,
        "minimum_fingertip_target_distance": minimum_distance,
        "maximum_fingertip_target_distance": maximum_distance,
        "fingertip_target_distance_reduction": initial_distance - distance,
        "closest_approach_step": closest_approach_step,
        "reward_distance": reward_terms["reward_distance"],
        "reward_control": reward_terms["reward_control"],
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_distance": cumulative_reward_distance,
        "cumulative_reward_control": cumulative_reward_control,
        "cumulative_return": cumulative_return,
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _fingertip_target_distance(
    observation: dict[str, PolicyValue],
) -> float:
    return math.hypot(
        _float_field(observation, "fingertip_target_x"),
        _float_field(observation, "fingertip_target_y"),
    )


def _float_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Reacher returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Reacher returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Reacher returned a non-finite {name}")
    return number


__all__ = ["ReacherEnvironment"]
