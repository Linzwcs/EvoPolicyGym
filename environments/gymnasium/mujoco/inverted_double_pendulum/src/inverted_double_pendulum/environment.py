"""One fresh InvertedDoublePendulum-v5 Environment per Episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import InvertedDoublePendulumConfig

_OBSERVATION_NAMES = (
    "cart_position",
    "pole1_sin",
    "pole2_relative_sin",
    "pole1_cos",
    "pole2_relative_cos",
    "cart_velocity",
    "pole1_angular_velocity",
    "pole2_relative_angular_velocity",
    "cart_constraint_force",
)
_MODEL_TIMESTEP_SECONDS = 0.01
_POLE_LENGTH_METERS = 0.6
_REWARD_TARGET_TIP_HEIGHT = 2.0
_TERMINATION_TIP_HEIGHT = 1.0
_ACTUATOR_GEAR = 500.0
_MAX_EPISODE_STEPS = 1_000


class _MujocoData(Protocol):
    qvel: NDArray[numpy.float64]
    site_xpos: NDArray[numpy.float64]


class _DoublePendulumProvider(Protocol):
    data: _MujocoData


@dataclass(frozen=True)
class _ExactState:
    tip_x_position: float
    tip_y_position: float
    pole1_angular_velocity: float
    pole2_relative_angular_velocity: float


class InvertedDoublePendulumEnvironment:
    """Strict adapter around configured InvertedDoublePendulum-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: InvertedDoublePendulumConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not InvertedDoublePendulumConfig:
            raise TypeError("config must be InvertedDoublePendulumConfig")
        if episode.scenario is not None:
            raise ValueError(
                "InvertedDoublePendulum configuration belongs in "
                "InvertedDoublePendulumConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "InvertedDoublePendulum-v5",
                frame_skip=config.frame_skip,
                healthy_reward=config.healthy_reward,
                reset_noise_scale=config.reset_noise_scale,
            ),
        )
        self._provider = cast(
            _DoublePendulumProvider,
            self._environment.unwrapped,
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._minimum_cart_position = math.inf
        self._maximum_cart_position = -math.inf
        self._minimum_tip_height = math.inf
        self._maximum_tip_height = -math.inf
        self._minimum_tip_height_margin = math.inf
        self._maximum_absolute_tip_x_position = 0.0
        self._maximum_absolute_pole1_angle = 0.0
        self._maximum_absolute_pole2_absolute_angle = 0.0
        self._maximum_observed_absolute_pole_angular_velocity = 0.0
        self._velocity_clip_limit_steps = 0
        self._cumulative_absolute_action = 0.0
        self._cumulative_survival_reward = 0.0
        self._cumulative_distance_penalty_reward = 0.0
        self._cumulative_velocity_penalty_reward = 0.0
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation)

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
            raise RuntimeError("InvertedDoublePendulum returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        exact_state = _exact_state(self._provider)
        derived = _derived_state(public_observation)
        self._steps += 1
        cart_position = _float_field(public_observation, "cart_position")
        pole1_angle = derived["pole1_angle_radians"]
        pole2_absolute_angle = derived["pole2_absolute_angle_radians"]
        observed_pole1_velocity = _float_field(
            public_observation,
            "pole1_angular_velocity",
        )
        observed_pole2_velocity = _float_field(
            public_observation,
            "pole2_relative_angular_velocity",
        )
        velocity_at_clip_limit = (
            abs(observed_pole1_velocity) == 10.0 or abs(observed_pole2_velocity) == 10.0
        )
        tip_height_margin = exact_state.tip_y_position - _TERMINATION_TIP_HEIGHT
        self._minimum_cart_position = min(
            self._minimum_cart_position,
            cart_position,
        )
        self._maximum_cart_position = max(
            self._maximum_cart_position,
            cart_position,
        )
        self._minimum_tip_height = min(
            self._minimum_tip_height,
            exact_state.tip_y_position,
        )
        self._maximum_tip_height = max(
            self._maximum_tip_height,
            exact_state.tip_y_position,
        )
        self._minimum_tip_height_margin = min(
            self._minimum_tip_height_margin,
            tip_height_margin,
        )
        self._maximum_absolute_tip_x_position = max(
            self._maximum_absolute_tip_x_position,
            abs(exact_state.tip_x_position),
        )
        self._maximum_absolute_pole1_angle = max(
            self._maximum_absolute_pole1_angle,
            abs(pole1_angle),
        )
        self._maximum_absolute_pole2_absolute_angle = max(
            self._maximum_absolute_pole2_absolute_angle,
            abs(pole2_absolute_angle),
        )
        self._maximum_observed_absolute_pole_angular_velocity = max(
            self._maximum_observed_absolute_pole_angular_velocity,
            abs(observed_pole1_velocity),
            abs(observed_pole2_velocity),
        )
        self._velocity_clip_limit_steps += int(velocity_at_clip_limit)
        self._cumulative_absolute_action += abs(float(applied_action[0]))
        self._cumulative_survival_reward += provider_metrics["reward_survive"]
        self._cumulative_distance_penalty_reward += provider_metrics["reward_distance_penalty"]
        self._cumulative_velocity_penalty_reward += provider_metrics["reward_velocity_penalty"]
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            applied_action,
            provider_metrics=provider_metrics,
            exact_state=exact_state,
            derived=derived,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            minimum_cart_position=self._minimum_cart_position,
            maximum_cart_position=self._maximum_cart_position,
            minimum_tip_height=self._minimum_tip_height,
            maximum_tip_height=self._maximum_tip_height,
            minimum_tip_height_margin=self._minimum_tip_height_margin,
            maximum_absolute_tip_x_position=(self._maximum_absolute_tip_x_position),
            maximum_absolute_pole1_angle=(self._maximum_absolute_pole1_angle),
            maximum_absolute_pole2_absolute_angle=(self._maximum_absolute_pole2_absolute_angle),
            maximum_observed_absolute_pole_angular_velocity=(
                self._maximum_observed_absolute_pole_angular_velocity
            ),
            velocity_clip_limit_steps=self._velocity_clip_limit_steps,
            cumulative_absolute_action=self._cumulative_absolute_action,
            cumulative_survival_reward=self._cumulative_survival_reward,
            cumulative_distance_penalty_reward=(self._cumulative_distance_penalty_reward),
            cumulative_velocity_penalty_reward=(self._cumulative_velocity_penalty_reward),
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
    if type(value) is not list or len(value) != 1:
        raise InvalidAction()
    item = value[0]
    if type(item) is not float or not math.isfinite(item) or not -1.0 <= item <= 1.0:
        raise InvalidAction()
    return numpy.asarray([item], dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (9,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("InvertedDoublePendulum returned an invalid observation")
    return {
        name: _number(item, name=name) for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("InvertedDoublePendulum returned invalid reward metrics")
    provider_names = (
        "reward_survive",
        "distance_penalty",
        "velocity_penalty",
    )
    if not set(provider_names).issubset(value):
        raise RuntimeError("InvertedDoublePendulum omitted reward metrics")
    return {
        "reward_survive": _number(
            value["reward_survive"],
            name="reward survive",
        ),
        "reward_distance_penalty": _number(
            value["distance_penalty"],
            name="distance penalty reward",
        ),
        "reward_velocity_penalty": _number(
            value["velocity_penalty"],
            name="velocity penalty reward",
        ),
    }


def _exact_state(provider: _DoublePendulumProvider) -> _ExactState:
    qvel = provider.data.qvel
    site_xpos = provider.data.site_xpos
    if (
        type(qvel) is not numpy.ndarray
        or qvel.shape != (3,)
        or qvel.dtype != numpy.dtype("float64")
        or type(site_xpos) is not numpy.ndarray
        or site_xpos.shape != (1, 3)
        or site_xpos.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("InvertedDoublePendulum returned an invalid internal state")
    return _ExactState(
        tip_x_position=_number(site_xpos[0, 0], name="tip x position"),
        tip_y_position=_number(site_xpos[0, 2], name="tip y position"),
        pole1_angular_velocity=_number(
            qvel[1],
            name="unclipped pole1 angular velocity",
        ),
        pole2_relative_angular_velocity=_number(
            qvel[2],
            name="unclipped pole2 relative angular velocity",
        ),
    )


def _derived_state(observation: dict[str, PolicyValue]) -> dict[str, float]:
    cart_position = _float_field(observation, "cart_position")
    pole1_sin = _float_field(observation, "pole1_sin")
    pole1_cos = _float_field(observation, "pole1_cos")
    pole2_sin = _float_field(observation, "pole2_relative_sin")
    pole2_cos = _float_field(observation, "pole2_relative_cos")
    pole1_angle = math.atan2(pole1_sin, pole1_cos)
    pole2_relative_angle = math.atan2(pole2_sin, pole2_cos)
    pole2_absolute_sin = pole1_sin * pole2_cos + pole1_cos * pole2_sin
    pole2_absolute_cos = pole1_cos * pole2_cos - pole1_sin * pole2_sin
    pole2_absolute_angle = math.atan2(
        pole2_absolute_sin,
        pole2_absolute_cos,
    )
    tip_x = cart_position + _POLE_LENGTH_METERS * (pole1_sin + pole2_absolute_sin)
    tip_y = _POLE_LENGTH_METERS * (pole1_cos + pole2_absolute_cos)
    return {
        "pole1_angle_radians": pole1_angle,
        "pole2_relative_angle_radians": pole2_relative_angle,
        "pole2_absolute_angle_radians": pole2_absolute_angle,
        "tip_x_position_from_observation": tip_x,
        "tip_y_position_from_observation": tip_y,
        "pole1_unit_circle_error": abs(pole1_sin**2 + pole1_cos**2 - 1.0),
        "pole2_relative_unit_circle_error": abs(pole2_sin**2 + pole2_cos**2 - 1.0),
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, float],
    exact_state: _ExactState,
    derived: dict[str, float],
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: InvertedDoublePendulumConfig,
    minimum_cart_position: float,
    maximum_cart_position: float,
    minimum_tip_height: float,
    maximum_tip_height: float,
    minimum_tip_height_margin: float,
    maximum_absolute_tip_x_position: float,
    maximum_absolute_pole1_angle: float,
    maximum_absolute_pole2_absolute_angle: float,
    maximum_observed_absolute_pole_angular_velocity: float,
    velocity_clip_limit_steps: int,
    cumulative_absolute_action: float,
    cumulative_survival_reward: float,
    cumulative_distance_penalty_reward: float,
    cumulative_velocity_penalty_reward: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    reward_survive = provider_metrics["reward_survive"]
    reward_distance = provider_metrics["reward_distance_penalty"]
    reward_velocity = provider_metrics["reward_velocity_penalty"]
    reconstructed_reward = reward_survive + reward_distance + reward_velocity
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("InvertedDoublePendulum reward decomposition drifted")
    expected_distance = -(
        0.01 * exact_state.tip_x_position**2
        + (exact_state.tip_y_position - _REWARD_TARGET_TIP_HEIGHT) ** 2
    )
    expected_velocity = -(
        1e-3 * exact_state.pole1_angular_velocity**2
        + 5e-3 * exact_state.pole2_relative_angular_velocity**2
    )
    if not math.isclose(
        reward_distance,
        expected_distance,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("InvertedDoublePendulum distance-penalty semantics drifted")
    if not math.isclose(
        reward_velocity,
        expected_velocity,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("InvertedDoublePendulum velocity-penalty semantics drifted")
    healthy = exact_state.tip_y_position > _TERMINATION_TIP_HEIGHT
    expected_survival = config.healthy_reward if healthy else 0.0
    if not math.isclose(
        reward_survive,
        expected_survival,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("InvertedDoublePendulum survival-reward semantics drifted")
    if terminated != (not healthy):
        raise RuntimeError("InvertedDoublePendulum health termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("InvertedDoublePendulum time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_survival_reward
        + cumulative_distance_penalty_reward
        + cumulative_velocity_penalty_reward,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("InvertedDoublePendulum cumulative reward decomposition drifted")
    requested_action = float(action[0])
    observed_pole1_velocity = _float_field(
        observation,
        "pole1_angular_velocity",
    )
    observed_pole2_velocity = _float_field(
        observation,
        "pole2_relative_angular_velocity",
    )
    velocity_at_clip_limit = (
        abs(observed_pole1_velocity) == 10.0 or abs(observed_pole2_velocity) == 10.0
    )
    terminal_reason = "none"
    if terminated and truncated:
        terminal_reason = "fallen_and_time_limit"
    elif terminated:
        terminal_reason = "fallen"
    elif truncated:
        terminal_reason = "time_limit"
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_cart_control": requested_action,
        "actuator_gear_scaled_cart_force": requested_action * _ACTUATOR_GEAR,
        "cumulative_absolute_action": cumulative_absolute_action,
        "cart_position": _float_field(observation, "cart_position"),
        "minimum_cart_position": minimum_cart_position,
        "maximum_cart_position": maximum_cart_position,
        "cart_velocity": _float_field(observation, "cart_velocity"),
        "pole1_angle_radians": derived["pole1_angle_radians"],
        "pole2_relative_angle_radians": derived["pole2_relative_angle_radians"],
        "pole2_absolute_angle_radians": derived["pole2_absolute_angle_radians"],
        "maximum_absolute_pole1_angle_radians": maximum_absolute_pole1_angle,
        "maximum_absolute_pole2_absolute_angle_radians": (maximum_absolute_pole2_absolute_angle),
        "pole1_angular_velocity_observed": observed_pole1_velocity,
        "pole2_relative_angular_velocity_observed": observed_pole2_velocity,
        "maximum_observed_absolute_pole_angular_velocity": (
            maximum_observed_absolute_pole_angular_velocity
        ),
        "velocity_observation_at_clip_limit": velocity_at_clip_limit,
        "velocity_clip_limit_step_fraction": (velocity_clip_limit_steps / step_count),
        "tip_x_position": exact_state.tip_x_position,
        "tip_y_position": exact_state.tip_y_position,
        "tip_x_position_from_observation": derived["tip_x_position_from_observation"],
        "tip_y_position_from_observation": derived["tip_y_position_from_observation"],
        "tip_position_reconstruction_error": math.hypot(
            exact_state.tip_x_position - derived["tip_x_position_from_observation"],
            exact_state.tip_y_position - derived["tip_y_position_from_observation"],
        ),
        "minimum_tip_y_position": minimum_tip_height,
        "maximum_tip_y_position": maximum_tip_height,
        "tip_height_termination_threshold": _TERMINATION_TIP_HEIGHT,
        "tip_height_margin": (exact_state.tip_y_position - _TERMINATION_TIP_HEIGHT),
        "minimum_tip_height_margin": minimum_tip_height_margin,
        "maximum_absolute_tip_x_position": maximum_absolute_tip_x_position,
        "reward_target_tip_height": _REWARD_TARGET_TIP_HEIGHT,
        "maximum_physical_tip_height": 2.0 * _POLE_LENGTH_METERS,
        "unavoidable_upright_distance_penalty": (
            -((_REWARD_TARGET_TIP_HEIGHT - 2.0 * _POLE_LENGTH_METERS) ** 2)
        ),
        "pole1_unit_circle_error": derived["pole1_unit_circle_error"],
        "pole2_relative_unit_circle_error": derived["pole2_relative_unit_circle_error"],
        "reward_survive": reward_survive,
        "reward_distance_penalty": reward_distance,
        "reward_velocity_penalty": reward_velocity,
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_survive": cumulative_survival_reward,
        "cumulative_reward_distance_penalty": (cumulative_distance_penalty_reward),
        "cumulative_reward_velocity_penalty": (cumulative_velocity_penalty_reward),
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"InvertedDoublePendulum returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"InvertedDoublePendulum returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"InvertedDoublePendulum returned a non-finite {name}")
    return number


__all__ = ["InvertedDoublePendulumEnvironment"]
