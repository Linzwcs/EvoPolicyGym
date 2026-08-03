"""One fresh Gymnasium LunarLander-v3 Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import LunarLanderConfig

_OBSERVATION_NAMES = (
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)
_MAX_EPISODE_STEPS = 1_000
_DISCRETE_ACTION_MEANINGS = (
    "do_nothing",
    "fire_left_orientation_engine",
    "fire_main_engine",
    "fire_right_orientation_engine",
)
_FRAMES_PER_SECOND = 50.0
_SCALE = 30.0
_VIEWPORT_WIDTH = 600.0
_VIEWPORT_HEIGHT = 400.0
_MAIN_ENGINE_POWER = 13.0
_SIDE_ENGINE_POWER = 0.6
_MAIN_ENGINE_FUEL_COEFFICIENT = 0.30
_SIDE_ENGINE_FUEL_COEFFICIENT = 0.03


class LunarLanderEnvironment:
    """The seeded strict adapter around configured LunarLander-v3."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: LunarLanderConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not LunarLanderConfig:
            raise TypeError("config must be LunarLanderConfig")
        if episode.scenario is not None:
            raise ValueError(
                "LunarLander configuration belongs in LunarLanderConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._continuous = config.continuous
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "LunarLander-v3",
                continuous=config.continuous,
                gravity=config.gravity,
                enable_wind=config.enable_wind,
                wind_power=config.wind_power,
                turbulence_power=config.turbulence_power,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._observation: dict[str, PolicyValue] | None = None
        self._steps = 0
        self._cumulative_requested_main_fuel_penalty = 0.0
        self._cumulative_requested_side_fuel_penalty = 0.0
        self._cumulative_charged_fuel_penalty = 0.0
        self._cumulative_position_shaping_delta = 0.0
        self._cumulative_velocity_shaping_delta = 0.0
        self._cumulative_angle_shaping_delta = 0.0
        self._cumulative_contact_shaping_delta = 0.0
        self._cumulative_terminal_override = 0.0
        self._cumulative_return = 0.0
        self._main_engine_firing_steps = 0
        self._side_engine_firing_steps = 0
        self._minimum_landing_state_penalty = math.inf

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        self._observation = public_observation
        self._minimum_landing_state_penalty = _landing_state_penalty(
            public_observation
        )
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
            raise RuntimeError("LunarLander observation is unavailable")
        applied = (
            _continuous_action(action)
            if self._continuous
            else _discrete_action(action)
        )
        action_meaning, main_power, side_power, side_engine = _action_effect(
            applied,
            continuous=self._continuous,
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            applied
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "LunarLander returned invalid termination flags"
            )
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        self._steps += 1
        if truncated != (self._steps == _MAX_EPISODE_STEPS):
            raise RuntimeError("LunarLander time-limit semantics drifted")
        (
            requested_main_fuel_penalty,
            requested_side_fuel_penalty,
            charged_fuel_penalty,
            position_shaping_delta,
            velocity_shaping_delta,
            angle_shaping_delta,
            contact_shaping_delta,
            terminal_override,
            reward_was_overridden,
        ) = _reward_components(
            previous_observation,
            public_observation,
            main_power=main_power,
            side_power=side_power,
            reward=public_reward,
            terminated=terminated,
        )
        self._cumulative_requested_main_fuel_penalty += (
            requested_main_fuel_penalty
        )
        self._cumulative_requested_side_fuel_penalty += (
            requested_side_fuel_penalty
        )
        self._cumulative_charged_fuel_penalty += charged_fuel_penalty
        self._cumulative_position_shaping_delta += position_shaping_delta
        self._cumulative_velocity_shaping_delta += velocity_shaping_delta
        self._cumulative_angle_shaping_delta += angle_shaping_delta
        self._cumulative_contact_shaping_delta += contact_shaping_delta
        self._cumulative_terminal_override += terminal_override
        self._cumulative_return += public_reward
        self._main_engine_firing_steps += int(main_power > 0.0)
        self._side_engine_firing_steps += int(side_power > 0.0)
        self._minimum_landing_state_penalty = min(
            self._minimum_landing_state_penalty,
            _landing_state_penalty(public_observation),
        )
        metrics = _transition_metrics(
            previous_observation,
            public_observation,
            applied_action=applied,
            continuous=self._continuous,
            action_meaning=action_meaning,
            main_power=main_power,
            side_power=side_power,
            side_engine=side_engine,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            requested_main_fuel_penalty=requested_main_fuel_penalty,
            requested_side_fuel_penalty=requested_side_fuel_penalty,
            charged_fuel_penalty=charged_fuel_penalty,
            position_shaping_delta=position_shaping_delta,
            velocity_shaping_delta=velocity_shaping_delta,
            angle_shaping_delta=angle_shaping_delta,
            contact_shaping_delta=contact_shaping_delta,
            terminal_override=terminal_override,
            reward_was_overridden=reward_was_overridden,
            cumulative_requested_main_fuel_penalty=(
                self._cumulative_requested_main_fuel_penalty
            ),
            cumulative_requested_side_fuel_penalty=(
                self._cumulative_requested_side_fuel_penalty
            ),
            cumulative_charged_fuel_penalty=self._cumulative_charged_fuel_penalty,
            cumulative_position_shaping_delta=(
                self._cumulative_position_shaping_delta
            ),
            cumulative_velocity_shaping_delta=(
                self._cumulative_velocity_shaping_delta
            ),
            cumulative_angle_shaping_delta=self._cumulative_angle_shaping_delta,
            cumulative_contact_shaping_delta=(
                self._cumulative_contact_shaping_delta
            ),
            cumulative_terminal_override=self._cumulative_terminal_override,
            cumulative_return=self._cumulative_return,
            main_engine_firing_steps=self._main_engine_firing_steps,
            side_engine_firing_steps=self._side_engine_firing_steps,
            minimum_landing_state_penalty=self._minimum_landing_state_penalty,
        )
        self._observation = public_observation
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


def _discrete_action(value: PolicyValue) -> int:
    if type(value) is not int or value not in {0, 1, 2, 3}:
        raise InvalidAction()
    return value


def _continuous_action(value: PolicyValue) -> list[float]:
    if type(value) is not list or len(value) != 2:
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
        raise RuntimeError("LunarLander returned an invalid observation")
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError(
            "LunarLander returned an invalid observation shape"
        )
    return {
        "x_position": _number(items[0], name="x position"),
        "y_position": _number(items[1], name="y position"),
        "x_velocity": _number(items[2], name="x velocity"),
        "y_velocity": _number(items[3], name="y velocity"),
        "angle": _number(items[4], name="angle"),
        "angular_velocity": _number(
            items[5],
            name="angular velocity",
        ),
        "left_leg_contact": _contact(items[6], name="left leg contact"),
        "right_leg_contact": _contact(
            items[7],
            name="right leg contact",
        ),
    }


def _contact(value: object, *, name: str) -> bool:
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"LunarLander returned an invalid {name}")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"LunarLander returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"LunarLander returned a non-finite {name}")
    return number


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    *,
    applied_action: int | list[float],
    continuous: bool,
    action_meaning: str,
    main_power: float,
    side_power: float,
    side_engine: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    requested_main_fuel_penalty: float,
    requested_side_fuel_penalty: float,
    charged_fuel_penalty: float,
    position_shaping_delta: float,
    velocity_shaping_delta: float,
    angle_shaping_delta: float,
    contact_shaping_delta: float,
    terminal_override: float,
    reward_was_overridden: bool,
    cumulative_requested_main_fuel_penalty: float,
    cumulative_requested_side_fuel_penalty: float,
    cumulative_charged_fuel_penalty: float,
    cumulative_position_shaping_delta: float,
    cumulative_velocity_shaping_delta: float,
    cumulative_angle_shaping_delta: float,
    cumulative_contact_shaping_delta: float,
    cumulative_terminal_override: float,
    cumulative_return: float,
    main_engine_firing_steps: int,
    side_engine_firing_steps: int,
    minimum_landing_state_penalty: float,
) -> dict[str, PolicyValue]:
    x_position = _float_field(current, "x_position")
    y_position = _float_field(current, "y_position")
    x_velocity = _float_field(current, "x_velocity")
    y_velocity = _float_field(current, "y_velocity")
    angle = _float_field(current, "angle")
    angular_velocity = _float_field(current, "angular_velocity")
    left_contact = _bool_field(current, "left_leg_contact")
    right_contact = _bool_field(current, "right_leg_contact")
    previous_left_contact = _bool_field(previous, "left_leg_contact")
    previous_right_contact = _bool_field(previous, "right_leg_contact")
    terminal_reason = "none"
    if terminated and reward == 100.0:
        terminal_reason = "settled_landing"
    elif terminated and abs(x_position) >= 1.0:
        terminal_reason = "left_viewport"
    elif terminated:
        terminal_reason = "crash"
    elif truncated:
        terminal_reason = "time_limit"

    reconstructed_cumulative_return = (
        cumulative_position_shaping_delta
        + cumulative_velocity_shaping_delta
        + cumulative_angle_shaping_delta
        + cumulative_contact_shaping_delta
        - cumulative_charged_fuel_penalty
        + cumulative_terminal_override
    )
    if not math.isclose(
        cumulative_return,
        reconstructed_cumulative_return,
        rel_tol=0.0,
        abs_tol=5e-5,
    ):
        raise RuntimeError("LunarLander cumulative reward decomposition drifted")
    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "simulated_seconds": step_count / _FRAMES_PER_SECOND,
        "action_mode": "continuous" if continuous else "discrete",
        "requested_action_meaning": action_meaning,
        "main_engine_active": main_power > 0.0,
        "main_engine_power_fraction": main_power,
        "main_engine_impulse_scale": main_power * _MAIN_ENGINE_POWER,
        "side_engine_active": side_power > 0.0,
        "side_engine_power_fraction": side_power,
        "side_engine_direction": side_engine,
        "side_engine_impulse_scale": side_power * _SIDE_ENGINE_POWER,
        "requested_main_engine_fuel_penalty": requested_main_fuel_penalty,
        "requested_side_engine_fuel_penalty": requested_side_fuel_penalty,
        "charged_fuel_penalty": charged_fuel_penalty,
        "reward_was_terminal_override": reward_was_overridden,
        "position_shaping_delta": position_shaping_delta,
        "velocity_shaping_delta": velocity_shaping_delta,
        "angle_shaping_delta": angle_shaping_delta,
        "leg_contact_shaping_delta": contact_shaping_delta,
        "terminal_override_reward": terminal_override,
        "reward_from_public_terms": (
            position_shaping_delta
            + velocity_shaping_delta
            + angle_shaping_delta
            + contact_shaping_delta
            - charged_fuel_penalty
            + terminal_override
        ),
        "cumulative_requested_main_engine_fuel_penalty": (
            cumulative_requested_main_fuel_penalty
        ),
        "cumulative_requested_side_engine_fuel_penalty": (
            cumulative_requested_side_fuel_penalty
        ),
        "cumulative_charged_fuel_penalty": cumulative_charged_fuel_penalty,
        "cumulative_position_shaping_delta": cumulative_position_shaping_delta,
        "cumulative_velocity_shaping_delta": cumulative_velocity_shaping_delta,
        "cumulative_angle_shaping_delta": cumulative_angle_shaping_delta,
        "cumulative_leg_contact_shaping_delta": cumulative_contact_shaping_delta,
        "cumulative_terminal_override_reward": cumulative_terminal_override,
        "cumulative_return": cumulative_return,
        "main_engine_firing_steps": main_engine_firing_steps,
        "side_engine_firing_steps": side_engine_firing_steps,
        "normalized_x_position_from_pad_center": x_position,
        "normalized_y_position_from_landing_target": y_position,
        "normalized_distance_to_landing_target": math.hypot(x_position, y_position),
        "normalized_x_velocity": x_velocity,
        "normalized_y_velocity": y_velocity,
        "normalized_speed": math.hypot(x_velocity, y_velocity),
        "landing_state_penalty": _landing_state_penalty(current),
        "minimum_landing_state_penalty": minimum_landing_state_penalty,
        "horizontal_displacement_from_pad_center_world_units": (
            x_position * (_VIEWPORT_WIDTH / _SCALE / 2.0)
        ),
        "vertical_displacement_from_target_world_units": (
            y_position * (_VIEWPORT_HEIGHT / _SCALE / 2.0)
        ),
        "horizontal_velocity_world_units_per_second": (
            x_velocity * _FRAMES_PER_SECOND / (_VIEWPORT_WIDTH / _SCALE / 2.0)
        ),
        "vertical_velocity_world_units_per_second": (
            y_velocity * _FRAMES_PER_SECOND / (_VIEWPORT_HEIGHT / _SCALE / 2.0)
        ),
        "angle_radians": angle,
        "angle_degrees": math.degrees(angle),
        "absolute_angle_radians": abs(angle),
        "normalized_angular_velocity": angular_velocity,
        "angular_velocity_radians_per_second": (
            angular_velocity * _FRAMES_PER_SECOND / 20.0
        ),
        "left_leg_contact": left_contact,
        "right_leg_contact": right_contact,
        "left_leg_contact_started": left_contact and not previous_left_contact,
        "left_leg_contact_ended": previous_left_contact and not left_contact,
        "right_leg_contact_started": right_contact and not previous_right_contact,
        "right_leg_contact_ended": previous_right_contact and not right_contact,
        "contacted_leg_count": int(left_contact) + int(right_contact),
        "support_state": _support_state(left_contact, right_contact),
        "inside_helipad_horizontal_span": abs(x_position) <= 0.2,
        "inside_horizontal_viewport": abs(x_position) < 1.0,
        "terminal_reason": terminal_reason,
    }
    if continuous:
        if type(applied_action) is not list:
            raise RuntimeError("continuous LunarLander Action is unavailable")
        metrics["main_engine_command"] = applied_action[0]
        metrics["lateral_engine_command"] = applied_action[1]
    else:
        if type(applied_action) is not int:
            raise RuntimeError("discrete LunarLander Action is unavailable")
        metrics["discrete_action"] = applied_action
    return metrics


def _action_effect(
    action: int | list[float],
    *,
    continuous: bool,
) -> tuple[str, float, float, str]:
    if continuous:
        if type(action) is not list or len(action) != 2:
            raise RuntimeError("continuous LunarLander Action is invalid")
        main, lateral = action
        main_power = (main + 1.0) * 0.5 if main > 0.0 else 0.0
        if lateral < -0.5:
            return "continuous_throttle", main_power, abs(lateral), "left"
        if lateral > 0.5:
            return "continuous_throttle", main_power, abs(lateral), "right"
        return "continuous_throttle", main_power, 0.0, "none"
    if type(action) is not int:
        raise RuntimeError("discrete LunarLander Action is invalid")
    return (
        _DISCRETE_ACTION_MEANINGS[action],
        1.0 if action == 2 else 0.0,
        1.0 if action in {1, 3} else 0.0,
        "left" if action == 1 else "right" if action == 3 else "none",
    )


def _reward_components(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    *,
    main_power: float,
    side_power: float,
    reward: float,
    terminated: bool,
) -> tuple[float, float, float, float, float, float, float, float, bool]:
    requested_main_fuel_penalty = main_power * _MAIN_ENGINE_FUEL_COEFFICIENT
    requested_side_fuel_penalty = side_power * _SIDE_ENGINE_FUEL_COEFFICIENT
    previous_terms = _shaping_terms(previous)
    current_terms = _shaping_terms(current)
    geometric_deltas = tuple(
        current_value - previous_value
        for previous_value, current_value in zip(
            previous_terms,
            current_terms,
            strict=True,
        )
    )
    reward_was_overridden = terminated and reward in {-100.0, 100.0}
    if terminated and not reward_was_overridden:
        raise RuntimeError("LunarLander terminal reward semantics drifted")
    if reward_was_overridden:
        charged_fuel_penalty = 0.0
        position_delta = 0.0
        velocity_delta = 0.0
        angle_delta = 0.0
        contact_delta = 0.0
        terminal_override = reward
    else:
        charged_fuel_penalty = (
            requested_main_fuel_penalty + requested_side_fuel_penalty
        )
        position_delta, velocity_delta, angle_delta, contact_delta = geometric_deltas
        terminal_override = 0.0
    reconstructed_reward = (
        position_delta
        + velocity_delta
        + angle_delta
        + contact_delta
        - charged_fuel_penalty
        + terminal_override
    )
    if not math.isclose(reward, reconstructed_reward, rel_tol=0.0, abs_tol=3e-5):
        raise RuntimeError("LunarLander reward decomposition drifted")
    return (
        requested_main_fuel_penalty,
        requested_side_fuel_penalty,
        charged_fuel_penalty,
        position_delta,
        velocity_delta,
        angle_delta,
        contact_delta,
        terminal_override,
        reward_was_overridden,
    )


def _shaping_terms(
    observation: dict[str, PolicyValue],
) -> tuple[float, float, float, float]:
    x_position = _float_field(observation, "x_position")
    y_position = _float_field(observation, "y_position")
    x_velocity = _float_field(observation, "x_velocity")
    y_velocity = _float_field(observation, "y_velocity")
    angle = _float_field(observation, "angle")
    left_contact = _bool_field(observation, "left_leg_contact")
    right_contact = _bool_field(observation, "right_leg_contact")
    return (
        -100.0 * math.hypot(x_position, y_position),
        -100.0 * math.hypot(x_velocity, y_velocity),
        -100.0 * abs(angle),
        10.0 * (int(left_contact) + int(right_contact)),
    )


def _landing_state_penalty(observation: dict[str, PolicyValue]) -> float:
    position, velocity, angle, _ = _shaping_terms(observation)
    return -(position + velocity + angle)


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"LunarLander returned invalid {name}")
    return value


def _bool_field(observation: dict[str, PolicyValue], name: str) -> bool:
    value = observation.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"LunarLander returned invalid {name}")
    return value


def _support_state(left_contact: bool, right_contact: bool) -> str:
    if left_contact and right_contact:
        return "both_legs"
    if left_contact:
        return "left_leg_only"
    if right_contact:
        return "right_leg_only"
    return "airborne"


__all__ = ["LunarLanderEnvironment"]
