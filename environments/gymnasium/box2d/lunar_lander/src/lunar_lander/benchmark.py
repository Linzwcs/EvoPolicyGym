"""A parameterized LunarLander-v3 Benchmark with public traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import LunarLanderConfig
from .environment import LunarLanderEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-lunar-lander/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_DISCRETE_ACTION_MEANINGS = (
    "do_nothing",
    "fire_left_orientation_engine",
    "fire_main_engine",
    "fire_right_orientation_engine",
)
_OBSERVATION_FIELDS = (
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    requested_main_fuel_penalty: float
    requested_side_fuel_penalty: float
    charged_fuel_penalty: float
    position_shaping_delta: float
    velocity_shaping_delta: float
    angle_shaping_delta: float
    contact_shaping_delta: float
    terminal_override_reward: float
    minimum_landing_state_penalty: float
    closest_normalized_pad_distance: float
    closest_normalized_speed: float
    closest_absolute_angle_radians: float
    final_normalized_pad_distance: float
    final_normalized_speed: float
    final_absolute_angle_radians: float
    main_engine_firing_fraction: float
    side_engine_firing_fraction: float
    left_leg_contact_fraction: float
    right_leg_contact_fraction: float


class LunarLanderBenchmark:
    """Mean LunarLander return over deterministic Episode plans."""

    def __init__(self, config: LunarLanderConfig | None = None) -> None:
        if config is None:
            config = LunarLanderConfig()
        if type(config) is not LunarLanderConfig:
            raise TypeError("config must be LunarLanderConfig")
        self._config = config
        self._spec = _benchmark_spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return LunarLanderEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            (
                record.total_reward
                if record.policy_failure is None
                else _FAILURE_RETURN
            )
            for record in records
        )
        score = statistics.fmean(returns)
        landings = sum(_landed(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        outcomes = tuple(_episode_outcome(record) for record in records)
        diagnostics = tuple(
            _episode_diagnostics(record, continuous=self._config.continuous)
            for record in records
        )
        crashes = outcomes.count("crash")
        viewport_exits = sum(
            outcome == "left_viewport" for outcome in outcomes
        )
        time_limits = outcomes.count("time_limit")
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{landings} successful landings."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "successful_landings": landings,
                "crash_episodes": crashes,
                "left_viewport_episodes": viewport_exits,
                "time_limit_episodes": time_limits,
                "mean_minimum_landing_state_penalty": statistics.fmean(
                    item.minimum_landing_state_penalty for item in diagnostics
                ),
                "mean_closest_normalized_pad_distance": statistics.fmean(
                    item.closest_normalized_pad_distance for item in diagnostics
                ),
                "mean_closest_normalized_speed": statistics.fmean(
                    item.closest_normalized_speed for item in diagnostics
                ),
                "mean_closest_absolute_angle_radians": statistics.fmean(
                    item.closest_absolute_angle_radians for item in diagnostics
                ),
                "mean_final_normalized_pad_distance": statistics.fmean(
                    item.final_normalized_pad_distance for item in diagnostics
                ),
                "mean_final_normalized_speed": statistics.fmean(
                    item.final_normalized_speed for item in diagnostics
                ),
                "mean_final_absolute_angle_radians": statistics.fmean(
                    item.final_absolute_angle_radians for item in diagnostics
                ),
                "mean_episode_requested_main_engine_fuel_penalty": statistics.fmean(
                    item.requested_main_fuel_penalty for item in diagnostics
                ),
                "mean_episode_requested_side_engine_fuel_penalty": statistics.fmean(
                    item.requested_side_fuel_penalty for item in diagnostics
                ),
                "mean_episode_charged_fuel_penalty": statistics.fmean(
                    item.charged_fuel_penalty for item in diagnostics
                ),
                "mean_episode_position_shaping_delta": statistics.fmean(
                    item.position_shaping_delta for item in diagnostics
                ),
                "mean_episode_velocity_shaping_delta": statistics.fmean(
                    item.velocity_shaping_delta for item in diagnostics
                ),
                "mean_episode_angle_shaping_delta": statistics.fmean(
                    item.angle_shaping_delta for item in diagnostics
                ),
                "mean_episode_leg_contact_shaping_delta": statistics.fmean(
                    item.contact_shaping_delta for item in diagnostics
                ),
                "mean_main_engine_firing_fraction": statistics.fmean(
                    item.main_engine_firing_fraction for item in diagnostics
                ),
                "mean_side_engine_firing_fraction": statistics.fmean(
                    item.side_engine_firing_fraction for item in diagnostics
                ),
                "mean_left_leg_contact_fraction": statistics.fmean(
                    item.left_leg_contact_fraction for item in diagnostics
                ),
                "mean_right_leg_contact_fraction": statistics.fmean(
                    item.right_leg_contact_fraction for item in diagnostics
                ),
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    continuous=self._config.continuous,
                ),
            ),
        )


def _benchmark_spec(config: LunarLanderConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        action_space = {
            "type": "array",
            "shape": [2],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": [
                "main_engine",
                "lateral_engine",
            ],
            "policy_carrier": "list[float]",
            "notes": (
                "Main <=0 is off; main >0 maps discontinuously to power "
                "(main+1)/2, from just over 50% through 100%. Lateral is off "
                "for abs(value)<=0.5; values <-0.5 fire left and >0.5 fire "
                "right at power abs(value)."
            ),
        }
    else:
        action_space = {
            "type": "discrete",
            "values": [0, 1, 2, 3],
            "meaning": {
                "0": "do_nothing",
                "1": "fire_left_orientation_engine",
                "2": "fire_main_engine",
                "3": "fire_right_orientation_engine",
            },
        }
    return BenchmarkSpec(
        id="gymnasium/LunarLander-v3/mean-return-v1",
        description=(
            "Control a lunar lander from a randomized initial impulse to a "
            "safe landing pad centered at normalized coordinate (0,0). Balance "
            "normalized position and velocity, angle, leg contact, and engine "
            "cost. Each normal reward is the change in public shaping "
            "[-100*position_distance-100*speed-100*abs(angle)+10 per contacting "
            "leg], minus 0.30*main_engine_power and 0.03*side_engine_power. A "
            "safe settled landing overrides reward to +100; a crash or leaving "
            "the horizontal viewport overrides it to -100. The TimeLimit is "
            "1,000 steps. Maximize mean Episode return."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float | bool]",
            "source_dtype": "float32",
            "fields": {
                "x_position": {
                    "type": "float",
                    "minimum": -2.5,
                    "maximum": 2.5,
                    "unit": "normalized_viewport_half_width",
                    "meaning": (
                        "Horizontal displacement from pad center; zero is centered "
                        "and abs(x)>=1 terminates as a viewport exit."
                    ),
                },
                "y_position": {
                    "type": "float",
                    "minimum": -2.5,
                    "maximum": 2.5,
                    "unit": "normalized_viewport_half_height",
                    "meaning": (
                        "Vertical body-center displacement from pad height plus "
                        "the leg-down offset; positive is above the pad."
                    ),
                },
                "x_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                    "unit": "normalized_velocity",
                    "meaning": (
                        "Signed horizontal velocity; multiply by 5 for Box2D "
                        "world units per second with default viewport constants."
                    ),
                },
                "y_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                    "unit": "normalized_velocity",
                    "meaning": (
                        "Signed vertical velocity; multiply by 7.5 for Box2D "
                        "world units per second with default viewport constants."
                    ),
                },
                "angle": {
                    "type": "float",
                    "minimum": -6.283185307179586,
                    "maximum": 6.283185307179586,
                    "unit": "radians",
                    "meaning": "Lander body angle; zero is upright.",
                },
                "angular_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                    "unit": "observation_angular_velocity",
                    "meaning": (
                        "Signed angular velocity in Gymnasium observation units; "
                        "multiply by 2.5 for radians per second."
                    ),
                },
                "left_leg_contact": {
                    "type": "boolean",
                    "meaning": "True while the left landing leg contacts terrain.",
                },
                "right_leg_contact": {
                    "type": "boolean",
                    "meaning": "True while the right landing leg contacts terrain.",
                },
            },
        },
        action_space=action_space,
        metadata={
            "environment": "LunarLander-v3",
            "provider": "Gymnasium",
            "reward_threshold": 200.0,
            "successful_landing_terminal_reward": 100.0,
            "crash_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
            "nonterminal_shaping_formula": (
                "-100*hypot(x,y)-100*hypot(vx,vy)-100*abs(angle)"
                "+10*left_contact+10*right_contact"
            ),
            "main_engine_cost_per_power": 0.30,
            "side_engine_cost_per_power": 0.03,
            "horizontal_viewport_exit_absolute_x": 1.0,
            "x_velocity_to_world_units_per_second": 5.0,
            "y_velocity_to_world_units_per_second": 7.5,
            "angular_velocity_to_radians_per_second": 2.5,
            "time_limit": _MAX_EPISODE_STEPS,
        },
        environment_parameters={
            "continuous": config.continuous,
            "gravity": config.gravity,
            "enable_wind": config.enable_wind,
            "wind_power": config.wind_power,
            "turbulence_power": config.turbulence_power,
            "frames_per_second": 50,
            "seconds_per_step": 0.02,
            "box2d_velocity_iterations": 180,
            "box2d_position_iterations": 60,
            "initial_random_force_maximum": 1_000.0,
            "main_engine_impulse_power": 13.0,
            "side_engine_impulse_power": 0.6,
            "continuous_main_engine_activation": "command>0",
            "continuous_main_engine_power_formula": "(command+1)/2",
            "continuous_side_engine_activation": "abs(command)>0.5",
            "continuous_side_engine_power_formula": "abs(command)",
            "main_engine_fuel_penalty_per_power": 0.30,
            "side_engine_fuel_penalty_per_power": 0.03,
            "position_shaping_formula": "-100*hypot(x_position,y_position)",
            "velocity_shaping_formula": "-100*hypot(x_velocity,y_velocity)",
            "angle_shaping_formula": "-100*abs(angle)",
            "leg_contact_shaping_per_contact": 10.0,
            "normal_reward_formula": "delta(total_shaping)-engine_fuel_penalty",
            "settled_landing_terminal_reward_override": 100.0,
            "crash_or_viewport_terminal_reward_override": -100.0,
            "horizontal_viewport_exit_absolute_normalized_x": 1.0,
            "helipad_horizontal_absolute_normalized_x": 0.2,
            "x_position_to_world_units": 10.0,
            "y_position_to_world_units": 20.0 / 3.0,
            "x_velocity_to_world_units_per_second": 5.0,
            "y_velocity_to_world_units_per_second": 7.5,
            "angular_velocity_to_radians_per_second": 2.5,
            "wind_force_formula": (
                "wind_power*tanh(sin(0.02*index)+sin(pi*0.01*index))"
            ),
            "wind_and_turbulence_apply_only_while_airborne": True,
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _landed(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward == 100.0
    )


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    continuous: bool,
) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        diagnostics = _episode_diagnostics(record, continuous=continuous)
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": (
                        record.total_reward
                        if record.policy_failure is None
                        else _FAILURE_RETURN
                    ),
                    "landed": _landed(record),
                    "outcome": _episode_outcome(record),
                    "minimum_landing_state_penalty": (
                        diagnostics.minimum_landing_state_penalty
                    ),
                    "closest_normalized_pad_distance": (
                        diagnostics.closest_normalized_pad_distance
                    ),
                    "closest_normalized_speed": diagnostics.closest_normalized_speed,
                    "closest_absolute_angle_radians": (
                        diagnostics.closest_absolute_angle_radians
                    ),
                    "final_normalized_pad_distance": (
                        diagnostics.final_normalized_pad_distance
                    ),
                    "final_normalized_speed": diagnostics.final_normalized_speed,
                    "final_absolute_angle_radians": (
                        diagnostics.final_absolute_angle_radians
                    ),
                    "requested_main_engine_fuel_penalty": (
                        diagnostics.requested_main_fuel_penalty
                    ),
                    "requested_side_engine_fuel_penalty": (
                        diagnostics.requested_side_fuel_penalty
                    ),
                    "charged_fuel_penalty": diagnostics.charged_fuel_penalty,
                    "position_shaping_delta": diagnostics.position_shaping_delta,
                    "velocity_shaping_delta": diagnostics.velocity_shaping_delta,
                    "angle_shaping_delta": diagnostics.angle_shaping_delta,
                    "leg_contact_shaping_delta": diagnostics.contact_shaping_delta,
                    "terminal_override_reward": (
                        diagnostics.terminal_override_reward
                    ),
                    "main_engine_firing_fraction": (
                        diagnostics.main_engine_firing_fraction
                    ),
                    "side_engine_firing_fraction": (
                        diagnostics.side_engine_firing_fraction
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(
                transition.action,
                continuous=continuous,
            )
            next_observation = _trace_observation(
                transition.step.observation
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "action_meaning": _trace_action_meaning(
                            action,
                            continuous=continuous,
                        ),
                        "reward": transition.step.reward,
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": transition.step.metrics,
                    }
                )
            )
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(
    action: PolicyValue,
    *,
    continuous: bool,
) -> PolicyValue:
    if continuous:
        if (
            type(action) is not list
            or len(action) != 2
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
                for value in action
            )
        ):
            raise ValueError("LunarLander trace Action is invalid")
        return list(action)
    if type(action) is not int or action not in {0, 1, 2, 3}:
        raise ValueError("LunarLander trace Action is invalid")
    return action


def _trace_action_meaning(
    action: PolicyValue,
    *,
    continuous: bool,
) -> str:
    if continuous:
        if type(action) is not list:
            raise ValueError("LunarLander trace Action is invalid")
        return "continuous_main_and_lateral_throttle"
    if type(action) is not int or action not in {0, 1, 2, 3}:
        raise ValueError("LunarLander trace Action is invalid")
    return _DISCRETE_ACTION_MEANINGS[action]


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("LunarLander trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("LunarLander trace observation is invalid")
    traced: dict[str, PolicyValue] = {}
    for key in _OBSERVATION_FIELDS[:6]:
        value = observation[key]
        if type(value) is not float:
            raise ValueError("LunarLander trace observation is invalid")
        traced[key] = value
    for key in _OBSERVATION_FIELDS[6:]:
        value = observation[key]
        if type(value) is not bool:
            raise ValueError("LunarLander trace observation is invalid")
        traced[key] = value
    return traced


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    final_step = record.transitions[-1].step
    if final_step.terminated and final_step.reward == 100.0:
        return "settled_landing"
    if final_step.terminated:
        final_observation = _trace_observation(final_step.observation)
        if abs(_float_observation_field(final_observation, "x_position")) >= 1.0:
            return "left_viewport"
        return "crash"
    if final_step.truncated:
        return "time_limit"
    return "incomplete"


def _episode_diagnostics(
    record: EpisodeRecord,
    *,
    continuous: bool,
) -> _EpisodeDiagnostics:
    observations = [_trace_observation(record.initial_observation)]
    requested_main_fuel_penalty = 0.0
    requested_side_fuel_penalty = 0.0
    charged_fuel_penalty = 0.0
    position_shaping_delta = 0.0
    velocity_shaping_delta = 0.0
    angle_shaping_delta = 0.0
    contact_shaping_delta = 0.0
    terminal_override_reward = 0.0
    main_engine_firing_steps = 0
    side_engine_firing_steps = 0
    previous = observations[0]
    for transition in record.transitions:
        current = _trace_observation(transition.step.observation)
        observations.append(current)
        main_power, side_power = _action_powers(
            transition.action,
            continuous=continuous,
        )
        requested_main = 0.30 * main_power
        requested_side = 0.03 * side_power
        requested_main_fuel_penalty += requested_main
        requested_side_fuel_penalty += requested_side
        main_engine_firing_steps += int(main_power > 0.0)
        side_engine_firing_steps += int(side_power > 0.0)
        reward = transition.step.reward
        if type(reward) is not float or not math.isfinite(reward):
            raise ValueError("LunarLander trace reward is invalid")
        reward_was_overridden = (
            transition.step.terminated and reward in {-100.0, 100.0}
        )
        if reward_was_overridden:
            terminal_override_reward += reward
        else:
            previous_terms = _shaping_terms(previous)
            current_terms = _shaping_terms(current)
            deltas = tuple(
                current_value - previous_value
                for previous_value, current_value in zip(
                    previous_terms,
                    current_terms,
                    strict=True,
                )
            )
            position_delta, velocity_delta, angle_delta, contact_delta = deltas
            position_shaping_delta += position_delta
            velocity_shaping_delta += velocity_delta
            angle_shaping_delta += angle_delta
            contact_shaping_delta += contact_delta
            charged_fuel_penalty += requested_main + requested_side
        previous = current
    distances = tuple(_normalized_pad_distance(value) for value in observations)
    speeds = tuple(_normalized_speed(value) for value in observations)
    absolute_angles = tuple(
        abs(_float_observation_field(value, "angle")) for value in observations
    )
    state_penalties = tuple(
        100.0 * (distance + speed + angle)
        for distance, speed, angle in zip(
            distances,
            speeds,
            absolute_angles,
            strict=True,
        )
    )
    left_contacts = tuple(
        1.0 if _bool_observation_field(value, "left_leg_contact") else 0.0
        for value in observations
    )
    right_contacts = tuple(
        1.0 if _bool_observation_field(value, "right_leg_contact") else 0.0
        for value in observations
    )
    transition_count = len(record.transitions)
    return _EpisodeDiagnostics(
        requested_main_fuel_penalty=requested_main_fuel_penalty,
        requested_side_fuel_penalty=requested_side_fuel_penalty,
        charged_fuel_penalty=charged_fuel_penalty,
        position_shaping_delta=position_shaping_delta,
        velocity_shaping_delta=velocity_shaping_delta,
        angle_shaping_delta=angle_shaping_delta,
        contact_shaping_delta=contact_shaping_delta,
        terminal_override_reward=terminal_override_reward,
        minimum_landing_state_penalty=min(state_penalties),
        closest_normalized_pad_distance=min(distances),
        closest_normalized_speed=min(speeds),
        closest_absolute_angle_radians=min(absolute_angles),
        final_normalized_pad_distance=distances[-1],
        final_normalized_speed=speeds[-1],
        final_absolute_angle_radians=absolute_angles[-1],
        main_engine_firing_fraction=(
            main_engine_firing_steps / transition_count if transition_count else 0.0
        ),
        side_engine_firing_fraction=(
            side_engine_firing_steps / transition_count if transition_count else 0.0
        ),
        left_leg_contact_fraction=statistics.fmean(left_contacts),
        right_leg_contact_fraction=statistics.fmean(right_contacts),
    )


def _action_powers(
    action: PolicyValue,
    *,
    continuous: bool,
) -> tuple[float, float]:
    traced = _trace_action(action, continuous=continuous)
    if continuous:
        if type(traced) is not list:
            raise ValueError("LunarLander trace Action is invalid")
        main = traced[0]
        lateral = traced[1]
        if type(main) is not float or type(lateral) is not float:
            raise ValueError("LunarLander trace Action is invalid")
        return (
            (main + 1.0) * 0.5 if main > 0.0 else 0.0,
            abs(lateral) if abs(lateral) > 0.5 else 0.0,
        )
    if type(traced) is not int:
        raise ValueError("LunarLander trace Action is invalid")
    return (1.0 if traced == 2 else 0.0, 1.0 if traced in {1, 3} else 0.0)


def _shaping_terms(
    observation: dict[str, PolicyValue],
) -> tuple[float, float, float, float]:
    x_position = _float_observation_field(observation, "x_position")
    y_position = _float_observation_field(observation, "y_position")
    x_velocity = _float_observation_field(observation, "x_velocity")
    y_velocity = _float_observation_field(observation, "y_velocity")
    angle = _float_observation_field(observation, "angle")
    contact_count = int(_bool_observation_field(observation, "left_leg_contact")) + int(
        _bool_observation_field(observation, "right_leg_contact")
    )
    return (
        -100.0 * math.hypot(x_position, y_position),
        -100.0 * math.hypot(x_velocity, y_velocity),
        -100.0 * abs(angle),
        10.0 * contact_count,
    )


def _normalized_pad_distance(observation: dict[str, PolicyValue]) -> float:
    return math.hypot(
        _float_observation_field(observation, "x_position"),
        _float_observation_field(observation, "y_position"),
    )


def _normalized_speed(observation: dict[str, PolicyValue]) -> float:
    return math.hypot(
        _float_observation_field(observation, "x_velocity"),
        _float_observation_field(observation, "y_velocity"),
    )


def _float_observation_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise ValueError("LunarLander trace observation is invalid")
    return value


def _bool_observation_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> bool:
    value = observation.get(name)
    if type(value) is not bool:
        raise ValueError("LunarLander trace observation is invalid")
    return value


def _json_line(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["LunarLanderBenchmark"]
