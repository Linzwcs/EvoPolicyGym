"""A reproducible Pendulum Benchmark with bounded public traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .environment import PendulumEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-pendulum/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -3300.0
_ANGLE_COST_COEFFICIENT = 1.0
_ANGULAR_VELOCITY_COST_COEFFICIENT = 0.1
_TORQUE_COST_COEFFICIENT = 0.001

_SPEC = BenchmarkSpec(
    id="gymnasium/Pendulum-v1/mean-return-v1",
    description=(
        "Control an inverted pendulum for 200 steps of 0.05 seconds. Theta=0 "
        "is upright and theta=±pi is downward; reconstruct normalized theta as "
        "atan2(sin_theta,cos_theta). Return one finite Python float in [-2,2] "
        "as torque in N·m. Each reward is "
        "-(theta^2+0.1*angular_velocity^2+0.001*torque^2), computed from the "
        "state before the action. Pendulum never succeeds early: maximize mean "
        "return by staying upright with low speed and control effort."
    ),
    observation_space={
        "type": "object",
        "policy_carrier": "dict[str, float]",
        "source_dtype": "float32",
        "fields": {
            "cos_theta": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": "Cosine of theta; theta=0 is upright.",
            },
            "sin_theta": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": "Sine of theta; atan2(sin_theta,cos_theta) gives theta.",
            },
            "theta_angular_velocity": {
                "type": "float",
                "minimum": -8.0,
                "maximum": 8.0,
                "unit": "radians_per_second",
                "meaning": "Signed angular velocity, clipped to ±8 rad/s.",
            },
        },
    },
    action_space={
        "type": "float",
        "minimum": -2.0,
        "maximum": 2.0,
        "policy_carrier": "float",
        "unit": "newton_meters",
        "meaning": "signed_torque: negative/zero/positive",
    },
    metadata={
        "environment": "Pendulum-v1",
        "provider": "Gymnasium",
        "minimum_step_reward": -16.2736044,
        "maximum_step_reward": 0.0,
        "failure_return": _FAILURE_RETURN,
        "reward_threshold": -200.0,
    },
    environment_parameters={
        "seconds_per_step": 0.05,
        "gravity_meters_per_second_squared": 10.0,
        "pendulum_mass_kilograms": 1.0,
        "pendulum_length_meters": 1.0,
        "maximum_absolute_torque_newton_meters": 2.0,
        "maximum_absolute_angular_velocity_radians_per_second": 8.0,
        "angular_velocity_update_formula": (
            "clip(theta_dot+(3*g/(2*l)*sin(theta)+3*torque/(m*l^2))*0.05,-8,8)"
        ),
        "angle_update_formula": "theta+updated_theta_dot*0.05",
        "integrator": "semi_implicit_euler",
        "initial_angle_minimum_radians": -math.pi,
        "initial_angle_maximum_radians": math.pi,
        "initial_angular_velocity_minimum_radians_per_second": -1.0,
        "initial_angular_velocity_maximum_radians_per_second": 1.0,
        "angle_cost_coefficient": _ANGLE_COST_COEFFICIENT,
        "angular_velocity_cost_coefficient": _ANGULAR_VELOCITY_COST_COEFFICIENT,
        "torque_cost_coefficient": _TORQUE_COST_COEFFICIENT,
        "reward_formula": "-(theta^2+0.1*theta_dot^2+0.001*torque^2)",
        "natural_termination": "none",
        "time_limit": 200,
    },
    max_episode_steps=200,
    primary_metric="mean_return",
    score_direction="maximize",
)


class PendulumBenchmark:
    """Mean Pendulum return over deterministic Episode plans."""

    @property
    def spec(self) -> BenchmarkSpec:
        return _SPEC

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
        return PendulumEnvironment(episode)

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
        failures = sum(record.policy_failure is not None for record in records)
        truncated = sum(_truncated(record) for record in records)
        completed = truncated
        cost_components = tuple(_episode_cost_components(record) for record in records)
        state_diagnostics = tuple(_episode_state_diagnostics(record) for record in records)
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{failures} Policy failures."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_episodes": completed,
                "truncated_episodes": truncated,
                "incomplete_episodes": len(records) - failures - completed,
                "mean_episode_angle_cost": statistics.fmean(
                    component[0] for component in cost_components
                ),
                "mean_episode_angular_velocity_cost": statistics.fmean(
                    component[1] for component in cost_components
                ),
                "mean_episode_torque_cost": statistics.fmean(
                    component[2] for component in cost_components
                ),
                "mean_episode_mean_absolute_angle_error_radians": statistics.fmean(
                    diagnostic[0] for diagnostic in state_diagnostics
                ),
                "mean_episode_closest_upright_angle_radians": statistics.fmean(
                    diagnostic[1] for diagnostic in state_diagnostics
                ),
                "mean_episode_fraction_in_upright_half": statistics.fmean(
                    diagnostic[2] for diagnostic in state_diagnostics
                ),
                "mean_episode_mean_absolute_angular_velocity": statistics.fmean(
                    diagnostic[3] for diagnostic in state_diagnostics
                ),
                "mean_episode_mean_absolute_torque": statistics.fmean(
                    _episode_mean_absolute_torque(record) for record in records
                ),
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        angle_cost, angular_velocity_cost, torque_cost = _episode_cost_components(record)
        mean_angle, closest_angle, upright_fraction, mean_velocity = (
            _episode_state_diagnostics(record)
        )
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
                    "ended_by_truncation": _truncated(record),
                    "outcome": (
                        "policy_failure"
                        if record.policy_failure is not None
                        else "time_limit"
                        if _truncated(record)
                        else "incomplete"
                    ),
                    "angle_cost": angle_cost,
                    "angular_velocity_cost": angular_velocity_cost,
                    "torque_cost": torque_cost,
                    "mean_absolute_angle_error_radians": mean_angle,
                    "closest_upright_angle_radians": closest_angle,
                    "fraction_in_upright_half": upright_fraction,
                    "mean_absolute_angular_velocity": mean_velocity,
                    "mean_absolute_torque": _episode_mean_absolute_torque(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
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
                        "action_meaning": "signed_torque_newton_meters",
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


def _trace_observation(observation: PolicyValue) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("Pendulum trace observation is invalid")
    expected = {
        "cos_theta",
        "sin_theta",
        "theta_angular_velocity",
    }
    if set(observation) != expected:
        raise ValueError("Pendulum trace observation is invalid")
    traced: dict[str, float] = {}
    for key in (
        "cos_theta",
        "sin_theta",
        "theta_angular_velocity",
    ):
        value = observation[key]
        if type(value) is not float:
            raise ValueError("Pendulum trace observation is invalid")
        traced[key] = value
    return traced


def _trace_action(action: PolicyValue) -> float:
    if type(action) is not float or not math.isfinite(action) or not -2.0 <= action <= 2.0:
        raise ValueError("Pendulum trace Action is invalid")
    return action


def _state(observation: PolicyValue) -> tuple[float, float]:
    fields = _trace_observation(observation)
    angle = math.atan2(fields["sin_theta"], fields["cos_theta"])
    return angle, fields["theta_angular_velocity"]


def _episode_cost_components(record: EpisodeRecord) -> tuple[float, float, float]:
    angle_costs: list[float] = []
    angular_velocity_costs: list[float] = []
    torque_costs: list[float] = []
    observation = record.initial_observation
    for transition in record.transitions:
        angle, angular_velocity = _state(observation)
        torque = _trace_action(transition.action)
        angle_costs.append(_ANGLE_COST_COEFFICIENT * angle**2)
        angular_velocity_costs.append(
            _ANGULAR_VELOCITY_COST_COEFFICIENT * angular_velocity**2
        )
        torque_costs.append(_TORQUE_COST_COEFFICIENT * torque**2)
        observation = transition.step.observation
    return math.fsum(angle_costs), math.fsum(angular_velocity_costs), math.fsum(torque_costs)


def _episode_state_diagnostics(record: EpisodeRecord) -> tuple[float, float, float, float]:
    observations = (
        record.initial_observation,
        *(transition.step.observation for transition in record.transitions),
    )
    states = tuple(_state(observation) for observation in observations)
    absolute_angles = tuple(abs(state[0]) for state in states)
    return (
        statistics.fmean(absolute_angles),
        min(absolute_angles),
        statistics.fmean(1.0 if math.cos(state[0]) >= 0.0 else 0.0 for state in states),
        statistics.fmean(abs(state[1]) for state in states),
    )


def _episode_mean_absolute_torque(record: EpisodeRecord) -> float:
    if not record.transitions:
        return 0.0
    return statistics.fmean(
        abs(_trace_action(transition.action)) for transition in record.transitions
    )


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


__all__ = ["PendulumBenchmark"]
