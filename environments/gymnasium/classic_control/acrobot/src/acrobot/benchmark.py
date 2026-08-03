"""A reproducible Acrobot Benchmark with bounded public traces."""

from __future__ import annotations

import hashlib
import json
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

from .environment import AcrobotEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-acrobot/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -500.0
_ACTION_MEANINGS = ("negative_torque", "zero_torque", "positive_torque")

_SPEC = BenchmarkSpec(
    id="gymnasium/Acrobot-v1/mean-return-v1",
    description=(
        "Swing a two-link underactuated chain from hanging downward until its "
        "free end height -cos(theta1)-cos(theta1+theta2) exceeds 1 meter. "
        "Theta1 is measured from downward; theta2 is the second link's angle "
        "relative to the first. Actions 0/1/2 apply -1/0/+1 N·m at the middle "
        "joint. Each 0.2-second nonterminal step rewards -1; the successful "
        "transition rewards 0 and terminates. The Episode truncates at 500 "
        "steps. Maximize mean return; less-negative return means faster success."
    ),
    observation_space={
        "type": "object",
        "policy_carrier": "dict[str, float]",
        "source_dtype": "float32",
        "fields": {
            "cos_theta_1": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": "Cosine of theta1, whose zero angle points link 1 downward.",
            },
            "sin_theta_1": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": "Sine of theta1, whose zero angle points link 1 downward.",
            },
            "cos_theta_2": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": (
                    "Cosine of theta2, the angle of link 2 relative to link 1; "
                    "theta2=0 aligns both links."
                ),
            },
            "sin_theta_2": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
                "meaning": (
                    "Sine of theta2, the angle of link 2 relative to link 1; "
                    "theta2=0 aligns both links."
                ),
            },
            "theta_1_angular_velocity": {
                "type": "float",
                "minimum": -12.566371,
                "maximum": 12.566371,
                "unit": "radians_per_second",
                "meaning": "Angular velocity of theta1, bounded to ±4π rad/s.",
            },
            "theta_2_angular_velocity": {
                "type": "float",
                "minimum": -28.274334,
                "maximum": 28.274334,
                "unit": "radians_per_second",
                "meaning": (
                    "Angular velocity of relative theta2, bounded to ±9π rad/s."
                ),
            },
        },
    },
    action_space={
        "type": "discrete",
        "values": [0, 1, 2],
        "component": "middle_joint_torque",
        "meaning": {
            "0": "apply_negative_torque",
            "1": "apply_zero_torque",
            "2": "apply_positive_torque",
        },
    },
    metadata={
        "environment": "Acrobot-v1",
        "provider": "Gymnasium",
        "dynamics": "book",
        "reward_per_nonterminal_step": -1.0,
        "reward_on_success": 0.0,
        "failure_return": _FAILURE_RETURN,
        "reward_threshold": -100.0,
    },
    environment_parameters={
        "dynamics_equations": "sutton_barto_book",
        "seconds_per_step": 0.2,
        "integrator": "runge_kutta_4",
        "link_1_length_meters": 1.0,
        "link_2_length_meters": 1.0,
        "link_1_mass_kilograms": 1.0,
        "link_2_mass_kilograms": 1.0,
        "link_center_of_mass_meters": 0.5,
        "link_moment_of_inertia": 1.0,
        "gravity_meters_per_second_squared": 9.8,
        "available_torques_newton_meters": [-1.0, 0.0, 1.0],
        "torque_noise_maximum": 0.0,
        "theta_1_velocity_limit_radians_per_second": 4.0 * 3.141592653589793,
        "theta_2_velocity_limit_radians_per_second": 9.0 * 3.141592653589793,
        "initial_underlying_state_minimum": -0.1,
        "initial_underlying_state_maximum": 0.1,
        "target_height_formula": "-cos(theta1)-cos(theta1+theta2)",
        "target_height_strictly_greater_than_meters": 1.0,
        "nonterminal_reward": -1.0,
        "successful_terminal_reward": 0.0,
        "time_limit": 500,
    },
    max_episode_steps=500,
    primary_metric="mean_return",
    score_direction="maximize",
)


class AcrobotBenchmark:
    """Mean Acrobot return over deterministic Episode plans."""

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
        return AcrobotEnvironment(episode)

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
        successes = sum(_successful(record) for record in records)
        time_limits = sum(
            bool(
                record.policy_failure is None
                and record.transitions
                and record.transitions[-1].step.truncated
                and not record.transitions[-1].step.terminated
            )
            for record in records
        )
        episode_maximum_heights = tuple(
            _episode_maximum_tip_height(record) for record in records
        )
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{successes} reached the target."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "successful_episodes": successes,
                "time_limit_episodes": time_limits,
                "mean_episode_maximum_tip_height_meters": statistics.fmean(
                    episode_maximum_heights
                ),
                "maximum_tip_height_meters": max(episode_maximum_heights),
                "mean_closest_height_gap_meters": statistics.fmean(
                    max(1.0 - height, 0.0) for height in episode_maximum_heights
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


def _successful(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
    )


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
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
                    "reached_target": _successful(record),
                    "outcome": (
                        "policy_failure"
                        if record.policy_failure is not None
                        else "target_height_reached"
                        if _successful(record)
                        else "time_limit"
                        if record.transitions and record.transitions[-1].step.truncated
                        else "incomplete"
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int or not 0 <= transition.action < 3:
                raise ValueError("Acrobot trace Action is invalid")
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
                        "action": transition.action,
                        "action_meaning": _ACTION_MEANINGS[transition.action],
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
        raise ValueError("Acrobot trace observation is invalid")
    expected = {
        "cos_theta_1",
        "sin_theta_1",
        "cos_theta_2",
        "sin_theta_2",
        "theta_1_angular_velocity",
        "theta_2_angular_velocity",
    }
    if set(observation) != expected:
        raise ValueError("Acrobot trace observation is invalid")
    traced: dict[str, float] = {}
    for key in (
        "cos_theta_1",
        "sin_theta_1",
        "cos_theta_2",
        "sin_theta_2",
        "theta_1_angular_velocity",
        "theta_2_angular_velocity",
    ):
        value = observation[key]
        if type(value) is not float:
            raise ValueError("Acrobot trace observation is invalid")
        traced[key] = value
    return traced


def _episode_maximum_tip_height(record: EpisodeRecord) -> float:
    observations = (
        record.initial_observation,
        *(transition.step.observation for transition in record.transitions),
    )
    return max(_tip_height(observation) for observation in observations)


def _tip_height(observation: PolicyValue) -> float:
    fields = _trace_observation(observation)
    return -fields["cos_theta_1"] - (
        fields["cos_theta_1"] * fields["cos_theta_2"]
        - fields["sin_theta_1"] * fields["sin_theta_2"]
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


__all__ = ["AcrobotBenchmark"]
