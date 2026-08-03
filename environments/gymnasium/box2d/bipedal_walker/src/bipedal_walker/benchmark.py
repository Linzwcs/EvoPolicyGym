"""A parameterized BipedalWalker-v3 Benchmark with public traces."""

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

from .config import BipedalWalkerConfig
from .environment import BipedalWalkerEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-bipedal-walker/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_600
_FAILURE_RETURN = -1_000.0
_OBSERVATION_FIELDS = (
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
    "lidar_ranges",
)
_CONTACT_FIELDS = frozenset(
    {"left_foot_contact", "right_foot_contact"}
)
_ACTION_COMPONENTS = (
    "left_hip",
    "left_knee",
    "right_hip",
    "right_knee",
)
_MOTOR_ENERGY_COEFFICIENT = 0.028
_FORWARD_SHAPING_COEFFICIENT = 130.0
_SCALE = 30.0


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    requested_motor_penalty: float
    charged_motor_penalty: float
    forward_shaping_reward: float
    posture_shaping_reward: float
    terminal_override_reward: float
    final_relative_progress_coordinate: float
    maximum_relative_progress_coordinate: float
    mean_absolute_hull_angle_radians: float
    maximum_absolute_hull_angle_radians: float
    mean_normalized_horizontal_velocity: float
    left_foot_contact_fraction: float
    right_foot_contact_fraction: float
    mean_absolute_motor_command: float
    closest_lidar_fraction: float


class BipedalWalkerBenchmark:
    """Mean BipedalWalker return over deterministic Episode plans."""

    def __init__(self, config: BipedalWalkerConfig | None = None) -> None:
        if config is None:
            config = BipedalWalkerConfig()
        if type(config) is not BipedalWalkerConfig:
            raise TypeError("config must be BipedalWalkerConfig")
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
        return BipedalWalkerEnvironment(episode, config=self._config)

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
        courses = sum(_completed_course(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        falls = sum(_fell_or_moved_behind_start(record) for record in records)
        time_limits = sum(_time_limit(record) for record in records)
        diagnostics = tuple(_episode_diagnostics(record) for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{courses} completed courses."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_courses": courses,
                "fall_or_behind_start_episodes": falls,
                "time_limit_episodes": time_limits,
                "mean_episode_requested_motor_energy_penalty": statistics.fmean(
                    item.requested_motor_penalty for item in diagnostics
                ),
                "mean_episode_charged_motor_energy_penalty": statistics.fmean(
                    item.charged_motor_penalty for item in diagnostics
                ),
                "mean_episode_forward_progress_shaping_reward": statistics.fmean(
                    item.forward_shaping_reward for item in diagnostics
                ),
                "mean_episode_hull_posture_shaping_reward": statistics.fmean(
                    item.posture_shaping_reward for item in diagnostics
                ),
                "mean_final_relative_progress_coordinate": statistics.fmean(
                    item.final_relative_progress_coordinate for item in diagnostics
                ),
                "maximum_relative_progress_coordinate": max(
                    item.maximum_relative_progress_coordinate for item in diagnostics
                ),
                "mean_absolute_hull_angle_radians": statistics.fmean(
                    item.mean_absolute_hull_angle_radians for item in diagnostics
                ),
                "maximum_absolute_hull_angle_radians": max(
                    item.maximum_absolute_hull_angle_radians for item in diagnostics
                ),
                "mean_normalized_horizontal_velocity": statistics.fmean(
                    item.mean_normalized_horizontal_velocity for item in diagnostics
                ),
                "mean_left_foot_contact_fraction": statistics.fmean(
                    item.left_foot_contact_fraction for item in diagnostics
                ),
                "mean_right_foot_contact_fraction": statistics.fmean(
                    item.right_foot_contact_fraction for item in diagnostics
                ),
                "mean_absolute_motor_command": statistics.fmean(
                    item.mean_absolute_motor_command for item in diagnostics
                ),
                "mean_closest_lidar_fraction": statistics.fmean(
                    item.closest_lidar_fraction for item in diagnostics
                ),
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: BipedalWalkerConfig) -> BenchmarkSpec:
    scalar_fields: dict[str, PolicyValue] = {
        "hull_angle": _float_field_spec(
            -math.pi,
            math.pi,
            unit="radians",
            meaning="Hull angle relative to upright; zero is upright.",
        ),
        "hull_angular_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_angular_velocity",
            meaning="Gym value 2*hull_angular_velocity/50.",
        ),
        "horizontal_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_horizontal_velocity",
            meaning="Gym value 0.3*world_velocity_x*(600/30)/50.",
        ),
        "vertical_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_vertical_velocity",
            meaning="Gym value 0.3*world_velocity_y*(400/30)/50.",
        ),
        "left_hip_angle": _float_field_spec(
            -math.pi,
            math.pi,
            unit="radians",
            meaning="Left hip joint angle; configured joint limits are -0.8 to 1.1 rad.",
        ),
        "left_hip_angular_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_joint_speed",
            meaning="Left hip joint angular speed divided by 4 rad/s.",
        ),
        "left_knee_angle": _float_field_spec(
            -math.pi,
            math.pi,
            unit="offset_radians",
            meaning="Left knee joint angle plus 1 radian; joint limits are -1.6 to -0.1 rad.",
        ),
        "left_knee_angular_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_joint_speed",
            meaning="Left knee joint angular speed divided by 6 rad/s.",
        ),
        "left_foot_contact": {
            "type": "boolean",
            "meaning": "Whether the left lower leg/foot has terrain contact.",
        },
        "right_hip_angle": _float_field_spec(
            -math.pi,
            math.pi,
            unit="radians",
            meaning="Right hip joint angle; configured joint limits are -0.8 to 1.1 rad.",
        ),
        "right_hip_angular_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_joint_speed",
            meaning="Right hip joint angular speed divided by 4 rad/s.",
        ),
        "right_knee_angle": _float_field_spec(
            -math.pi,
            math.pi,
            unit="offset_radians",
            meaning="Right knee joint angle plus 1 radian; joint limits are -1.6 to -0.1 rad.",
        ),
        "right_knee_angular_velocity": _float_field_spec(
            -5.0,
            5.0,
            unit="normalized_joint_speed",
            meaning="Right knee joint angular speed divided by 6 rad/s.",
        ),
        "right_foot_contact": {
            "type": "boolean",
            "meaning": "Whether the right lower leg/foot has terrain contact.",
        },
    }
    scalar_fields["lidar_ranges"] = {
        "type": "array",
        "shape": [10],
        "items": {
            "type": "float",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "meaning": (
            "Terrain-hit fraction of a 5.333-world-unit ray; 1 means no hit. "
            "Ray i points 0.15*i radians from downward toward forward."
        ),
        "order": "ray_index_0_downward_to_ray_index_9_near_forward",
    }
    return BenchmarkSpec(
        id="gymnasium/BipedalWalker-v3/mean-return-v1",
        description=(
            "Control left/right hip and knee motors to walk across uneven terrain. "
            "Each action sign selects target motor direction (hips ±4 rad/s, "
            "knees ±6 rad/s), while magnitude selects up to 80 N·m torque. "
            "Normal reward is the change in 130*x/30-5*abs(hull_angle), minus "
            "0.028*sum(abs(action)). A hull-ground collision or movement behind "
            "the start overrides that transition reward to -100 and terminates; "
            "reaching the far end terminates normally. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, PolicyValue]",
            "source_dtype": "float32",
            "fields": scalar_fields,
            "notes": (
                "All continuous body and joint values are normalized by "
                "Gymnasium. Lidar values are fractions of maximum range."
            ),
        },
        action_space={
            "type": "array",
            "shape": [4],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": list(_ACTION_COMPONENTS),
            "policy_carrier": "list[float]",
            "component_meaning": {
                "left_hip": "left hip motor command",
                "left_knee": "left knee motor command",
                "right_hip": "right hip motor command",
                "right_knee": "right knee motor command",
            },
            "notes": (
                "Sign selects fixed target motor-speed direction; absolute "
                "magnitude selects maximum motor torque, not speed."
            ),
        },
        metadata={
            "environment": "BipedalWalker-v3",
            "provider": "Gymnasium",
            "reward_threshold": 300.0,
            "fall_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "hardcore": config.hardcore,
            "frames_per_second": 50,
            "seconds_per_step": 0.02,
            "box2d_velocity_iterations": 180,
            "box2d_position_iterations": 60,
            "maximum_motor_torque": 80.0,
            "hip_target_speed_absolute_radians_per_second": 4.0,
            "knee_target_speed_absolute_radians_per_second": 6.0,
            "motor_energy_penalty_per_absolute_action_component": 0.028,
            "forward_shaping_formula": "130*world_x/30",
            "posture_shaping_formula": "-5*abs(hull_angle)",
            "normal_reward_formula": (
                "delta(forward_shaping+posture_shaping)-0.028*sum(abs(action))"
            ),
            "fall_or_behind_start_terminal_reward_override": -100.0,
            "terrain_length_segments": 200,
            "finish_grass_segments": 10,
            "terrain_step_world_units": 14.0 / 30.0,
            "course_end_world_x": 190.0 * 14.0 / 30.0,
            "lidar_ray_count": 10,
            "lidar_range_world_units": 160.0 / 30.0,
            "lidar_ray_angle_from_down_formula_radians": "1.5*ray_index/10",
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _float_field_spec(
    minimum: float,
    maximum: float,
    *,
    unit: str,
    meaning: str,
) -> dict[str, PolicyValue]:
    return {
        "type": "float",
        "minimum": minimum,
        "maximum": maximum,
        "unit": unit,
        "meaning": meaning,
    }


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _completed_course(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward > -100.0
    )


def _fell_or_moved_behind_start(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward == -100.0
    )


def _time_limit(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
        and not record.transitions[-1].step.terminated
    )


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        diagnostics = _episode_diagnostics(record)
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
                    "completed_course": _completed_course(record),
                    "outcome": _episode_outcome(record),
                    "requested_motor_energy_penalty": (
                        diagnostics.requested_motor_penalty
                    ),
                    "charged_motor_energy_penalty": diagnostics.charged_motor_penalty,
                    "forward_progress_shaping_reward": diagnostics.forward_shaping_reward,
                    "hull_posture_shaping_reward": diagnostics.posture_shaping_reward,
                    "terminal_override_reward": diagnostics.terminal_override_reward,
                    "final_relative_progress_coordinate": (
                        diagnostics.final_relative_progress_coordinate
                    ),
                    "maximum_relative_progress_coordinate": (
                        diagnostics.maximum_relative_progress_coordinate
                    ),
                    "mean_absolute_hull_angle_radians": (
                        diagnostics.mean_absolute_hull_angle_radians
                    ),
                    "mean_normalized_horizontal_velocity": (
                        diagnostics.mean_normalized_horizontal_velocity
                    ),
                    "mean_absolute_motor_command": (
                        diagnostics.mean_absolute_motor_command
                    ),
                    "closest_lidar_fraction": diagnostics.closest_lidar_fraction,
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
                        "action_components": dict(
                            zip(_ACTION_COMPONENTS, action, strict=True)
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


def _trace_action(action: PolicyValue) -> list[float]:
    if type(action) is not list or len(action) != 4:
        raise ValueError("BipedalWalker trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("BipedalWalker trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("BipedalWalker trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("BipedalWalker trace observation is invalid")
    traced: dict[str, PolicyValue] = {}
    for key in _OBSERVATION_FIELDS[:-1]:
        value = observation[key]
        if key in _CONTACT_FIELDS:
            if type(value) is not bool:
                raise ValueError(
                    "BipedalWalker trace observation is invalid"
                )
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("BipedalWalker trace observation is invalid")
        traced[key] = value
    lidar = observation["lidar_ranges"]
    if (
        type(lidar) is not list
        or len(lidar) != 10
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in lidar
        )
    ):
        raise ValueError("BipedalWalker trace observation is invalid")
    traced["lidar_ranges"] = list(lidar)
    return traced


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if _completed_course(record):
        return "course_complete"
    if _fell_or_moved_behind_start(record):
        return "fall_or_behind_start"
    if _time_limit(record):
        return "time_limit"
    return "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    observations = [record.initial_observation]
    requested_motor_penalty = 0.0
    charged_motor_penalty = 0.0
    forward_shaping_reward = 0.0
    posture_shaping_reward = 0.0
    terminal_override_reward = 0.0
    relative_progress = 0.0
    maximum_relative_progress = 0.0
    absolute_commands: list[float] = []
    previous = _trace_observation(record.initial_observation)
    for transition in record.transitions:
        action = _trace_action(transition.action)
        current = _trace_observation(transition.step.observation)
        observations.append(transition.step.observation)
        requested = _MOTOR_ENERGY_COEFFICIENT * math.fsum(
            abs(value) for value in action
        )
        requested_motor_penalty += requested
        absolute_commands.extend(abs(value) for value in action)
        reward = transition.step.reward
        if type(reward) is not float or not math.isfinite(reward):
            raise ValueError("BipedalWalker trace reward is invalid")
        reward_overridden = transition.step.terminated and reward == -100.0
        if reward_overridden:
            terminal_override_reward += -100.0
        else:
            posture_delta = -5.0 * (
                abs(_float_observation_field(current, "hull_angle"))
                - abs(_float_observation_field(previous, "hull_angle"))
            )
            forward_delta = reward + requested - posture_delta
            charged_motor_penalty += requested
            posture_shaping_reward += posture_delta
            forward_shaping_reward += forward_delta
            relative_progress += (
                forward_delta * _SCALE / _FORWARD_SHAPING_COEFFICIENT
            )
            maximum_relative_progress = max(
                maximum_relative_progress,
                relative_progress,
            )
        previous = current
    traced_observations = tuple(_trace_observation(value) for value in observations)
    absolute_hull_angles = tuple(
        abs(_float_observation_field(value, "hull_angle"))
        for value in traced_observations
    )
    horizontal_velocities = tuple(
        _float_observation_field(value, "horizontal_velocity")
        for value in traced_observations
    )
    left_contacts = tuple(
        1.0 if _bool_observation_field(value, "left_foot_contact") else 0.0
        for value in traced_observations
    )
    right_contacts = tuple(
        1.0 if _bool_observation_field(value, "right_foot_contact") else 0.0
        for value in traced_observations
    )
    closest_lidar_fraction = min(
        min(_lidar_observation_field(value)) for value in traced_observations
    )
    return _EpisodeDiagnostics(
        requested_motor_penalty=requested_motor_penalty,
        charged_motor_penalty=charged_motor_penalty,
        forward_shaping_reward=forward_shaping_reward,
        posture_shaping_reward=posture_shaping_reward,
        terminal_override_reward=terminal_override_reward,
        final_relative_progress_coordinate=relative_progress,
        maximum_relative_progress_coordinate=maximum_relative_progress,
        mean_absolute_hull_angle_radians=statistics.fmean(absolute_hull_angles),
        maximum_absolute_hull_angle_radians=max(absolute_hull_angles),
        mean_normalized_horizontal_velocity=statistics.fmean(horizontal_velocities),
        left_foot_contact_fraction=statistics.fmean(left_contacts),
        right_foot_contact_fraction=statistics.fmean(right_contacts),
        mean_absolute_motor_command=(
            statistics.fmean(absolute_commands) if absolute_commands else 0.0
        ),
        closest_lidar_fraction=closest_lidar_fraction,
    )


def _float_observation_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise ValueError("BipedalWalker trace observation is invalid")
    return value


def _bool_observation_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> bool:
    value = observation.get(name)
    if type(value) is not bool:
        raise ValueError("BipedalWalker trace observation is invalid")
    return value


def _lidar_observation_field(
    observation: dict[str, PolicyValue],
) -> list[float]:
    value = observation.get("lidar_ranges")
    if type(value) is not list or len(value) != 10:
        raise ValueError("BipedalWalker trace observation is invalid")
    result: list[float] = []
    for item in value:
        if type(item) is not float or not 0.0 <= item <= 1.0:
            raise ValueError("BipedalWalker trace observation is invalid")
        result.append(item)
    return result


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


__all__ = ["BipedalWalkerBenchmark"]
