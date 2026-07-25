"""A parameterized Ant-v5 Benchmark with public nested traces."""

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

from .config import AntConfig
from .environment import AntEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-ant/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_000
_BODY_FIELDS = (
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    "front_left_hip_angle",
    "front_left_ankle_angle",
    "front_right_hip_angle",
    "front_right_ankle_angle",
    "back_left_hip_angle",
    "back_left_ankle_angle",
    "back_right_hip_angle",
    "back_right_ankle_angle",
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    "front_left_hip_angular_velocity",
    "front_left_ankle_angular_velocity",
    "front_right_hip_angular_velocity",
    "front_right_ankle_angular_velocity",
    "back_left_hip_angular_velocity",
    "back_left_ankle_angular_velocity",
    "back_right_hip_angular_velocity",
    "back_right_ankle_angular_velocity",
)
_CONTACT_BODIES = (
    "torso",
    "front_left_leg",
    "front_left_aux",
    "front_left_ankle",
    "front_right_leg",
    "front_right_aux",
    "front_right_ankle",
    "back_left_leg",
    "back_left_aux",
    "back_left_ankle",
    "back_right_leg",
    "back_right_aux",
    "back_right_ankle",
)
_CONTACT_COMPONENTS = (
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
)
_METRIC_FIELDS = (
    "x_position",
    "y_position",
    "distance_from_origin",
    "x_velocity",
    "y_velocity",
    "reward_forward",
    "reward_control",
    "reward_contact",
    "reward_survive",
)


class AntBenchmark:
    """Mean Ant return over deterministic Episode plans."""

    def __init__(self, config: AntConfig | None = None) -> None:
        if config is None:
            config = AntConfig()
        if type(config) is not AntConfig:
            raise TypeError("config must be AntConfig")
        self._config = config
        self._scalar_observation_fields = _scalar_observation_fields(config)
        self._failure_return = _failure_return(config)
        self._spec = _benchmark_spec(
            config,
            failure_return=self._failure_return,
        )

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
        return AntEnvironment(episode, config=self._config)

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
                else self._failure_return
            )
            for record in records
        )
        score = statistics.fmean(returns)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        final_positions = tuple(
            _final_x_position(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        mean_final_x = (
            statistics.fmean(final_positions)
            if final_positions
            else None
        )
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final x position "
                    f"{_position_summary(mean_final_x)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_final_x_position": mean_final_x,
                "episodes": len(records),
                "policy_failures": failures,
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    scalar_observation_fields=(
                        self._scalar_observation_fields
                    ),
                    include_contact=(
                        self._config.include_cfrc_ext_in_observation
                    ),
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: AntConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
        }
        fields["torso_y_position"] = {
            "type": "float",
            "unit": "meters",
        }
    for name in _BODY_FIELDS:
        if name == "torso_z_position":
            unit = "meters"
        elif name.startswith("torso_orientation_"):
            unit = "quaternion_component"
        elif name in {
            "torso_x_velocity",
            "torso_y_velocity",
            "torso_z_velocity",
        }:
            unit = "meters_per_second"
        elif name.endswith("_angular_velocity"):
            unit = "radians_per_second"
        else:
            unit = "radians"
        fields[name] = {"type": "float", "unit": unit}
    if config.include_cfrc_ext_in_observation:
        fields["contact_forces"] = _contact_space()
    return BenchmarkSpec(
        id="gymnasium/Ant-v5/mean-return-v1",
        description=(
            "Coordinate eight leg torques to keep a quadruped Ant healthy "
            "and move its selected main body in the positive x direction. "
            "Maximize mean Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space={
            "type": "array",
            "shape": [8],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": [
                "back_right_hip_torque",
                "back_right_ankle_torque",
                "front_left_hip_torque",
                "front_left_ankle_torque",
                "front_right_hip_torque",
                "front_right_ankle_torque",
                "back_left_hip_torque",
                "back_left_ankle_torque",
            ],
        },
        metadata={
            "environment": "Ant-v5",
            "provider": "Gymnasium",
            "reward_threshold": 6000.0,
            "official_model": "ant.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "contact_cost_weight": config.contact_cost_weight,
            "healthy_reward": config.healthy_reward,
            "main_body": config.main_body,
            "terminate_when_unhealthy": config.terminate_when_unhealthy,
            "healthy_z_range": list(config.healthy_z_range),
            "contact_force_range": list(config.contact_force_range),
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "include_cfrc_ext_in_observation": (
                config.include_cfrc_ext_in_observation
            ),
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _contact_space() -> PolicyValue:
    bodies: dict[str, PolicyValue] = {}
    for body in _CONTACT_BODIES:
        components: dict[str, PolicyValue] = {}
        for component in _CONTACT_COMPONENTS:
            components[component] = {
                "type": "float",
                "unit": (
                    "newtons"
                    if component.startswith("force_")
                    else "newton_meters"
                ),
            }
        bodies[body] = {"type": "object", "fields": components}
    return {"type": "object", "fields": bodies}


def _scalar_observation_fields(config: AntConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", "torso_y_position", *_BODY_FIELDS)
    )


def _failure_return(config: AntConfig) -> float:
    maximum_contact = max(
        abs(config.contact_force_range[0]),
        abs(config.contact_force_range[1]),
    )
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        config.healthy_reward,
        8.0 * config.ctrl_cost_weight,
        78.0 * config.contact_cost_weight * maximum_contact**2,
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _final_x_position(record: EpisodeRecord) -> float:
    metrics = record.transitions[-1].step.metrics
    traced = _trace_metrics(metrics)
    return traced["x_position"]


def _position_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    scalar_observation_fields: tuple[str, ...],
    include_contact: bool,
    failure_return: float,
) -> Artifact:
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
                        else failure_return
                    ),
                    "final_x_position": (
                        _final_x_position(record)
                        if record.policy_failure is None
                        and record.transitions
                        else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(
            record.initial_observation,
            scalar_fields=scalar_observation_fields,
            include_contact=include_contact,
        )
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation,
                scalar_fields=scalar_observation_fields,
                include_contact=include_contact,
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "reward": transition.step.reward,
                        "metrics": _trace_metrics(
                            transition.step.metrics
                        ),
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
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
    if type(action) is not list or len(action) != 8:
        raise ValueError("Ant trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("Ant trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    scalar_fields: tuple[str, ...],
    include_contact: bool,
) -> dict[str, object]:
    expected = set(scalar_fields)
    if include_contact:
        expected.add("contact_forces")
    if type(observation) is not dict or set(observation) != expected:
        raise ValueError("Ant trace observation is invalid")
    traced: dict[str, object] = {}
    for key in scalar_fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Ant trace observation is invalid")
        traced[key] = value
    if include_contact:
        traced["contact_forces"] = _trace_contact_forces(
            observation["contact_forces"]
        )
    return traced


def _trace_contact_forces(value: PolicyValue) -> dict[str, dict[str, float]]:
    if type(value) is not dict or set(value) != set(_CONTACT_BODIES):
        raise ValueError("Ant trace contact forces are invalid")
    traced: dict[str, dict[str, float]] = {}
    for body in _CONTACT_BODIES:
        components = value[body]
        if (
            type(components) is not dict
            or set(components) != set(_CONTACT_COMPONENTS)
        ):
            raise ValueError("Ant trace contact forces are invalid")
        traced_components: dict[str, float] = {}
        for component in _CONTACT_COMPONENTS:
            item = components[component]
            if type(item) is not float or not math.isfinite(item):
                raise ValueError("Ant trace contact forces are invalid")
            traced_components[component] = item
        traced[body] = traced_components
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Ant trace metrics are invalid")
    traced: dict[str, float] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Ant trace metrics are invalid")
        traced[key] = value
    return traced


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


__all__ = ["AntBenchmark"]
