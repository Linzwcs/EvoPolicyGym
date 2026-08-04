"""robosuite manipulation profiles with deterministic Episode plans."""

from __future__ import annotations

import base64
import hashlib
import json
import statistics
import struct
from collections.abc import Sequence
from typing import cast

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue

from .config import RobosuiteConfig
from .environment import RobosuiteEnvironment
from .video import (
    VIDEO_FRAME_SHAPE,
    VIDEO_MAX_FRAMES_PER_EPISODE,
    trace_metrics,
    video_capture_interval,
    video_feedback,
)

_SEED_DOMAIN = b"evopolicygym-robosuite/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_TRACE_PREFIX_STEPS = 32
_TRACE_SUFFIX_STEPS = 8


class RobosuiteBenchmark:
    """Success rate for one fixed robosuite manipulation profile."""

    def __init__(self, config: RobosuiteConfig | None = None) -> None:
        if config is None:
            config = RobosuiteConfig()
        if type(config) is not RobosuiteConfig:
            raise TypeError("config must be RobosuiteConfig")
        self._config = config
        self._spec = _spec(config)

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
            EpisodeSpec(environment_seed=_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return RobosuiteEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        score = successes / len(records)
        final_metrics = tuple(
            metrics for record in records if (metrics := _final_metrics(record)) is not None
        )
        traced = records[:_MAX_TRACED_EPISODES]
        trace, traced_transitions, omitted_transitions = _trace(
            traced,
            total_transitions=sum(record.steps for record in records),
        )
        video_artifacts, video_manifests, video_unavailable = video_feedback(
            records,
            profile=self._config.profile,
            capture_interval=video_capture_interval(self._config.max_episode_steps),
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Solved {successes}/{len(records)} robosuite Episodes "
                    f"({score:.3f} success rate)."
                ),
                "profile": self._config.profile,
                "success_rate": score,
                "mean_return": statistics.fmean(
                    record.total_reward if record.policy_failure is None else 0.0
                    for record in records
                ),
                "mean_steps": statistics.fmean(record.steps for record in records),
                "mean_steps_to_first_success": _mean_present(
                    tuple(
                        _required_int(metrics, "first_success_step")
                        for metrics in final_metrics
                        if _required_int(metrics, "first_success_step") >= 0
                    )
                ),
                "mean_best_dense_reward": _mean_metric(
                    final_metrics,
                    "best_dense_reward",
                ),
                "mean_action_l2_norm": _mean_metric(
                    final_metrics,
                    "mean_action_l2_norm",
                ),
                "mean_proprioception_motion_l2": _mean_metric(
                    final_metrics,
                    "mean_proprioception_motion_l2",
                ),
                "mean_object_motion_l2": _mean_metric(
                    final_metrics,
                    "mean_object_motion_l2",
                ),
                "successful_episodes": successes,
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "traced_transitions": traced_transitions,
                "trace_transitions_omitted": omitted_transitions,
                "video_episodes": len(video_manifests),
                "video_episodes_omitted": len(records) - len(video_manifests),
                "video_capture_unavailable_episodes": video_unavailable,
                "video_camera": "agentview",
                "video_frame_shape": list(VIDEO_FRAME_SHAPE),
                "video_capture_interval_steps": video_capture_interval(
                    self._config.max_episode_steps
                ),
                "video_frame_cap_per_episode": VIDEO_MAX_FRAMES_PER_EPISODE,
                "video_artifacts": video_manifests,
            },
            artifacts=(trace, *video_artifacts),
        )


def _spec(config: RobosuiteConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=f"robosuite/{config.environment_id}/panda-state/success-rate-v1",
        description=(
            f"Complete robosuite {config.environment_id} with fixed Panda "
            "robots and the BASIC operational-space controller."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "proprioception": {
                    "type": "tensor",
                    "dtype": "float64",
                    "shape": [config.proprioception_size],
                },
                "objects": {
                    "type": "tensor",
                    "dtype": "float64",
                    "shape": [config.object_state_size],
                },
            },
        },
        action_space={
            "type": "array",
            "shape": [config.action_size],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "meaning": _action_meanings(config),
        },
        metadata={
            "environment": config.environment_id,
            "provider": "robosuite",
            "upstream_version": "1.5.2",
            "mujoco_version": ">=3.3.0,<3.4",
            "reward_mode": "upstream dense shaped reward where implemented",
            "success_scoring": "an Episode succeeds if _check_success is ever true",
        },
        environment_parameters={
            "profile": config.profile,
            "environment_id": config.environment_id,
            "robot": "Panda",
            "robot_count": config.robot_count,
            "environment_configuration": (
                "default" if config.robot_count == 1 else "opposed"
            ),
            "controller": "BASIC/OSC_POSE",
            "observation_mode": "proprioception_plus_object_state",
            "tensor_encoding": (
                "Observation fields are TensorValue objects, not indexable sequences. "
                "For float64 tensors, decode TensorValue.data as packed little-endian "
                "doubles, for example with struct.iter_unpack('<d', value.data)."
            ),
            "reward_shaping": True,
            "continuous_actions": True,
            "action_size": config.action_size,
            "action_handling": (
                "Every component must be a finite exact float in [-1,1]; "
                "invalid Actions are rejected rather than clipped or repaired."
            ),
            "max_episode_steps": config.max_episode_steps,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="success_rate",
        score_direction="maximize",
    )


def _action_meanings(config: RobosuiteConfig) -> list[PolicyValue]:
    components_per_robot = config.action_size // config.robot_count
    meanings: list[PolicyValue] = []
    for robot_index in range(config.robot_count):
        prefix = f"robot{robot_index}_" if config.robot_count > 1 else ""
        meanings.extend(
            [
                prefix + "end_effector_delta_x",
                prefix + "end_effector_delta_y",
                prefix + "end_effector_delta_z",
                prefix + "end_effector_delta_rotation_x",
                prefix + "end_effector_delta_rotation_y",
                prefix + "end_effector_delta_rotation_z",
            ]
        )
        if components_per_robot == 7:
            meanings.append(prefix + "gripper_effort")
    return meanings


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _success(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and any(
            type(transition.step.metrics) is dict
            and transition.step.metrics.get("success_ever") is True
            for transition in record.transitions
        )
    )


def _terminated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
    )


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _final_metrics(record: EpisodeRecord) -> dict[str, PolicyValue] | None:
    if record.policy_failure is not None or not record.transitions:
        return None
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("robosuite metrics are invalid")
    required = {
        "first_success_step",
        "best_dense_reward",
        "mean_action_l2_norm",
        "mean_proprioception_motion_l2",
        "mean_object_motion_l2",
    }
    if not required <= set(metrics):
        raise ValueError("robosuite metrics are incomplete")
    return metrics


def _required_int(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics[name]
    if type(value) is not int:
        raise ValueError(f"robosuite {name} metric is invalid")
    return value


def _mean_metric(
    metrics: Sequence[dict[str, PolicyValue]],
    name: str,
) -> float | None:
    values = tuple(item[name] for item in metrics)
    if any(type(value) not in {int, float} for value in values):
        raise ValueError(f"robosuite {name} metric is invalid")
    return _mean_present(tuple(float(cast(float | int, value)) for value in values))


def _mean_present(values: Sequence[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _trace(
    records: Sequence[EpisodeRecord],
    *,
    total_transitions: int,
) -> tuple[Artifact, int, int]:
    lines: list[bytes] = []
    traced_transitions = 0
    for episode_index, record in enumerate(records):
        step_indices = _trace_step_indices(record.steps)
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": (
                        "completed" if record.policy_failure is None else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": record.total_reward,
                    "success": _success(record),
                    "failure": record.policy_failure,
                    "traced_steps": len(step_indices),
                    "omitted_steps": record.steps - len(step_indices),
                }
            )
        )
        for step_index in step_indices:
            transition = record.transitions[step_index]
            observation = (
                record.initial_observation
                if step_index == 0
                else record.transitions[step_index - 1].step.observation
            )
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": _trace_value(observation),
                        "action": _trace_value(transition.action),
                        "reward": transition.step.reward,
                        "next_observation": _trace_value(transition.step.observation),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": _trace_value(trace_metrics(transition.step.metrics)),
                    }
                )
            )
            traced_transitions += 1
    return (
        Artifact(
            name="trace.jsonl",
            media_type="application/x-ndjson",
            content=b"".join(lines),
            retention="bulk",
        ),
        traced_transitions,
        total_transitions - traced_transitions,
    )


def _trace_step_indices(steps: int) -> tuple[int, ...]:
    if steps <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS:
        return tuple(range(steps))
    return tuple(range(_TRACE_PREFIX_STEPS)) + tuple(
        range(steps - _TRACE_SUFFIX_STEPS, steps)
    )


def _trace_value(value: PolicyValue) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is bytes:
        return {"$type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if type(value) is TensorValue:
        if value.dtype != "float64":
            raise ValueError("robosuite trace contains an unexpected Tensor dtype")
        return {
            "$type": "tensor",
            "dtype": value.dtype,
            "shape": list(value.shape),
            "values": [
                item[0] for item in struct.iter_unpack("<d", value.data)
            ],
        }
    if type(value) is list:
        return [_trace_value(item) for item in value]
    if type(value) is tuple:
        return [_trace_value(item) for item in value]
    if type(value) is dict:
        return {key: _trace_value(item) for key, item in value.items()}
    raise TypeError("unsupported robosuite trace value")


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


__all__ = ["RobosuiteBenchmark"]
