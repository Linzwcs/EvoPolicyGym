"""Single-agent Gymnasium-Robotics profiles and public feedback."""

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

from .config import RoboticsConfig
from .environment import RoboticsEnvironment
from .video import (
    VIDEO_FRAME_SHAPE,
    VIDEO_MAX_FRAMES_PER_EPISODE,
    trace_metrics,
    video_camera,
    video_camera_label,
    video_capture_interval,
    video_feedback,
)

_SEED_DOMAIN = b"evopolicygym-gymnasium-robotics/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
_KITCHEN_GOALS: dict[str, PolicyValue] = {
    "type": "object",
    "fields": {
        "bottom burner": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "top burner": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "light switch": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "slide cabinet": {"type": "tensor", "dtype": "float64", "shape": [1]},
        "hinge cabinet": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "microwave": {"type": "tensor", "dtype": "float64", "shape": [1]},
        "kettle": {"type": "tensor", "dtype": "float64", "shape": [7]},
    },
}


class RoboticsBenchmark:
    """Mean upstream Episode return for this Benchmark."""

    def __init__(self, config: RoboticsConfig | None = None) -> None:
        if config is None:
            config = RoboticsConfig()
        if type(config) is not RoboticsConfig:
            raise TypeError("config must be RoboticsConfig")
        self._config = config
        self._failure_return = _failure_return(config)
        self._spec = _spec(config, failure_return=self._failure_return)

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
            EpisodeSpec(environment_seed=_seed(split, seed, index)) for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return RoboticsEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        success_rate = successes / len(records)
        score = statistics.fmean(
            record.total_reward if record.policy_failure is None else self._failure_return
            for record in records
        )
        traced = records[:_MAX_TRACED_EPISODES]
        trace, traced_transitions, omitted_transitions = _trace(
            traced,
            total_transitions=sum(record.steps for record in records),
        )
        camera = video_camera(self._config.profile)
        capture_interval = video_capture_interval(self._config.max_episode_steps)
        video_artifacts, video_manifests, video_unavailable = video_feedback(
            records,
            profile=self._config.profile,
            camera=video_camera_label(camera),
            capture_interval=capture_interval,
        )
        video_episode_count = _artifact_episode_count(video_manifests, "preview_artifact")
        frame_evidence_episode_count = _artifact_episode_count(
            video_manifests,
            "evidence_artifact",
        )
        final_metrics = tuple(
            metrics for record in records if (metrics := _final_metrics(record)) is not None
        )
        successful = tuple(record for record in records if _success(record))
        zero_action_fractions = tuple(
            _required_int(metrics, "zero_action_count") / _required_int(metrics, "step_count")
            for metrics in final_metrics
        )
        saturated_action_fractions = tuple(
            _required_int(
                metrics,
                "cumulative_saturated_action_component_count",
            )
            / (_required_int(metrics, "step_count") * self._config.action_size)
            for metrics in final_metrics
        )
        no_state_change_fractions = tuple(
            _required_int(metrics, "no_state_change_count") / _required_int(metrics, "step_count")
            for metrics in final_metrics
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Solved {successes}/{len(records)} "
                    f"{self._config.profile} Episodes "
                    f"({success_rate:.3f} success rate) with {score:.3f} mean return."
                ),
                "success_rate": success_rate,
                "mean_return": score,
                "mean_steps": statistics.fmean(r.steps for r in records),
                "mean_steps_to_first_success": _mean_present(
                    tuple(
                        _required_int(metrics, "first_success_step")
                        for metrics in final_metrics
                        if _required_int(metrics, "first_success_step") >= 0
                    )
                ),
                "mean_steps_on_success": (
                    statistics.fmean(record.steps for record in successful) if successful else None
                ),
                "mean_initial_goal_distance": _mean_metric(
                    final_metrics,
                    "initial_goal_distance",
                ),
                "mean_final_goal_distance": _mean_metric(
                    final_metrics,
                    "goal_distance",
                ),
                "mean_best_goal_distance": _mean_metric(
                    final_metrics,
                    "best_goal_distance",
                ),
                "mean_goal_distance_improvement_from_initial": _mean_metric(
                    final_metrics,
                    "goal_distance_improvement_from_initial",
                ),
                "mean_final_goal_position_distance": _mean_metric(
                    final_metrics,
                    "goal_position_distance",
                ),
                "mean_final_goal_rotation_distance": _mean_metric(
                    final_metrics,
                    "goal_rotation_distance",
                ),
                "mean_successful_step_fraction": _mean_metric(
                    final_metrics,
                    "successful_step_fraction",
                ),
                "mean_success_lost_count": _mean_metric(
                    final_metrics,
                    "success_lost_count",
                ),
                "mean_action_l2_norm": _mean_metric(
                    final_metrics,
                    "mean_action_l2_norm",
                ),
                "mean_action_max_abs": _mean_metric(
                    final_metrics,
                    "mean_action_max_abs",
                ),
                "mean_saturated_action_component_fraction": _mean_present(
                    saturated_action_fractions
                ),
                "mean_zero_action_fraction": _mean_present(zero_action_fractions),
                "mean_state_motion_l2": _mean_metric(
                    final_metrics,
                    "mean_state_motion_l2",
                ),
                "mean_no_state_change_fraction": _mean_present(no_state_change_fractions),
                "mean_completed_tasks": _mean_metric(
                    final_metrics,
                    "completed_tasks",
                ),
                "mean_task_completion_fraction": _mean_metric(
                    final_metrics,
                    "task_completion_fraction",
                ),
                "episodes": len(records),
                "successful_episodes": successes,
                "terminated_episodes": sum(_terminated(r) for r in records),
                "truncated_episodes": sum(_truncated(r) for r in records),
                "policy_failures": sum(r.policy_failure is not None for r in records),
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "trace_prefix_steps": _TRACE_PREFIX_STEPS,
                "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
                "traced_transitions": traced_transitions,
                "trace_transitions_omitted": omitted_transitions,
                "video_episodes": video_episode_count,
                "video_episode_results": len(video_manifests),
                "video_episodes_without_gif": len(records) - video_episode_count,
                "rendered_frame_evidence_episodes": frame_evidence_episode_count,
                "rendered_frame_evidence_format": "lossless NPZ",
                "video_capture_unavailable_episodes": video_unavailable,
                "video_camera": video_camera_label(camera),
                "video_frame_shape": list(VIDEO_FRAME_SHAPE),
                "video_capture_interval_steps": capture_interval,
                "video_frame_cap_per_episode": VIDEO_MAX_FRAMES_PER_EPISODE,
                "video_artifacts": video_manifests,
                "rendered_frame_evidence": video_manifests,
            },
            artifacts=(trace, *video_artifacts),
        )


def _artifact_episode_count(manifests: Sequence[PolicyValue], key: str) -> int:
    return sum(
        type(manifest) is dict and type(manifest.get(key)) is str
        for manifest in manifests
    )


def _spec(config: RoboticsConfig, *, failure_return: float) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=f"gymnasium-robotics/{config.environment_id}/mean-return-v1",
        description=(
            f"Complete Gymnasium-Robotics' {config.profile} task. "
            "Maximize mean upstream Episode return; success rate remains a "
            "reported task-completion outcome."
        ),
        observation_space=_observation_space(config),
        action_space={
            "type": "array",
            "shape": [config.action_size],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
        },
        metadata={
            "environment": config.environment_id,
            "provider": "Gymnasium-Robotics",
            "upstream_version": "1.4.2",
            "reward_mode": _reward_mode(config),
            "failure_return": failure_return,
            "success_persistence": (
                "Success is scored if any transition reports the upstream "
                "success condition; most profiles continue until TimeLimit, "
                "so current success and later regression are traced."
            ),
        },
        environment_parameters={
            "profile": config.profile,
            "family": config.family,
            "continuous_actions": True,
            "action_size": config.action_size,
            "action_dtype": config.action_dtype,
            "tensor_encoding": (
                "Observation tensors are TensorValue objects, not iterable sequences. "
                "For float64 tensors, decode TensorValue.data as packed little-endian "
                "doubles, for example with struct.iter_unpack('<d', tensor.data)."
            ),
            "action_handling": (
                "Every component must be a finite float in [-1,1]; invalid "
                "Actions are rejected rather than clipped or repaired."
            ),
            "reward_semantics": _reward_semantics(config),
            "success_condition": _success_condition(config),
            "goal_diagnostics": (
                "Distances are computed only from Policy-visible goal/state "
                "tensors. Manipulation goal_distance is 10*position_error + "
                "quaternion_angle; Adroit uses public task-specific state "
                "components where an exact progress value is available."
            ),
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_space(config: RoboticsConfig) -> PolicyValue:
    state: PolicyValue = {
        "type": "tensor",
        "dtype": "float64",
        "shape": [config.observation_size],
    }
    if config.goal_size is None:
        return state
    goal: PolicyValue
    if config.goal_size == -1:
        goal = _KITCHEN_GOALS
    else:
        goal = {
            "type": "tensor",
            "dtype": "float64",
            "shape": [config.goal_size],
        }
    return {
        "type": "object",
        "fields": {
            "observation": state,
            "achieved_goal": goal,
            "desired_goal": goal,
        },
    }


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
            type(item.step.metrics) is dict and item.step.metrics.get("success") is True
            for item in record.transitions
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
        raise ValueError("Gymnasium-Robotics metrics are invalid")
    required = {
        "step_count",
        "first_success_step",
        "initial_goal_distance",
        "goal_distance",
        "best_goal_distance",
        "goal_distance_improvement_from_initial",
        "goal_position_distance",
        "goal_rotation_distance",
        "successful_step_fraction",
        "success_lost_count",
        "mean_action_l2_norm",
        "mean_action_max_abs",
        "cumulative_saturated_action_component_count",
        "zero_action_count",
        "mean_state_motion_l2",
        "no_state_change_count",
        "completed_tasks",
        "task_completion_fraction",
    }
    if not required <= set(metrics):
        raise ValueError("Gymnasium-Robotics metrics are incomplete")
    return metrics


def _required_int(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics[name]
    if type(value) is not int:
        raise ValueError(f"Gymnasium-Robotics {name} metric is invalid")
    return value


def _mean_metric(
    metrics: Sequence[dict[str, PolicyValue]],
    name: str,
) -> float | None:
    values = tuple(value for item in metrics if (value := item[name]) is not None)
    if any(type(value) not in {int, float} for value in values):
        raise ValueError(f"Gymnasium-Robotics {name} metric is invalid")
    return _mean_present(tuple(float(cast(float | int, value)) for value in values))


def _mean_present(values: Sequence[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _reward_mode(config: RoboticsConfig) -> str:
    if config.family == "adroit":
        return "upstream dense shaping"
    if config.family == "franka-kitchen":
        return "newly completed task count"
    return "upstream sparse goal reward"


def _failure_return(config: RoboticsConfig) -> float:
    if config.family in {"fetch", "shadow-hand", "shadow-hand-touch"}:
        return -float(config.max_episode_steps + 1)
    if config.family in {"maze", "franka-kitchen"}:
        return -1.0
    return -1_000_000.0


def _reward_semantics(config: RoboticsConfig) -> str:
    if config.family in {"fetch", "shadow-hand", "shadow-hand-touch"}:
        return "-1 until the current goal is achieved, then 0"
    if config.family == "maze":
        return "1 while within 0.45 of the goal, otherwise 0"
    if config.family == "franka-kitchen":
        return "number of newly completed kitchen tasks on this step"
    return "profile-specific upstream dense shaping reward"


def _success_condition(config: RoboticsConfig) -> PolicyValue:
    if config.family == "fetch":
        return {"euclidean_goal_distance_strictly_below": 0.05}
    if config.family == "maze":
        return {"euclidean_goal_distance_at_most": 0.45}
    if config.profile == "hand-reach":
        return {"euclidean_goal_distance_strictly_below": 0.01}
    if config.profile.startswith("hand-manipulate-pen"):
        return {
            "position_distance_strictly_below": 0.05,
            "rotation_angle_strictly_below": 0.1,
            "z_rotation_ignored": True,
        }
    if config.family in {"shadow-hand", "shadow-hand-touch"}:
        return {
            "position_distance_strictly_below": 0.01,
            "rotation_angle_strictly_below": 0.1,
        }
    if config.profile == "adroit-hand-door":
        return {"door_hinge_angle_at_least": 1.35}
    if config.profile == "adroit-hand-hammer":
        return {"nail_to_goal_distance_strictly_below": 0.01}
    if config.profile == "adroit-hand-pen":
        return {
            "position_distance_strictly_below": 0.075,
            "orientation_similarity_strictly_above": 0.95,
        }
    if config.profile == "adroit-hand-relocate":
        return {"object_goal_distance_strictly_below": 0.1}
    return {"all_kitchen_tasks_completed": True}


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
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
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
        ),
        traced_transitions,
        total_transitions - traced_transitions,
    )


def _trace_step_indices(steps: int) -> tuple[int, ...]:
    if steps <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS:
        return tuple(range(steps))
    return tuple(range(_TRACE_PREFIX_STEPS)) + tuple(range(steps - _TRACE_SUFFIX_STEPS, steps))


_TENSOR_FORMATS = {
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}


def _trace_value(value: PolicyValue) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is bytes:
        return {
            "$type": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is TensorValue:
        if value.dtype == "bool":
            values: list[object] = [bool(item) for item in value.data]
        else:
            format_code = _TENSOR_FORMATS[value.dtype]
            values = [
                item[0]
                for item in struct.iter_unpack(
                    f"<{format_code}",
                    value.data,
                )
            ]
        return {
            "$type": "tensor",
            "dtype": value.dtype,
            "shape": list(value.shape),
            "values": values,
        }
    if type(value) is list:
        return [_trace_value(item) for item in value]
    if type(value) is tuple:
        return [_trace_value(item) for item in value]
    if type(value) is dict:
        return {key: _trace_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported Robotics trace value: {type(value).__name__}")


def _json(document: dict[str, object]) -> bytes:
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


__all__ = ["RoboticsBenchmark"]
