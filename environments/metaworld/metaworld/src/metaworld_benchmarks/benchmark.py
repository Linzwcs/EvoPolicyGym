"""MetaWorld MT collections with deterministic Episode plans."""

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

from .config import MetaWorldConfig
from .environment import MetaWorldEnvironment
from .video import (
    VIDEO_CAMERA,
    VIDEO_FRAME_SHAPE,
    VIDEO_MAX_FRAMES_PER_EPISODE,
    trace_metrics,
    video_capture_interval,
    video_feedback,
)

_SEED_DOMAIN = b"evopolicygym-metaworld/episode-seed/v1\0"
_TASK_DOMAIN = b"evopolicygym-metaworld/task-offset/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 500
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32


class MetaWorldBenchmark:
    """Success rate for one fixed MetaWorld MT task collection."""

    def __init__(self, config: MetaWorldConfig | None = None) -> None:
        if config is None:
            config = MetaWorldConfig()
        if type(config) is not MetaWorldConfig:
            raise TypeError("config must be MetaWorldConfig")
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
        task_count = len(self._config.task_names)
        offset = _task_offset(split, seed, task_count)
        return tuple(
            EpisodeSpec(
                environment_seed=_seed(split, seed, index),
                scenario=(
                    None if task_count == 1 else {"task_index": (offset + index) % task_count}
                ),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return MetaWorldEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        score = successes / len(records)
        traced = records[:_MAX_TRACED_EPISODES]
        trace, traced_transitions, omitted_transitions = _trace(
            traced,
            total_transitions=sum(record.steps for record in records),
        )
        capture_interval = video_capture_interval(_MAX_EPISODE_STEPS)
        video_artifacts, video_manifests, video_unavailable = video_feedback(
            records,
            profile=self._config.profile,
            capture_interval=capture_interval,
        )
        final_metrics = tuple(
            metrics for record in records if (metrics := _final_metrics(record)) is not None
        )
        successful = tuple(record for record in records if _success(record))
        task_episode_counts: dict[str, int] = {}
        task_success_counts: dict[str, int] = {}
        task_returns: dict[str, list[float]] = {}
        for record in records:
            task_name = _record_task_name(record, config=self._config)
            task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1
            task_success_counts[task_name] = task_success_counts.get(task_name, 0) + int(
                _success(record)
            )
            task_returns.setdefault(task_name, []).append(
                record.total_reward if record.policy_failure is None else 0.0
            )
        task_success_rates = {
            name: task_success_counts[name] / count for name, count in task_episode_counts.items()
        }
        task_mean_returns = {
            name: statistics.fmean(task_returns[name]) for name in task_episode_counts
        }
        task_episode_count_values: dict[str, PolicyValue] = dict(task_episode_counts)
        task_success_rate_values: dict[str, PolicyValue] = dict(task_success_rates)
        task_mean_return_values: dict[str, PolicyValue] = dict(task_mean_returns)
        zero_action_fractions = tuple(
            _required_int(metrics, "zero_action_count") / _required_int(metrics, "step_count")
            for metrics in final_metrics
        )
        saturated_action_fractions = tuple(
            _required_int(
                metrics,
                "cumulative_saturated_action_component_count",
            )
            / (_required_int(metrics, "step_count") * 4)
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
                    f"Solved {successes}/{len(records)} MetaWorld Episodes "
                    f"({score:.3f} success rate)."
                ),
                "success_rate": score,
                "mean_return": statistics.fmean(
                    r.total_reward if r.policy_failure is None else 0.0 for r in records
                ),
                "mean_steps": statistics.fmean(r.steps for r in records),
                "mean_steps_on_success": (
                    statistics.fmean(record.steps for record in successful) if successful else None
                ),
                "mean_steps_to_first_success": _mean_present(
                    tuple(
                        _required_int(metrics, "first_success_step")
                        for metrics in final_metrics
                        if _required_int(metrics, "first_success_step") >= 0
                    )
                ),
                "mean_final_reward": _mean_metric(
                    final_metrics,
                    "unscaled_reward",
                ),
                "mean_best_reward": _mean_metric(
                    final_metrics,
                    "best_reward",
                ),
                "mean_best_near_object": _mean_metric(
                    final_metrics,
                    "best_near_object",
                ),
                "mean_best_grasp_reward": _mean_metric(
                    final_metrics,
                    "best_grasp_reward",
                ),
                "mean_best_in_place_reward": _mean_metric(
                    final_metrics,
                    "best_in_place_reward",
                ),
                "mean_best_obj_to_target": _mean_metric(
                    final_metrics,
                    "best_obj_to_target",
                ),
                "mean_successful_step_fraction": _mean_metric(
                    final_metrics,
                    "successful_step_fraction",
                ),
                "mean_success_lost_count": _mean_metric(
                    final_metrics,
                    "success_lost_count",
                ),
                "mean_grasp_success_lost_count": _mean_metric(
                    final_metrics,
                    "grasp_success_lost_count",
                ),
                "mean_action_l2_norm": _mean_metric(
                    final_metrics,
                    "mean_action_l2_norm",
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
                "task_episode_counts": task_episode_count_values,
                "task_success_rates": task_success_rate_values,
                "task_mean_returns": task_mean_return_values,
                "episodes": len(records),
                "successful_episodes": successes,
                "terminated_episodes": sum(_terminated(r) for r in records),
                "truncated_episodes": sum(_truncated(r) for r in records),
                "policy_failures": sum(r.policy_failure is not None for r in records),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "trace_prefix_steps": _TRACE_PREFIX_STEPS,
                "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
                "traced_transitions": traced_transitions,
                "trace_transitions_omitted": omitted_transitions,
                "video_episodes": len(video_artifacts),
                "video_episode_results": len(video_manifests),
                "video_episodes_without_gif": len(records) - len(video_artifacts),
                "video_capture_unavailable_episodes": video_unavailable,
                "video_camera": VIDEO_CAMERA,
                "video_frame_shape": list(VIDEO_FRAME_SHAPE),
                "video_capture_interval_steps": capture_interval,
                "video_frame_cap_per_episode": VIDEO_MAX_FRAMES_PER_EPISODE,
                "video_artifacts": video_manifests,
            },
            artifacts=(trace, *video_artifacts),
        )


def _spec(config: MetaWorldConfig) -> BenchmarkSpec:
    task_count = len(config.task_names)
    state: PolicyValue = {
        "type": "tensor",
        "dtype": "float64",
        "shape": [39],
        "goal_observable": True,
    }
    observation: PolicyValue
    if task_count == 1:
        observation = state
    else:
        observation = {
            "type": "object",
            "fields": {
                "state": state,
                "task": {
                    "type": "tensor",
                    "dtype": "bool",
                    "shape": [task_count],
                    "encoding": "one_hot",
                },
            },
        }
    benchmark_name = (
        f"MT1/{config.profile}"
        if config.collection_name == "mt1"
        else config.collection_name.upper()
    )
    return BenchmarkSpec(
        id=f"metaworld/{benchmark_name}/success-rate-v1",
        description=(
            f"Complete tasks from MetaWorld's {benchmark_name} collection. "
            "Maximize the fraction of Episodes reaching the public success "
            "condition."
        ),
        observation_space=observation,
        action_space={
            "type": "array",
            "shape": [4],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "meaning": [
                "end_effector_delta_x",
                "end_effector_delta_y",
                "end_effector_delta_z",
                "gripper_effort",
            ],
        },
        metadata={
            "environment": "Meta-World/MT1",
            "provider": "MetaWorld",
            "upstream_version": "3.1.1",
            "reward_function_version": "v2",
            "reward_mode": "upstream dense shaped reward in [0,10]",
            "success_persistence": (
                "Success is scored if reached on any transition; the "
                "Environment does not terminate on success, so later loss is traced."
            ),
        },
        environment_parameters={
            "profile": config.profile,
            "collection": config.collection_name,
            "task_count": task_count,
            "task_index_to_environment": list(config.task_names),
            "goal_observable": True,
            "continuous_actions": True,
            "action_size": 4,
            "tensor_encoding": (
                "State observations are TensorValue objects, not iterable sequences. "
                "Decode float64 TensorValue.data as packed little-endian doubles, "
                "for example with struct.iter_unpack('<d', tensor.data)."
            ),
            "action_handling": (
                "Every component must be a finite float in [-1,1]; invalid "
                "Actions are rejected rather than clipped or repaired."
            ),
            "horizon": (
                "All tasks continue for at most 500 steps because terminate_on_success is disabled."
            ),
            "feedback_diagnostics": (
                "Upstream near_object, grasp_reward, in_place_reward, "
                "obj_to_target, grasp_success, and unscaled_reward are traced "
                "with per-step deltas and Episode best values."
            ),
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="success_rate",
        score_direction="maximize",
    )


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _task_offset(split: str, seed: int, task_count: int) -> int:
    if task_count == 1:
        return 0
    digest = hashlib.sha256()
    digest.update(_TASK_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big") % task_count


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
        raise ValueError("MetaWorld metrics are invalid")
    required = {
        "step_count",
        "first_success_step",
        "unscaled_reward",
        "best_reward",
        "best_near_object",
        "best_grasp_reward",
        "best_in_place_reward",
        "best_obj_to_target",
        "successful_step_fraction",
        "success_lost_count",
        "grasp_success_lost_count",
        "mean_action_l2_norm",
        "cumulative_saturated_action_component_count",
        "zero_action_count",
        "mean_state_motion_l2",
        "no_state_change_count",
    }
    if not required <= set(metrics):
        raise ValueError("MetaWorld metrics are incomplete")
    return metrics


def _required_int(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics[name]
    if type(value) is not int:
        raise ValueError(f"MetaWorld {name} metric is invalid")
    return value


def _mean_metric(
    metrics: Sequence[dict[str, PolicyValue]],
    name: str,
) -> float | None:
    values = tuple(item[name] for item in metrics)
    if any(type(value) not in {int, float} for value in values):
        raise ValueError(f"MetaWorld {name} metric is invalid")
    return _mean_present(tuple(float(cast(float | int, value)) for value in values))


def _mean_present(values: Sequence[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _record_task_name(
    record: EpisodeRecord,
    *,
    config: MetaWorldConfig,
) -> str:
    if len(config.task_names) == 1:
        return config.task_names[0]
    scenario = record.episode.scenario
    if type(scenario) is not dict or set(scenario) != {"task_index"}:
        raise ValueError("MetaWorld Episode task identity is invalid")
    index = scenario["task_index"]
    if type(index) is not int or not 0 <= index < len(config.task_names):
        raise ValueError("MetaWorld Episode task identity is invalid")
    return config.task_names[index]


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
    raise TypeError(f"unsupported MetaWorld trace value: {type(value).__name__}")


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


__all__ = ["MetaWorldBenchmark"]
