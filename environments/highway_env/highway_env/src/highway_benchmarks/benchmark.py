"""The canonical HighwayEnv profiles with bounded public feedback."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy
from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
    Transition,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import HighwayConfig
from .environment import HighwayEnvironment
from .visual import (
    VISUAL_MAX_FRAMES_PER_EPISODE,
    trace_metrics,
    visual_capture_interval,
    visual_feedback,
)

_SEED_DOMAIN = b"evopolicygym-highway-env/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_STEPS_PER_EPISODE = 48
_TRACE_EDGE_STEPS = 12
_MAX_EVENT_STEPS = 12
_STANDARD_DISCRETE_ACTIONS = {
    0: "lane_left",
    1: "idle",
    2: "lane_right",
    3: "faster",
    4: "slower",
}
_LONGITUDINAL_DISCRETE_ACTIONS = {0: "slower", 1: "idle", 2: "faster"}
_KINEMATICS_PROFILES = {
    "highway": ((5, 5), ("presence", "x", "y", "vx", "vy")),
    "merge": ((5, 5), ("presence", "x", "y", "vx", "vy")),
    "roundabout": ((5, 5), ("presence", "x", "y", "vx", "vy")),
    "intersection": (
        (15, 7),
        ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h"),
    ),
    "exit": (
        (15, 7),
        ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h"),
    ),
}
_TTC_PROFILES = {"two-way": (3, 3, 5), "u-turn": (3, 3, 16)}
_PARKING_FEATURES = ("x", "y", "vx", "vy", "cos_h", "sin_h")
_OCCUPANCY_FEATURES = ("presence", "on_road")
_LANE_STATE_FEATURES = (
    "y_position",
    "heading",
    "lateral_speed",
    "yaw_rate",
)


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    step_indices: tuple[int, ...]
    config: HighwayConfig

    @property
    def observation_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/observations.npz"


class HighwayBenchmark:
    """Mean return for one fixed HighwayEnv profile."""

    def __init__(self, config: HighwayConfig | None = None) -> None:
        if config is None:
            config = HighwayConfig()
        if type(config) is not HighwayConfig:
            raise TypeError("config must be HighwayConfig")
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
        return HighwayEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        failure_return = -float(self._config.max_episode_steps)
        returns = tuple(
            record.total_reward
            if record.policy_failure is None
            else failure_return
            for record in records
        )
        score = statistics.fmean(returns)
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        traced = tuple(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                step_indices=_trace_step_indices(record),
                config=self._config,
            )
            for episode_index, record in enumerate(
                records[:_MAX_TRACED_EPISODES]
            )
        )
        observation_artifacts: list[Artifact] = []
        observation_manifests: list[PolicyValue] = []
        for episode in traced:
            artifact, fields = _observation_artifact(episode)
            observation_artifacts.append(artifact)
            observation_manifests.append(
                {
                    "episode_index": episode.episode_index,
                    "artifact": episode.observation_artifact_name,
                    "artifact_sha256": hashlib.sha256(
                        artifact.content
                    ).hexdigest(),
                    "fields": fields,
                    "stored_transition_pairs": len(episode.step_indices),
                    "step_indices": list(episode.step_indices),
                    "omitted_steps": (
                        episode.record.steps - len(episode.step_indices)
                    ),
                }
            )
        traced_steps = {
            episode.episode_index: len(episode.step_indices)
            for episode in traced
        }
        capture_interval = visual_capture_interval(self._config.max_episode_steps)
        visual_artifacts, visual_manifests, visual_unavailable = visual_feedback(
            records,
            profile=self._config.profile,
            capture_interval=capture_interval,
        )
        preview_episode_count = _artifact_episode_count(
            visual_manifests,
            "preview_artifact",
        )
        frame_evidence_episode_count = _artifact_episode_count(
            visual_manifests,
            "evidence_artifact",
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} "
                    f"{self._config.profile} Episodes."
                ),
                "mean_return": score,
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(r) for r in records),
                "truncated_episodes": sum(_truncated(r) for r in records),
                "crashed_episodes": sum(_reached(r, "crashed") for r in records),
                "successful_episodes": sum(
                    _reached(r, "is_success") for r in records
                ),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_return": failure_return,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        episode_index=episode_index,
                        failure_return=failure_return,
                        traced_steps=traced_steps.get(episode_index, 0),
                        config=self._config,
                    )
                    for episode_index, record in enumerate(summarized)
                ],
                "summarized_episodes": len(summarized),
                "summary_episodes_omitted": len(records) - len(summarized),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "traced_steps": sum(traced_steps.values()),
                "trace_steps_omitted": sum(
                    episode.record.steps - len(episode.step_indices)
                    for episode in traced
                ),
                "trace_step_cap_per_episode": (
                    _MAX_TRACED_STEPS_PER_EPISODE
                ),
                "trace_selection": (
                    "Every short Episode is complete. Long Episodes retain "
                    "the first and last steps, bounded crash/success/terminal "
                    "events, and an even sample of remaining steps."
                ),
                "trace_format": (
                    "trace.jsonl contains profile-specific semantic snapshots "
                    "and references every selected lossless observation field "
                    "in per-Episode observations.npz artifacts. Omitted steps "
                    "and Episodes are reported explicitly."
                ),
                "observation_artifacts": observation_manifests,
                "rendered_frame_evidence_episodes": frame_evidence_episode_count,
                "rendered_frame_evidence_format": "lossless NPZ",
                "visual_preview_episodes": preview_episode_count,
                "visual_episode_results": len(visual_manifests),
                "visual_capture_unavailable_episodes": visual_unavailable,
                "visual_renderer": "HighwayEnv rgb_array",
                "visual_capture_interval_steps": capture_interval,
                "visual_frame_cap_per_episode": VISUAL_MAX_FRAMES_PER_EPISODE,
                "rendered_frame_evidence": visual_manifests,
            },
            artifacts=(
                _trace(traced, failure_return=failure_return),
                *observation_artifacts,
                *visual_artifacts,
            ),
        )


def _artifact_episode_count(manifests: Sequence[PolicyValue], key: str) -> int:
    return sum(
        type(manifest) is dict and type(manifest.get(key)) is str
        for manifest in manifests
    )


def _spec(config: HighwayConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        meanings = _continuous_action_meanings(config)
        action_space = {
            "type": "array",
            "shape": [config.action_size],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "meaning": list(meanings),
        }
    else:
        discrete_meanings = _discrete_action_meanings(config)
        action_space = {
            "type": "discrete",
            "values": list(range(config.action_size)),
            "meaning": {
                str(action): meaning for action, meaning in discrete_meanings.items()
            },
        }
    return BenchmarkSpec(
        id=f"highway-env/{config.environment_id}/mean-return-v1",
        description=(
            f"Control the ego vehicle in HighwayEnv's {config.profile} task. "
            "Maximize mean return while following the selected task's "
            "driving objective."
        ),
        observation_space=_observation_space(config),
        action_space=action_space,
        metadata={
            "environment": config.environment_id,
            "provider": "HighwayEnv",
            "upstream_version": "1.12",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters={
            "profile": config.profile,
            "observation_kind": config.observation_kind,
            "continuous_actions": config.continuous,
            "action_size": config.action_size,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_space(config: HighwayConfig) -> PolicyValue:
    if config.profile in _KINEMATICS_PROFILES:
        shape, features = _KINEMATICS_PROFILES[config.profile]
        return {
            "type": "tensor",
            "dtype": "float32",
            "shape": list(shape),
            "rows": "ego vehicle followed by observed nearby vehicles",
            "features": list(features),
            "normalization": "upstream HighwayEnv profile configuration",
        }
    if config.profile in _TTC_PROFILES:
        return {
            "type": "tensor",
            "dtype": "float32",
            "shape": list(_TTC_PROFILES[config.profile]),
            "axes": ["candidate_speed", "relative_lane", "future_time"],
            "values": "predicted collision cost in [0, 1]",
        }
    if config.profile == "parking":
        field: dict[str, PolicyValue] = {
            "type": "tensor",
            "dtype": "float64",
            "shape": [6],
            "features": list(_PARKING_FEATURES),
        }
        return {
            "type": "object",
            "fields": {
                "observation": field,
                "achieved_goal": field,
                "desired_goal": field,
            },
        }
    if config.profile == "racetrack":
        return {
            "type": "tensor",
            "dtype": "float32",
            "shape": [2, 12, 12],
            "axes": ["feature", "longitudinal_cell", "lateral_cell"],
            "features": list(_OCCUPANCY_FEATURES),
            "grid_size_metres": [[-18, 18], [-18, 18]],
            "grid_step_metres": [3, 3],
            "aligned_to_vehicle_axes": True,
        }
    if config.profile == "lane-keeping":
        field = {
            "type": "tensor",
            "dtype": "float64",
            "shape": [4, 1],
            "features": list(_LANE_STATE_FEATURES),
        }
        return {
            "type": "object",
            "fields": {
                "state": field,
                "derivative": field,
                "reference_state": field,
            },
        }
    raise ValueError("HighwayEnv observation profile is invalid")


def _discrete_action_meanings(config: HighwayConfig) -> dict[int, str]:
    return (
        _LONGITUDINAL_DISCRETE_ACTIONS
        if config.profile == "intersection"
        else _STANDARD_DISCRETE_ACTIONS
    )


def _continuous_action_meanings(config: HighwayConfig) -> tuple[str, ...]:
    return (
        ("acceleration", "steering")
        if config.profile == "parking"
        else ("steering",)
    )


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


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


def _reached(record: EpisodeRecord, name: str) -> bool:
    return any(
        type(item.step.metrics) is dict
        and item.step.metrics.get(name) is True
        for item in record.transitions
    )


def _episode_summary(
    record: EpisodeRecord,
    *,
    episode_index: int,
    failure_return: float,
    traced_steps: int,
    config: HighwayConfig,
) -> PolicyValue:
    action_counts: dict[str, int] = {}
    control_totals = [0.0] * config.action_size
    control_absolute_totals = [0.0] * config.action_size
    speeds: list[float] = []
    crash_step: int | None = None
    success_step: int | None = None
    for step_index, transition in enumerate(record.transitions):
        action = _trace_action(transition.action, config=config)
        if isinstance(action, int):
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        else:
            for index, value in enumerate(cast(list[float], action)):
                assert isinstance(value, float)
                control_totals[index] += value
                control_absolute_totals[index] += abs(value)
        metrics = transition.step.metrics
        if isinstance(metrics, dict):
            speed = metrics.get("speed")
            if isinstance(speed, float):
                speeds.append(speed)
            if metrics.get("crashed") is True and crash_step is None:
                crash_step = step_index
            if metrics.get("is_success") is True and success_step is None:
                success_step = step_index
    used_action_counts: dict[str, PolicyValue] = {
        action: count
        for action, count in sorted(
            action_counts.items(), key=lambda item: int(item[0])
        )
    }
    control_summary: dict[str, PolicyValue] = {}
    if config.continuous and record.steps:
        control_summary = {
            name: {
                "mean": control_totals[index] / record.steps,
                "mean_absolute": (
                    control_absolute_totals[index] / record.steps
                ),
            }
            for index, name in enumerate(_continuous_action_meanings(config))
        }
    speed_summary: dict[str, PolicyValue] = {}
    if speeds:
        speed_summary = {
            "minimum": min(speeds),
            "mean": statistics.fmean(speeds),
            "maximum": max(speeds),
            "final": speeds[-1],
        }
    return {
        "episode_index": episode_index,
        "status": (
            "completed" if record.policy_failure is None else "policy_failed"
        ),
        "return": (
            record.total_reward if record.policy_failure is None else None
        ),
        "scored_return": (
            record.total_reward
            if record.policy_failure is None
            else failure_return
        ),
        "steps": record.steps,
        "terminated": _terminated(record),
        "truncated": _truncated(record),
        "crashed": crash_step is not None,
        "crash_step": crash_step,
        "is_success": success_step is not None,
        "success_step": success_step,
        "failure": record.policy_failure,
        "action_counts": used_action_counts,
        "control_summary": control_summary,
        "speed_summary": speed_summary,
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _trace_step_indices(record: EpisodeRecord) -> tuple[int, ...]:
    if record.steps <= _MAX_TRACED_STEPS_PER_EPISODE:
        return tuple(range(record.steps))
    selected = set(range(_TRACE_EDGE_STEPS))
    selected.update(range(record.steps - _TRACE_EDGE_STEPS, record.steps))
    event_steps = tuple(
        step_index
        for step_index, transition in enumerate(record.transitions)
        if _event_transition(transition) and step_index not in selected
    )
    selected.update(_even_sample(event_steps, _MAX_EVENT_STEPS))
    remaining_capacity = _MAX_TRACED_STEPS_PER_EPISODE - len(selected)
    remaining_steps = tuple(
        step_index
        for step_index in range(record.steps)
        if step_index not in selected
    )
    selected.update(_even_sample(remaining_steps, remaining_capacity))
    return tuple(sorted(selected))


def _event_transition(transition: Transition) -> bool:
    metrics = transition.step.metrics
    return bool(
        transition.step.terminated
        or transition.step.truncated
        or (
            isinstance(metrics, dict)
            and (
                metrics.get("crashed") is True
                or metrics.get("is_success") is True
            )
        )
    )


def _even_sample(values: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0 or not values:
        return ()
    if len(values) <= count:
        return tuple(values)
    if count == 1:
        return (values[len(values) // 2],)
    return tuple(
        values[index * (len(values) - 1) // (count - 1)]
        for index in range(count)
    )


def _trace(
    episodes: Sequence[_TracedEpisode],
    *,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode in episodes:
        record = episode.record
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode.episode_index,
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": (
                        record.total_reward
                        if record.policy_failure is None
                        else None
                    ),
                    "scored_return": (
                        record.total_reward
                        if record.policy_failure is None
                        else failure_return
                    ),
                    "failure": record.policy_failure,
                    "traced_steps": len(episode.step_indices),
                    "omitted_steps": (
                        record.steps - len(episode.step_indices)
                    ),
                    "initial_observation": _observation_reference(
                        episode,
                        observation=record.initial_observation,
                        kind="initial",
                        trace_index=None,
                    ),
                }
            )
        )
        for trace_index, step_index in enumerate(episode.step_indices):
            transition = record.transitions[step_index]
            action = _trace_action(
                transition.action,
                config=episode.config,
            )
            decision_observation = (
                record.initial_observation
                if step_index == 0
                else record.transitions[step_index - 1].step.observation
            )
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode.episode_index,
                        "step_index": step_index,
                        "action": action,
                        "action_meaning": _action_meaning(
                            action,
                            config=episode.config,
                        ),
                        "reward": transition.step.reward,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": trace_metrics(transition.step.metrics),
                        "event": _event_transition(transition),
                        "decision_observation": _observation_reference(
                            episode,
                            observation=decision_observation,
                            kind="decision",
                            trace_index=trace_index,
                        ),
                        "result_observation": _observation_reference(
                            episode,
                            observation=transition.step.observation,
                            kind="result",
                            trace_index=trace_index,
                        ),
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(action: PolicyValue, *, config: HighwayConfig) -> PolicyValue:
    if not config.continuous:
        if type(action) is not int or not 0 <= action < config.action_size:
            raise ValueError("HighwayEnv trace Action is invalid")
        return action
    if type(action) is not list or len(action) != config.action_size:
        raise ValueError("HighwayEnv trace Action is invalid")
    traced: list[PolicyValue] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("HighwayEnv trace Action is invalid")
        traced.append(value)
    return traced


def _action_meaning(action: PolicyValue, *, config: HighwayConfig) -> str:
    if not config.continuous:
        if not isinstance(action, int):
            raise ValueError("HighwayEnv trace Action meaning is invalid")
        return _discrete_action_meanings(config)[action]
    if not isinstance(action, list):
        raise ValueError("HighwayEnv trace Action meaning is invalid")
    parts: list[str] = []
    for name, value in zip(
        _continuous_action_meanings(config),
        action,
        strict=True,
    ):
        if not isinstance(value, float):
            raise ValueError("HighwayEnv trace Action meaning is invalid")
        parts.append(f"{name}={value:g}")
    return ",".join(parts)


def _observation_artifact(
    episode: _TracedEpisode,
) -> tuple[Artifact, list[PolicyValue]]:
    record = episode.record
    initial = _observation_fields(
        record.initial_observation,
        config=episode.config,
    )
    decisions = tuple(
        initial
        if step_index == 0
        else _observation_fields(
            record.transitions[step_index - 1].step.observation,
            config=episode.config,
        )
        for step_index in episode.step_indices
    )
    results = tuple(
        _observation_fields(
            record.transitions[step_index].step.observation,
            config=episode.config,
        )
        for step_index in episode.step_indices
    )
    arrays: dict[str, NDArray[numpy.generic]] = {
        "step_indices": numpy.asarray(
            episode.step_indices,
            dtype=numpy.int32,
        )
    }
    manifests: list[PolicyValue] = []
    for field, initial_array in initial.items():
        initial_name = f"initial__{field}"
        decision_name = f"decision__{field}"
        result_name = f"result__{field}"
        arrays[initial_name] = initial_array
        arrays[decision_name] = _field_array(
            decisions,
            field=field,
            template=initial_array,
        )
        arrays[result_name] = _field_array(
            results,
            field=field,
            template=initial_array,
        )
        manifests.append(
            {
                "field": field,
                "dtype": initial_array.dtype.name,
                "shape": list(initial_array.shape),
                "initial_array": initial_name,
                "decision_array": decision_name,
                "result_array": result_name,
            }
        )
    buffer = io.BytesIO()
    numpy.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return (
        Artifact(
            name=episode.observation_artifact_name,
            media_type="application/x-npz",
            content=buffer.getvalue(),
        ),
        manifests,
    )


def _field_array(
    observations: Sequence[dict[str, NDArray[numpy.generic]]],
    *,
    field: str,
    template: NDArray[numpy.generic],
) -> NDArray[numpy.generic]:
    arrays: list[NDArray[numpy.generic]] = []
    expected_fields = set(observations[0]) if observations else set()
    for observation in observations:
        if set(observation) != expected_fields:
            raise ValueError("HighwayEnv trace observation fields changed")
        array = observation.get(field)
        if (
            array is None
            or array.dtype != template.dtype
            or array.shape != template.shape
        ):
            raise ValueError("HighwayEnv trace observation field changed")
        arrays.append(array)
    if not arrays:
        return numpy.empty((0, *template.shape), dtype=template.dtype)
    return numpy.stack(arrays)


def _observation_reference(
    episode: _TracedEpisode,
    *,
    observation: PolicyValue,
    kind: str,
    trace_index: int | None,
) -> dict[str, object]:
    fields = _observation_fields(observation, config=episode.config)
    references: list[dict[str, object]] = []
    for field in fields:
        reference: dict[str, object] = {
            "field": field,
            "array": f"{kind}__{field}",
        }
        if trace_index is not None:
            reference["index"] = trace_index
        references.append(reference)
    return {
        "artifact": episode.observation_artifact_name,
        "fields": references,
        "semantics": _observation_semantics(
            fields,
            config=episode.config,
        ),
    }


def _observation_fields(
    value: PolicyValue,
    *,
    config: HighwayConfig,
) -> dict[str, NDArray[numpy.generic]]:
    if config.profile in _KINEMATICS_PROFILES:
        shape, _ = _KINEMATICS_PROFILES[config.profile]
        return {
            "root": _trace_tensor(
                value,
                dtype="float32",
                shape=shape,
                field="root",
            )
        }
    if config.profile in _TTC_PROFILES:
        return {
            "root": _trace_tensor(
                value,
                dtype="float32",
                shape=_TTC_PROFILES[config.profile],
                field="root",
            )
        }
    if config.profile == "parking":
        expected = {"observation", "achieved_goal", "desired_goal"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("HighwayEnv parking trace observation is invalid")
        return {
            field: _trace_tensor(
                value[field],
                dtype="float64",
                shape=(6,),
                field=field,
            )
            for field in sorted(expected)
        }
    if config.profile == "racetrack":
        return {
            "root": _trace_tensor(
                value,
                dtype="float32",
                shape=(2, 12, 12),
                field="root",
            )
        }
    if config.profile == "lane-keeping":
        expected = {"state", "derivative", "reference_state"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError(
                "HighwayEnv lane-keeping trace observation is invalid"
            )
        return {
            field: _trace_tensor(
                value[field],
                dtype="float64",
                shape=(4, 1),
                field=field,
            )
            for field in sorted(expected)
        }
    raise ValueError("HighwayEnv trace observation profile is invalid")


def _trace_tensor(
    value: PolicyValue,
    *,
    dtype: str,
    shape: tuple[int, ...],
    field: str,
) -> NDArray[numpy.generic]:
    expected_bytes = math.prod(shape) * numpy.dtype(dtype).itemsize
    if (
        type(value) is not TensorValue
        or value.dtype != dtype
        or value.shape != shape
        or len(value.data) != expected_bytes
    ):
        raise ValueError(
            f"HighwayEnv trace observation field {field!r} is invalid"
        )
    array = numpy.frombuffer(value.data, dtype=numpy.dtype(dtype)).reshape(shape)
    if numpy.issubdtype(array.dtype, numpy.floating) and not numpy.isfinite(
        array
    ).all():
        raise ValueError(
            f"HighwayEnv trace observation field {field!r} is non-finite"
        )
    return array


def _observation_semantics(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    config: HighwayConfig,
) -> dict[str, object]:
    if config.profile in _KINEMATICS_PROFILES:
        return _kinematics_semantics(fields["root"], config=config)
    if config.profile in _TTC_PROFILES:
        return _ttc_semantics(fields["root"])
    if config.profile == "parking":
        return _parking_semantics(fields)
    if config.profile == "racetrack":
        return _occupancy_semantics(fields["root"])
    if config.profile == "lane-keeping":
        return _lane_keeping_semantics(fields)
    raise ValueError("HighwayEnv semantic profile is invalid")


def _kinematics_semantics(
    array: NDArray[numpy.generic],
    *,
    config: HighwayConfig,
) -> dict[str, object]:
    _, features = _KINEMATICS_PROFILES[config.profile]
    values = array.astype(numpy.float64, copy=False)
    present = tuple(
        row_index
        for row_index in range(values.shape[0])
        if values[row_index, 0] > 0.5
    )

    def vehicle(row_index: int) -> dict[str, float]:
        return {
            feature: float(values[row_index, column])
            for column, feature in enumerate(features)
        }

    return {
        "kind": "kinematics",
        "features": list(features),
        "ego": vehicle(0),
        "observed_vehicle_count": max(0, len(present) - 1),
        "observed_vehicles": [
            vehicle(row_index) for row_index in present[1:5]
        ],
        "observed_vehicles_omitted": max(0, len(present) - 5),
    }


def _ttc_semantics(array: NDArray[numpy.generic]) -> dict[str, object]:
    values = array.astype(numpy.float64, copy=False)
    risky = numpy.argwhere(values > 0.0)
    centre = values[1, 1]
    centre_risky = numpy.flatnonzero(centre > 0.0)
    return {
        "kind": "time_to_collision",
        "maximum_collision_cost": float(values.max(initial=0.0)),
        "risky_cells": int(risky.shape[0]),
        "earliest_risk_time_index": (
            None if not risky.size else int(risky[:, 2].min())
        ),
        "current_speed_lane_earliest_risk_time_index": (
            None if not centre_risky.size else int(centre_risky[0])
        ),
    }


def _parking_semantics(
    fields: dict[str, NDArray[numpy.generic]],
) -> dict[str, object]:
    achieved = fields["achieved_goal"].astype(numpy.float64, copy=False)
    desired = fields["desired_goal"].astype(numpy.float64, copy=False)
    error = desired - achieved
    return {
        "kind": "goal_kinematics",
        "features": list(_PARKING_FEATURES),
        "achieved_goal": _named_vector(achieved, _PARKING_FEATURES),
        "desired_goal": _named_vector(desired, _PARKING_FEATURES),
        "goal_error": _named_vector(error, _PARKING_FEATURES),
        "position_error": float(numpy.linalg.norm(error[:2])),
        "velocity_error": float(numpy.linalg.norm(error[2:4])),
        "heading_vector_error": float(numpy.linalg.norm(error[4:6])),
    }


def _occupancy_semantics(array: NDArray[numpy.generic]) -> dict[str, object]:
    values = array.astype(numpy.float64, copy=False)
    occupied = numpy.argwhere(values[0] > 0.0)
    return {
        "kind": "occupancy_grid",
        "features": list(_OCCUPANCY_FEATURES),
        "occupied_cells": int(occupied.shape[0]),
        "occupied_cell_indices": [
            [int(index[0]), int(index[1])] for index in occupied[:16]
        ],
        "occupied_cells_omitted": max(0, int(occupied.shape[0]) - 16),
        "on_road_cells": int(numpy.count_nonzero(values[1] > 0.0)),
    }


def _lane_keeping_semantics(
    fields: dict[str, NDArray[numpy.generic]],
) -> dict[str, object]:
    state = fields["state"].astype(numpy.float64, copy=False).reshape(4)
    derivative = fields["derivative"].astype(
        numpy.float64,
        copy=False,
    ).reshape(4)
    reference = fields["reference_state"].astype(
        numpy.float64,
        copy=False,
    ).reshape(4)
    return {
        "kind": "vehicle_attributes",
        "features": list(_LANE_STATE_FEATURES),
        "state": _named_vector(state, _LANE_STATE_FEATURES),
        "derivative": _named_vector(derivative, _LANE_STATE_FEATURES),
        "reference_state": _named_vector(reference, _LANE_STATE_FEATURES),
        "tracking_error": _named_vector(
            reference - state,
            _LANE_STATE_FEATURES,
        ),
    }


def _named_vector(
    values: NDArray[numpy.generic],
    names: Sequence[str],
) -> dict[str, float]:
    flat = values.reshape(-1)
    if flat.size != len(names):
        raise ValueError("HighwayEnv semantic vector has invalid size")
    return {
        name: float(flat[index]) for index, name in enumerate(names)
    }


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


__all__ = ["HighwayBenchmark"]
