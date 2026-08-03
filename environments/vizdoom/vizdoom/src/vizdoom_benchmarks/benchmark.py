"""Bundled ViZDoom scenarios with bounded public traces."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy
from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .config import ViZDoomConfig
from .environment import ViZDoomEnvironment

_SEED_DOMAIN = b"evopolicygym-vizdoom/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_STEPS_PER_EPISODE = 32
_MAX_REWARD_EVENT_STEPS = 8
_MAX_STATE_EVENT_STEPS = 8
_TRACE_EDGE_STEPS = 8
_MAX_CONTACT_SHEET_FRAMES = 12
_CONTACT_SHEET_COLUMNS = 4
_MAX_REPLAY_FRAMES = 24
_MAX_REPLAY_ARTIFACT_BYTES = 3 * 1024 * 1024
_REPLAY_FRAME_DURATION_MS = 160
_REPLAY_SCALE = 2
_REPLAY_STATUS_HEIGHT = 28
_FRAME_SHAPE = (240, 320, 3)
_FRAME_BYTES = 240 * 320 * 3
_THUMBNAIL_SHAPE = (120, 160, 3)
_AUDIO_SHAPE = (1260, 2)


@dataclass(frozen=True, slots=True)
class _TraceObservation:
    screen: NDArray[numpy.uint8]
    audio: NDArray[numpy.int16] | None
    notifications: str | None
    game_variables: NDArray[numpy.float32] | None


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    step_indices: tuple[int, ...]
    config: ViZDoomConfig

    @property
    def observation_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/observations.npz"

    @property
    def contact_sheet_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/contact-sheet.png"

    @property
    def replay_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/replay.gif"


@dataclass(frozen=True, slots=True)
class _ReplayFrame:
    step_index: int | None
    observation: _TraceObservation
    action: PolicyValue
    action_meaning: str | None
    reward: float | None
    semantic_event: bool


class ViZDoomBenchmark:
    """Mean return for one fixed bundled ViZDoom scenario."""

    def __init__(self, config: ViZDoomConfig | None = None) -> None:
        if config is None:
            config = ViZDoomConfig()
        if type(config) is not ViZDoomConfig:
            raise TypeError("config must be ViZDoomConfig")
        self._config = config
        self._spec = _spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self, split: str, *, seed: int, count: int
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
        return ViZDoomEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        floor = -float(self._config.max_episode_steps)
        returns = tuple(
            r.total_reward if r.policy_failure is None else floor
            for r in records
        )
        score = statistics.fmean(returns)
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        traced = tuple(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                step_indices=_trace_step_indices(record, config=self._config),
                config=self._config,
            )
            for episode_index, record in enumerate(
                records[:_MAX_TRACED_EPISODES]
            )
        )
        trace_artifact = _trace_artifact(traced, failure_return=floor)
        visual_artifacts: list[Artifact] = []
        observation_manifests: list[PolicyValue] = []
        raw_frame_bytes = 0
        for episode in traced:
            observation_artifact = _observation_artifact(episode)
            contact_sheet, contact_sheet_tiles = _contact_sheet_artifact(
                episode
            )
            replay, replay_timeline, replay_scale = _replay_artifact(episode)
            visual_artifacts.extend(
                (observation_artifact, contact_sheet, replay)
            )
            stored_observations = 1 + 2 * len(episode.step_indices)
            episode_raw_frame_bytes = stored_observations * _FRAME_BYTES
            raw_frame_bytes += episode_raw_frame_bytes
            observation_manifests.append(
                {
                    "episode_index": episode.episode_index,
                    "observation_artifact": (
                        episode.observation_artifact_name
                    ),
                    "observation_artifact_sha256": hashlib.sha256(
                        observation_artifact.content
                    ).hexdigest(),
                    "stored_channels": _stored_channels(self._config),
                    "contact_sheet_artifact": (
                        episode.contact_sheet_artifact_name
                    ),
                    "contact_sheet_tiles": contact_sheet_tiles,
                    "replay_artifact": episode.replay_artifact_name,
                    "replay_frames": len(replay_timeline),
                    "replay_frames_omitted": (
                        1 + len(episode.step_indices) - len(replay_timeline)
                    ),
                    "replay_scale": replay_scale,
                    "replay_timeline": replay_timeline,
                    "stored_transition_pairs": len(episode.step_indices),
                    "step_indices": list(episode.step_indices),
                    "omitted_steps": (
                        episode.record.steps - len(episode.step_indices)
                    ),
                    "raw_rgb_bytes": episode_raw_frame_bytes,
                }
            )
        traced_steps = {
            episode.episode_index: len(episode.step_indices)
            for episode in traced
        }
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
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_return": floor,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        episode_index=episode_index,
                        failure_return=floor,
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
                "trace_raw_rgb_bytes": raw_frame_bytes,
                "trace_raw_rgb_bytes_cap_per_episode": (
                    (1 + 2 * _MAX_TRACED_STEPS_PER_EPISODE) * _FRAME_BYTES
                ),
                "trace_raw_rgb_bytes_cap_total": (
                    _MAX_TRACED_EPISODES
                    * (1 + 2 * _MAX_TRACED_STEPS_PER_EPISODE)
                    * _FRAME_BYTES
                ),
                "trace_selection": (
                    "Every short Episode is complete. Long Episodes retain "
                    "the first and last steps, bounded samples of non-zero "
                    "reward and game-variable-change events, and an even "
                    "sample of remaining steps."
                ),
                "trace_format": (
                    "trace.jsonl references lossless selected screen, audio, "
                    "and game-variable arrays in per-Episode observations.npz "
                    "artifacts. Notifications remain inline. Contact sheets "
                    "and replay GIFs are nearest-neighbor previews; omitted "
                    "steps and Episodes are reported explicitly."
                ),
                "replay_frame_cap_per_episode": _MAX_REPLAY_FRAMES,
                "replay_frame_duration_ms": _REPLAY_FRAME_DURATION_MS,
                "replay_artifact_byte_cap": _MAX_REPLAY_ARTIFACT_BYTES,
                "observation_artifacts": observation_manifests,
            },
            artifacts=(trace_artifact, *visual_artifacts),
        )


def _spec(config: ViZDoomConfig) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {
        "screen": {
            "type": "tensor",
            "dtype": "uint8",
            "shape": [240, 320, 3],
        }
    }
    if config.audio:
        fields["audio"] = {
            "type": "tensor",
            "dtype": "int16",
            "shape": [1260, 2],
        }
    if config.notifications:
        fields["notifications"] = {"type": "string"}
    if config.game_variables:
        fields["gamevariables"] = {
            "type": "tensor",
            "dtype": "float32",
            "shape": [config.game_variables],
            "names": list(config.game_variable_names),
        }
    action: PolicyValue
    if config.hybrid_action:
        action = {
            "type": "object",
            "fields": {
                "binary": {
                    "type": "discrete",
                    "values": list(range(config.action_size)),
                    "meaning": {
                        str(action): meaning
                        for action, meaning in enumerate(
                            config.action_meanings
                        )
                    },
                },
                "continuous": {
                    "type": "array",
                    "shape": [3],
                    "items": {"type": "finite_float32"},
                    "meaning": list(config.continuous_controls),
                },
            },
        }
    else:
        action = {
            "type": "discrete",
            "values": list(range(config.action_size)),
            "meaning": {
                str(action): meaning
                for action, meaning in enumerate(config.action_meanings)
            },
        }
    return BenchmarkSpec(
        id=f"vizdoom/{config.environment_id}/mean-return-v1",
        description=(
            f"Control the agent in ViZDoom's {config.profile} scenario. "
            "Maximize mean Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space=action,
        metadata={
            "environment": config.environment_id,
            "provider": "ViZDoom",
            "upstream_version": "1.3.0",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters={
            "profile": config.profile,
            "action_size": config.action_size,
            "hybrid_action": config.hybrid_action,
            "action_meanings": list(config.action_meanings),
            "continuous_controls": list(config.continuous_controls),
            "game_variable_names": list(config.game_variable_names),
            "rgb_resolution": [320, 240],
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
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


def _episode_summary(
    record: EpisodeRecord,
    *,
    episode_index: int,
    failure_return: float,
    traced_steps: int,
    config: ViZDoomConfig,
) -> PolicyValue:
    action_counts: dict[str, int] = {}
    positive_reward_events = 0
    negative_reward_events = 0
    continuous_totals = [0.0] * len(config.continuous_controls)
    variable_samples: list[NDArray[numpy.float32]] = []
    notification_events = 0
    initial = _trace_observation(record.initial_observation, config=config)
    if initial.game_variables is not None:
        variable_samples.append(initial.game_variables)
    if initial.notifications:
        notification_events += 1
    for transition in record.transitions:
        action = _trace_action(transition.action, config=config)
        binary = action["binary"] if isinstance(action, dict) else action
        assert isinstance(binary, int)
        action_key = str(binary)
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        if isinstance(action, dict):
            continuous = action["continuous"]
            assert isinstance(continuous, list)
            for index, value in enumerate(continuous):
                assert isinstance(value, float)
                continuous_totals[index] += abs(value)
        if transition.step.reward > 0.0:
            positive_reward_events += 1
        elif transition.step.reward < 0.0:
            negative_reward_events += 1
        observation = _trace_observation(
            transition.step.observation,
            config=config,
        )
        if observation.game_variables is not None:
            variable_samples.append(observation.game_variables)
        if observation.notifications:
            notification_events += 1
    used_action_counts: dict[str, PolicyValue] = {
        action: count
        for action, count in sorted(
            action_counts.items(), key=lambda item: int(item[0])
        )
    }
    continuous_mean_abs: dict[str, PolicyValue] = {}
    if record.steps:
        continuous_mean_abs = {
            name: total / record.steps
            for name, total in zip(
                config.continuous_controls,
                continuous_totals,
                strict=True,
            )
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
        "failure": record.policy_failure,
        "action_counts": used_action_counts,
        "continuous_control_mean_abs": continuous_mean_abs,
        "positive_reward_events": positive_reward_events,
        "negative_reward_events": negative_reward_events,
        "notification_events": notification_events,
        "game_variable_ranges": _game_variable_ranges(
            variable_samples,
            config=config,
        ),
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _game_variable_ranges(
    samples: Sequence[NDArray[numpy.float32]],
    *,
    config: ViZDoomConfig,
) -> dict[str, PolicyValue]:
    if not samples:
        return {}
    values = numpy.stack(samples)
    return {
        name: {
            "initial": float(values[0, index]),
            "final": float(values[-1, index]),
            "minimum": float(values[:, index].min()),
            "maximum": float(values[:, index].max()),
        }
        for index, name in enumerate(config.game_variable_names)
    }


def _trace_step_indices(
    record: EpisodeRecord,
    *,
    config: ViZDoomConfig,
) -> tuple[int, ...]:
    if record.steps <= _MAX_TRACED_STEPS_PER_EPISODE:
        return tuple(range(record.steps))

    selected = set(range(_TRACE_EDGE_STEPS))
    selected.update(range(record.steps - _TRACE_EDGE_STEPS, record.steps))
    reward_steps = tuple(
        step_index
        for step_index, transition in enumerate(record.transitions)
        if transition.step.reward != 0.0 and step_index not in selected
    )
    selected.update(
        _even_sample(
            reward_steps,
            min(
                _MAX_REWARD_EVENT_STEPS,
                _MAX_TRACED_STEPS_PER_EPISODE - len(selected),
            ),
        )
    )
    semantic_steps = tuple(
        step_index
        for step_index in range(record.steps)
        if step_index not in selected
        and _semantic_event(record, step_index=step_index, config=config)
    )
    selected.update(
        _even_sample(
            semantic_steps,
            min(
                _MAX_STATE_EVENT_STEPS,
                _MAX_TRACED_STEPS_PER_EPISODE - len(selected),
            ),
        )
    )
    remaining_capacity = _MAX_TRACED_STEPS_PER_EPISODE - len(selected)
    remaining_steps = tuple(
        step_index
        for step_index in range(record.steps)
        if step_index not in selected
    )
    selected.update(_even_sample(remaining_steps, remaining_capacity))
    return tuple(sorted(selected))


def _semantic_event(
    record: EpisodeRecord,
    *,
    step_index: int,
    config: ViZDoomConfig,
) -> bool:
    decision_value = (
        record.initial_observation
        if step_index == 0
        else record.transitions[step_index - 1].step.observation
    )
    decision = _trace_observation(decision_value, config=config)
    result = _trace_observation(
        record.transitions[step_index].step.observation,
        config=config,
    )
    variables_changed = bool(
        decision.game_variables is not None
        and result.game_variables is not None
        and not numpy.array_equal(
            decision.game_variables,
            result.game_variables,
        )
    )
    return variables_changed or bool(result.notifications)


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


def _trace_artifact(
    episodes: Sequence[_TracedEpisode],
    *,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode in episodes:
        record = episode.record
        initial = _trace_observation(
            record.initial_observation,
            config=episode.config,
        )
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
                        observation=initial,
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
            decision_value = (
                record.initial_observation
                if step_index == 0
                else record.transitions[step_index - 1].step.observation
            )
            decision = _trace_observation(
                decision_value,
                config=episode.config,
            )
            result = _trace_observation(
                transition.step.observation,
                config=episode.config,
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
                        "semantic_event": _semantic_event(
                            record,
                            step_index=step_index,
                            config=episode.config,
                        ),
                        "decision_observation": _observation_reference(
                            episode,
                            observation=decision,
                            kind="decision",
                            trace_index=trace_index,
                        ),
                        "result_observation": _observation_reference(
                            episode,
                            observation=result,
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


def _observation_reference(
    episode: _TracedEpisode,
    *,
    observation: _TraceObservation,
    kind: str,
    trace_index: int | None,
) -> dict[str, object]:
    suffix = "initial" if kind == "initial" else kind
    document: dict[str, object] = {
        "artifact": episode.observation_artifact_name,
        "screen_array": (
            "initial_screen" if kind == "initial" else f"{suffix}_screens"
        ),
        "semantics": _observation_semantics(
            observation,
            config=episode.config,
        ),
    }
    if trace_index is not None:
        document["index"] = trace_index
    if episode.config.audio:
        document["audio_array"] = (
            "initial_audio" if kind == "initial" else f"{suffix}_audio"
        )
    if episode.config.game_variables:
        document["game_variables_array"] = (
            "initial_game_variables"
            if kind == "initial"
            else f"{suffix}_game_variables"
        )
    if observation.notifications is not None:
        document["notifications"] = observation.notifications
    return document


def _observation_semantics(
    observation: _TraceObservation,
    *,
    config: ViZDoomConfig,
) -> dict[str, PolicyValue]:
    document: dict[str, PolicyValue] = {}
    if observation.game_variables is not None:
        game_variables: dict[str, PolicyValue] = {
            name: float(observation.game_variables[index])
            for index, name in enumerate(config.game_variable_names)
        }
        document["game_variables"] = game_variables
    if observation.audio is not None:
        samples = observation.audio.astype(numpy.float64)
        audio: dict[str, PolicyValue] = {
            "peak_absolute": int(numpy.abs(samples).max(initial=0.0)),
            "rms": float(numpy.sqrt(numpy.mean(numpy.square(samples)))),
        }
        document["audio"] = audio
    return document


def _observation_artifact(episode: _TracedEpisode) -> Artifact:
    record = episode.record
    initial = _trace_observation(
        record.initial_observation,
        config=episode.config,
    )
    decisions = tuple(
        initial
        if step_index == 0
        else _trace_observation(
            record.transitions[step_index - 1].step.observation,
            config=episode.config,
        )
        for step_index in episode.step_indices
    )
    results = tuple(
        _trace_observation(
            record.transitions[step_index].step.observation,
            config=episode.config,
        )
        for step_index in episode.step_indices
    )
    arrays: dict[str, object] = {
        "initial_screen": initial.screen,
        "step_indices": numpy.asarray(
            episode.step_indices,
            dtype=numpy.int32,
        ),
        "decision_screens": _screen_array(decisions),
        "result_screens": _screen_array(results),
    }
    if episode.config.audio:
        if initial.audio is None:
            raise ValueError("ViZDoom initial trace audio is missing")
        arrays.update(
            {
                "initial_audio": initial.audio,
                "decision_audio": _audio_array(decisions),
                "result_audio": _audio_array(results),
            }
        )
    if episode.config.game_variables:
        if initial.game_variables is None:
            raise ValueError(
                "ViZDoom initial trace game variables are missing"
            )
        arrays.update(
            {
                "initial_game_variables": initial.game_variables,
                "decision_game_variables": _game_variable_array(decisions),
                "result_game_variables": _game_variable_array(results),
            }
        )
    buffer = io.BytesIO()
    numpy.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return Artifact(
        name=episode.observation_artifact_name,
        media_type="application/x-npz",
        content=buffer.getvalue(),
    )


def _contact_sheet_artifact(
    episode: _TracedEpisode,
) -> tuple[Artifact, list[PolicyValue]]:
    candidates = (
        (
            -1,
            _trace_observation(
                episode.record.initial_observation,
                config=episode.config,
            ).screen,
        ),
        *(
            (
                step_index,
                _trace_observation(
                    episode.record.transitions[step_index].step.observation,
                    config=episode.config,
                ).screen,
            )
            for step_index in episode.step_indices
        ),
    )
    selected_indices = _even_sample(
        tuple(range(len(candidates))),
        _MAX_CONTACT_SHEET_FRAMES,
    )
    selected = tuple(candidates[index] for index in selected_indices)
    rows = max(
        1,
        (len(selected) + _CONTACT_SHEET_COLUMNS - 1)
        // _CONTACT_SHEET_COLUMNS,
    )
    thumbnail_height, thumbnail_width, _ = _THUMBNAIL_SHAPE
    canvas = numpy.zeros(
        (
            rows * thumbnail_height,
            _CONTACT_SHEET_COLUMNS * thumbnail_width,
            3,
        ),
        dtype=numpy.uint8,
    )
    tiles: list[PolicyValue] = []
    for tile_index, (step_index, frame) in enumerate(selected):
        row, column = divmod(tile_index, _CONTACT_SHEET_COLUMNS)
        canvas[
            row * thumbnail_height : (row + 1) * thumbnail_height,
            column * thumbnail_width : (column + 1) * thumbnail_width,
        ] = frame[::2, ::2]
        tiles.append(
            {
                "tile_index": tile_index,
                "kind": "initial" if step_index == -1 else "result",
                "step_index": None if step_index == -1 else step_index,
            }
        )
    return (
        Artifact(
            name=episode.contact_sheet_artifact_name,
            media_type="image/png",
            content=_png_rgb(canvas),
        ),
        tiles,
    )


def _replay_artifact(
    episode: _TracedEpisode,
) -> tuple[Artifact, list[PolicyValue], int]:
    record = episode.record
    candidates = (
        _ReplayFrame(
            step_index=None,
            observation=_trace_observation(
                record.initial_observation,
                config=episode.config,
            ),
            action=None,
            action_meaning=None,
            reward=None,
            semantic_event=False,
        ),
        *(
            _replay_frame(episode, step_index=step_index)
            for step_index in episode.step_indices
        ),
    )
    required_indices = {0, len(candidates) - 1}
    reward_indices = tuple(
        index
        for index, frame in enumerate(candidates)
        if frame.reward is not None
        and frame.reward != 0.0
        and index not in required_indices
    )
    required_indices.update(
        _even_sample(
            reward_indices,
            min(
                _MAX_REWARD_EVENT_STEPS,
                _MAX_REPLAY_FRAMES - len(required_indices),
            ),
        )
    )
    semantic_indices = tuple(
        index
        for index, frame in enumerate(candidates)
        if frame.semantic_event and index not in required_indices
    )
    required_indices.update(
        _even_sample(
            semantic_indices,
            min(
                _MAX_STATE_EVENT_STEPS,
                _MAX_REPLAY_FRAMES - len(required_indices),
            ),
        )
    )
    remaining_capacity = _MAX_REPLAY_FRAMES - len(required_indices)
    remaining_indices = tuple(
        index
        for index in range(len(candidates))
        if index not in required_indices
    )
    required_indices.update(
        _even_sample(remaining_indices, remaining_capacity)
    )
    selected = tuple(candidates[index] for index in sorted(required_indices))
    timeline: list[PolicyValue] = [
        {
            "frame_index": frame_index,
            "kind": "initial" if frame.step_index is None else "result",
            "step_index": frame.step_index,
            "action": frame.action,
            "action_meaning": frame.action_meaning,
            "reward": frame.reward,
            "semantic_event": frame.semantic_event,
            "semantics": _observation_semantics(
                frame.observation,
                config=episode.config,
            ),
            "notifications": frame.observation.notifications,
        }
        for frame_index, frame in enumerate(selected)
    ]
    content, scale = _encode_replay_gif(selected, config=episode.config)
    return (
        Artifact(
            name=episode.replay_artifact_name,
            media_type="image/gif",
            content=content,
        ),
        timeline,
        scale,
    )


def _replay_frame(
    episode: _TracedEpisode,
    *,
    step_index: int,
) -> _ReplayFrame:
    transition = episode.record.transitions[step_index]
    action = _trace_action(transition.action, config=episode.config)
    return _ReplayFrame(
        step_index=step_index,
        observation=_trace_observation(
            transition.step.observation,
            config=episode.config,
        ),
        action=action,
        action_meaning=_action_meaning(action, config=episode.config),
        reward=transition.step.reward,
        semantic_event=_semantic_event(
            episode.record,
            step_index=step_index,
            config=episode.config,
        ),
    )


def _encode_replay_gif(
    frames: Sequence[_ReplayFrame],
    *,
    config: ViZDoomConfig,
) -> tuple[bytes, int]:
    content = _replay_gif_bytes(frames, config=config, scale=_REPLAY_SCALE)
    if len(content) <= _MAX_REPLAY_ARTIFACT_BYTES:
        return content, _REPLAY_SCALE
    content = _replay_gif_bytes(frames, config=config, scale=1)
    if len(content) > _MAX_REPLAY_ARTIFACT_BYTES:
        raise ValueError("ViZDoom replay GIF exceeds its bounded artifact limit")
    return content, 1


def _replay_gif_bytes(
    frames: Sequence[_ReplayFrame],
    *,
    config: ViZDoomConfig,
    scale: int,
) -> bytes:
    if not frames:
        raise ValueError("ViZDoom replay GIF requires at least one frame")
    rendered = [
        _render_replay_frame(frame, config=config, scale=scale)
        for frame in frames
    ]
    stream = io.BytesIO()
    rendered[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=_REPLAY_FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return stream.getvalue()


def _render_replay_frame(
    frame: _ReplayFrame,
    *,
    config: ViZDoomConfig,
    scale: int,
) -> Image.Image:
    game_width = _FRAME_SHAPE[1] * scale
    game_height = _FRAME_SHAPE[0] * scale
    image = Image.fromarray(frame.observation.screen, mode="RGB").resize(
        (game_width, game_height),
        resample=Image.Resampling.NEAREST,
    )
    rendered = Image.new(
        "RGB",
        (game_width, game_height + _REPLAY_STATUS_HEIGHT),
        color=(0, 0, 0),
    )
    rendered.paste(image, (0, 0))
    label = "initial observation"
    if frame.step_index is not None and frame.action_meaning is not None:
        label = (
            f"step {frame.step_index}  action {frame.action_meaning}  "
            f"reward {frame.reward:g}"
        )
    state_label = _game_variable_label(frame.observation, config=config)
    if state_label:
        label += f"  {state_label}"
    ImageDraw.Draw(rendered).text(
        (4, game_height + 6),
        label,
        fill=(255, 255, 255),
    )
    return rendered


def _game_variable_label(
    observation: _TraceObservation,
    *,
    config: ViZDoomConfig,
) -> str:
    if observation.game_variables is None:
        return ""
    return " ".join(
        f"{name.lower()}={float(observation.game_variables[index]):g}"
        for index, name in enumerate(config.game_variable_names[:2])
    )


def _trace_action(
    value: PolicyValue,
    *,
    config: ViZDoomConfig,
) -> PolicyValue:
    if not config.hybrid_action:
        if type(value) is not int or not 0 <= value < config.action_size:
            raise ValueError("ViZDoom trace Action is invalid")
        return value
    if type(value) is not dict or set(value) != {"binary", "continuous"}:
        raise ValueError("ViZDoom trace hybrid Action is invalid")
    binary = value["binary"]
    continuous = value["continuous"]
    if type(binary) is not int or not 0 <= binary < config.action_size:
        raise ValueError("ViZDoom trace binary Action is invalid")
    if type(continuous) is not list or len(continuous) != 3:
        raise ValueError("ViZDoom trace continuous Action is invalid")
    controls: list[PolicyValue] = []
    for item in continuous:
        if type(item) is not float or not math.isfinite(item):
            raise ValueError("ViZDoom trace continuous Action is invalid")
        controls.append(item)
    return {"binary": binary, "continuous": controls}


def _action_meaning(
    action: PolicyValue,
    *,
    config: ViZDoomConfig,
) -> str:
    binary = action["binary"] if isinstance(action, dict) else action
    if not isinstance(binary, int):
        raise ValueError("ViZDoom trace Action meaning is invalid")
    meaning = config.action_meanings[binary]
    if not isinstance(action, dict):
        return meaning
    continuous = action["continuous"]
    if not isinstance(continuous, list):
        raise ValueError("ViZDoom trace Action meaning is invalid")
    controls: list[float] = []
    for value in continuous:
        if not isinstance(value, float):
            raise ValueError("ViZDoom trace Action meaning is invalid")
        controls.append(value)
    deltas = ",".join(
        f"{name.lower()}={value:g}"
        for name, value in zip(
            config.continuous_controls,
            controls,
            strict=True,
        )
    )
    return f"{meaning};{deltas}"


def _trace_observation(
    value: PolicyValue,
    *,
    config: ViZDoomConfig,
) -> _TraceObservation:
    expected = {"screen"}
    if config.audio:
        expected.add("audio")
    if config.notifications:
        expected.add("notifications")
    if config.game_variables:
        expected.add("gamevariables")
    if type(value) is not dict or set(value) != expected:
        raise ValueError("ViZDoom trace observation is invalid")
    screen = _trace_tensor(
        value["screen"],
        dtype="uint8",
        shape=_FRAME_SHAPE,
        name="screen",
    )
    audio: NDArray[numpy.int16] | None = None
    if config.audio:
        audio_value = _trace_tensor(
            value["audio"],
            dtype="int16",
            shape=_AUDIO_SHAPE,
            name="audio",
        )
        audio = audio_value.astype(numpy.int16, copy=False)
    notifications: str | None = None
    if config.notifications:
        notification_value = value["notifications"]
        if type(notification_value) is not str:
            raise ValueError("ViZDoom trace notifications are invalid")
        notifications = notification_value
    game_variables: NDArray[numpy.float32] | None = None
    if config.game_variables:
        game_variable_value = _trace_tensor(
            value["gamevariables"],
            dtype="float32",
            shape=(config.game_variables,),
            name="game variables",
        )
        if not numpy.isfinite(game_variable_value).all():
            raise ValueError("ViZDoom trace game variables are non-finite")
        game_variables = game_variable_value.astype(numpy.float32, copy=False)
    return _TraceObservation(
        screen=screen.astype(numpy.uint8, copy=False),
        audio=audio,
        notifications=notifications,
        game_variables=game_variables,
    )


def _trace_tensor(
    value: PolicyValue,
    *,
    dtype: str,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[numpy.generic]:
    expected_bytes = math.prod(shape) * numpy.dtype(dtype).itemsize
    if (
        type(value) is not TensorValue
        or value.dtype != dtype
        or value.shape != shape
        or len(value.data) != expected_bytes
    ):
        raise ValueError(f"ViZDoom trace {name} is invalid")
    return numpy.frombuffer(value.data, dtype=numpy.dtype(dtype)).reshape(shape)


def _screen_array(
    observations: Sequence[_TraceObservation],
) -> NDArray[numpy.uint8]:
    if not observations:
        return numpy.empty((0, *_FRAME_SHAPE), dtype=numpy.uint8)
    return numpy.stack(tuple(observation.screen for observation in observations))


def _audio_array(
    observations: Sequence[_TraceObservation],
) -> NDArray[numpy.int16]:
    arrays: list[NDArray[numpy.int16]] = []
    for observation in observations:
        if observation.audio is None:
            raise ValueError("ViZDoom trace audio is missing")
        arrays.append(observation.audio)
    if not arrays:
        return numpy.empty((0, *_AUDIO_SHAPE), dtype=numpy.int16)
    return numpy.stack(arrays)


def _game_variable_array(
    observations: Sequence[_TraceObservation],
) -> NDArray[numpy.float32]:
    arrays: list[NDArray[numpy.float32]] = []
    for observation in observations:
        if observation.game_variables is None:
            raise ValueError("ViZDoom trace game variables are missing")
        arrays.append(observation.game_variables)
    if not arrays:
        return numpy.empty((0, 0), dtype=numpy.float32)
    return numpy.stack(arrays)


def _stored_channels(config: ViZDoomConfig) -> list[PolicyValue]:
    channels: list[PolicyValue] = ["screen"]
    if config.audio:
        channels.append("audio")
    if config.game_variables:
        channels.append("gamevariables")
    if config.notifications:
        channels.append("notifications-inline")
    return channels


def _png_rgb(image: NDArray[numpy.uint8]) -> bytes:
    if (
        image.dtype != numpy.dtype("uint8")
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("ViZDoom contact sheet image is invalid")
    height, width, _ = image.shape
    scanlines = b"".join(
        b"\0" + image[row].tobytes(order="C") for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _png_chunk(kind: bytes, content: bytes) -> bytes:
    payload = kind + content
    return (
        struct.pack(">I", len(content))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


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


__all__ = ["ViZDoomBenchmark"]
