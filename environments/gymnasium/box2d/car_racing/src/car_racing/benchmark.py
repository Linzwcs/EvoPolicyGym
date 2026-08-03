"""A parameterized CarRacing-v3 Benchmark with lossless pixel traces."""

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
from PIL import Image, ImageDraw

from .config import CarRacingConfig
from .environment import CarRacingEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-car-racing/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_STEPS_PER_EPISODE = 48
_MAX_PROGRESS_EVENT_STEPS = 16
_TRACE_EDGE_STEPS = 12
_MAX_CONTACT_SHEET_FRAMES = 12
_CONTACT_SHEET_COLUMNS = 4
_MAX_REPLAY_FRAMES = 24
_MAX_REPLAY_PROGRESS_EVENTS = 12
_MAX_REPLAY_ARTIFACT_BYTES = 3 * 1024 * 1024
_REPLAY_FRAME_DURATION_MS = 160
_REPLAY_SCALE = 3
_REPLAY_STATUS_HEIGHT = 28
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_FRAME_SHAPE = (96, 96, 3)
_FRAME_BYTES = 96 * 96 * 3
_THUMBNAIL_SHAPE = _FRAME_SHAPE
_FRAME_PENALTY = 0.1
_TRACK_REWARD_TOTAL = 1_000.0
_DISCRETE_ACTIONS = {
    0: "do_nothing",
    1: "steer_right",
    2: "steer_left",
    3: "gas",
    4: "brake",
}


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    step_indices: tuple[int, ...]
    continuous: bool

    @property
    def frame_artifact_name(self) -> str:
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
    frame: NDArray[numpy.uint8]
    action: PolicyValue
    action_meaning: str | None
    reward: float | None
    cumulative_return: float
    inferred_track_coverage: float
    progress_event: bool
    off_playfield: bool


class CarRacingBenchmark:
    """Mean CarRacing return over deterministic Episode plans."""

    def __init__(self, config: CarRacingConfig | None = None) -> None:
        if config is None:
            config = CarRacingConfig()
        if type(config) is not CarRacingConfig:
            raise TypeError("config must be CarRacingConfig")
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
        return CarRacingEnvironment(episode, config=self._config)

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
        laps = sum(_completed_lap(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        traced = tuple(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                step_indices=_trace_step_indices(record),
                continuous=self._config.continuous,
            )
            for episode_index, record in enumerate(
                records[:_MAX_TRACED_EPISODES]
            )
        )
        trace_artifact = _trace_artifact(traced)
        visual_artifacts: list[Artifact] = []
        frame_manifests: list[PolicyValue] = []
        raw_frame_bytes = 0
        for episode in traced:
            frame_artifact = _frame_artifact(episode)
            contact_sheet, contact_sheet_tiles = _contact_sheet_artifact(
                episode
            )
            replay, replay_timeline, replay_scale = _replay_artifact(episode)
            visual_artifacts.extend((frame_artifact, contact_sheet, replay))
            episode_raw_frame_bytes = (
                1 + 2 * len(episode.step_indices)
            ) * _FRAME_BYTES
            raw_frame_bytes += episode_raw_frame_bytes
            frame_manifests.append(
                {
                    "episode_index": episode.episode_index,
                    "frame_artifact": episode.frame_artifact_name,
                    "frame_artifact_sha256": hashlib.sha256(
                        frame_artifact.content
                    ).hexdigest(),
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
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{laps} completed laps."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_laps": laps,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        episode_index=episode_index,
                        traced_steps=traced_steps.get(episode_index, 0),
                        continuous=self._config.continuous,
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
                    "the first and last steps, a bounded sample of inferred "
                    "new-tile events, and an even sample of remaining steps."
                ),
                "trace_format": (
                    "trace.jsonl references lossless decision/result RGB "
                    "arrays in per-Episode observations.npz artifacts. "
                    "Contact sheets and replay GIFs are nearest-neighbor "
                    "previews; omitted steps and Episodes are reported "
                    "explicitly."
                ),
                "track_progress_inference": (
                    "Inferred coverage sums positive (reward + 0.1) tile "
                    "bonuses and divides by the public 1000-point track "
                    "reward total."
                ),
                "replay_frame_cap_per_episode": _MAX_REPLAY_FRAMES,
                "replay_frame_duration_ms": _REPLAY_FRAME_DURATION_MS,
                "replay_artifact_byte_cap": _MAX_REPLAY_ARTIFACT_BYTES,
                "frame_artifacts": frame_manifests,
            },
            artifacts=(trace_artifact, *visual_artifacts),
        )


def _benchmark_spec(config: CarRacingConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        action_space = {
            "type": "array",
            "shape": [3],
            "components": [
                {
                    "name": "steering",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                {
                    "name": "gas",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                {
                    "name": "brake",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            ],
            "items": {"type": "float"},
        }
    else:
        action_space = {
            "type": "discrete",
            "values": [0, 1, 2, 3, 4],
            "meaning": {
                "0": "do_nothing",
                "1": "steer_right",
                "2": "steer_left",
                "3": "gas",
                "4": "brake",
            },
        }
    return BenchmarkSpec(
        id="gymnasium/CarRacing-v3/mean-return-v1",
        description=(
            "Drive a rear-wheel-drive car around a procedurally generated "
            "track using exact RGB observations. Maximize tile coverage while "
            "finishing quickly and staying inside the playfield."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [96, 96, 3],
            "layout": "HWC",
            "channels": ["red", "green", "blue"],
            "minimum": 0,
            "maximum": 255,
        },
        action_space=action_space,
        metadata={
            "environment": "CarRacing-v3",
            "provider": "Gymnasium",
            "reward_threshold": 900.0,
            "off_playfield_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "continuous": config.continuous,
            "lap_complete_percent": config.lap_complete_percent,
            "domain_randomize": config.domain_randomize,
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


def _completed_lap(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward > -100.0
    )


def _episode_summary(
    record: EpisodeRecord,
    *,
    episode_index: int,
    traced_steps: int,
    continuous: bool,
) -> PolicyValue:
    action_counts: dict[str, int] = {}
    steering_total = 0.0
    steering_absolute_total = 0.0
    gas_total = 0.0
    brake_total = 0.0
    simultaneous_gas_brake_steps = 0
    positive_progress_events = 0
    negative_reward_events = 0
    off_playfield = False
    for transition in record.transitions:
        action = _trace_action(transition.action, continuous=continuous)
        if isinstance(action, int):
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        else:
            steering, gas, brake = cast(list[float], action)
            steering_total += steering
            steering_absolute_total += abs(steering)
            gas_total += gas
            brake_total += brake
            simultaneous_gas_brake_steps += gas > 0.0 and brake > 0.0
        positive_progress_events += _progress_bonus(
            transition.step.reward
        ) > 0.0
        negative_reward_events += transition.step.reward < -_FRAME_PENALTY
        off_playfield = off_playfield or _off_playfield_transition(transition)
    used_action_counts: dict[str, PolicyValue] = {
        action: count
        for action, count in sorted(
            action_counts.items(), key=lambda item: int(item[0])
        )
    }
    control_summary: dict[str, PolicyValue] = {}
    if continuous and record.steps:
        control_summary = {
            "mean_steering": steering_total / record.steps,
            "mean_absolute_steering": steering_absolute_total / record.steps,
            "mean_gas": gas_total / record.steps,
            "mean_brake": brake_total / record.steps,
            "simultaneous_gas_brake_steps": simultaneous_gas_brake_steps,
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
            else _FAILURE_RETURN
        ),
        "steps": record.steps,
        "completed_lap": _completed_lap(record),
        "off_playfield": off_playfield,
        "terminated": bool(
            record.policy_failure is None
            and record.transitions
            and record.transitions[-1].step.terminated
        ),
        "truncated": bool(
            record.policy_failure is None
            and record.transitions
            and record.transitions[-1].step.truncated
        ),
        "failure": record.policy_failure,
        "action_counts": used_action_counts,
        "control_summary": control_summary,
        "inferred_track_coverage": _inferred_track_coverage(record),
        "inferred_new_tile_events": positive_progress_events,
        "negative_reward_events": negative_reward_events,
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _trace_step_indices(record: EpisodeRecord) -> tuple[int, ...]:
    if record.steps <= _MAX_TRACED_STEPS_PER_EPISODE:
        return tuple(range(record.steps))

    selected = set(range(_TRACE_EDGE_STEPS))
    selected.update(range(record.steps - _TRACE_EDGE_STEPS, record.steps))
    progress_steps = tuple(
        step_index
        for step_index, transition in enumerate(record.transitions)
        if _progress_bonus(transition.step.reward) > 0.0
        and step_index not in selected
    )
    selected.update(
        _even_sample(progress_steps, _MAX_PROGRESS_EVENT_STEPS)
    )
    remaining_capacity = _MAX_TRACED_STEPS_PER_EPISODE - len(selected)
    remaining_steps = tuple(
        step_index
        for step_index in range(record.steps)
        if step_index not in selected
    )
    selected.update(_even_sample(remaining_steps, remaining_capacity))
    return tuple(sorted(selected))


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


def _trace_artifact(episodes: Sequence[_TracedEpisode]) -> Artifact:
    lines: list[bytes] = []
    for episode in episodes:
        record = episode.record
        cumulative_metrics = _cumulative_metrics(record)
        lines.append(
            _json_line(
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
                        else _FAILURE_RETURN
                    ),
                    "completed_lap": _completed_lap(record),
                    "off_playfield": any(
                        _off_playfield_transition(transition)
                        for transition in record.transitions
                    ),
                    "failure": record.policy_failure,
                    "traced_steps": len(episode.step_indices),
                    "omitted_steps": (
                        record.steps - len(episode.step_indices)
                    ),
                    "initial_observation": {
                        "artifact": episode.frame_artifact_name,
                        "array": "initial_frame",
                    },
                }
            )
        )
        for trace_index, step_index in enumerate(episode.step_indices):
            transition = record.transitions[step_index]
            action = _trace_action(
                transition.action,
                continuous=episode.continuous,
            )
            cumulative_return, inferred_track_coverage = cumulative_metrics[
                step_index
            ]
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode.episode_index,
                        "step_index": step_index,
                        "action": action,
                        "action_meaning": _action_meaning(
                            action,
                            continuous=episode.continuous,
                        ),
                        "reward": transition.step.reward,
                        "cumulative_return": cumulative_return,
                        "inferred_new_tile_bonus": _progress_bonus(
                            transition.step.reward
                        ),
                        "inferred_track_coverage": inferred_track_coverage,
                        "off_playfield": _off_playfield_transition(
                            transition
                        ),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "decision_observation": {
                            "artifact": episode.frame_artifact_name,
                            "array": "decision_frames",
                            "index": trace_index,
                        },
                        "result_observation": {
                            "artifact": episode.frame_artifact_name,
                            "array": "result_frames",
                            "index": trace_index,
                        },
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _frame_artifact(episode: _TracedEpisode) -> Artifact:
    record = episode.record
    initial_frame = _trace_frame(record.initial_observation)
    decision_frames = tuple(
        initial_frame
        if step_index == 0
        else _trace_frame(
            record.transitions[step_index - 1].step.observation
        )
        for step_index in episode.step_indices
    )
    result_frames = tuple(
        _trace_frame(record.transitions[step_index].step.observation)
        for step_index in episode.step_indices
    )
    buffer = io.BytesIO()
    numpy.savez_compressed(
        buffer,
        initial_frame=initial_frame,
        step_indices=numpy.asarray(episode.step_indices, dtype=numpy.int32),
        decision_frames=_frame_array(decision_frames),
        result_frames=_frame_array(result_frames),
    )
    return Artifact(
        name=episode.frame_artifact_name,
        media_type="application/x-npz",
        content=buffer.getvalue(),
    )


def _contact_sheet_artifact(
    episode: _TracedEpisode,
) -> tuple[Artifact, list[PolicyValue]]:
    candidates = (
        (-1, _trace_frame(episode.record.initial_observation)),
        *(
            (
                step_index,
                _trace_frame(
                    episode.record.transitions[step_index].step.observation
                ),
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
        ] = frame
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
    cumulative_metrics = _cumulative_metrics(record)
    candidates = (
        _ReplayFrame(
            step_index=None,
            frame=_trace_frame(record.initial_observation),
            action=None,
            action_meaning=None,
            reward=None,
            cumulative_return=0.0,
            inferred_track_coverage=0.0,
            progress_event=False,
            off_playfield=False,
        ),
        *(
            _replay_frame(
                episode,
                step_index=step_index,
                cumulative_metrics=cumulative_metrics,
            )
            for step_index in episode.step_indices
        ),
    )
    required_indices = {0, len(candidates) - 1}
    off_playfield_indices = tuple(
        index
        for index, frame in enumerate(candidates)
        if frame.off_playfield and index not in required_indices
    )
    required_indices.update(off_playfield_indices)
    progress_indices = tuple(
        index
        for index, frame in enumerate(candidates)
        if frame.progress_event and index not in required_indices
    )
    required_indices.update(
        _even_sample(
            progress_indices,
            min(
                _MAX_REPLAY_PROGRESS_EVENTS,
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
            "cumulative_return": frame.cumulative_return,
            "inferred_track_coverage": frame.inferred_track_coverage,
            "progress_event": frame.progress_event,
            "off_playfield": frame.off_playfield,
        }
        for frame_index, frame in enumerate(selected)
    ]
    content, scale = _encode_replay_gif(selected)
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
    cumulative_metrics: Sequence[tuple[float, float]],
) -> _ReplayFrame:
    transition = episode.record.transitions[step_index]
    action = _trace_action(
        transition.action,
        continuous=episode.continuous,
    )
    cumulative_return, inferred_track_coverage = cumulative_metrics[step_index]
    return _ReplayFrame(
        step_index=step_index,
        frame=_trace_frame(transition.step.observation),
        action=action,
        action_meaning=_action_meaning(
            action,
            continuous=episode.continuous,
        ),
        reward=transition.step.reward,
        cumulative_return=cumulative_return,
        inferred_track_coverage=inferred_track_coverage,
        progress_event=_progress_bonus(transition.step.reward) > 0.0,
        off_playfield=_off_playfield_transition(transition),
    )


def _encode_replay_gif(
    frames: Sequence[_ReplayFrame],
) -> tuple[bytes, int]:
    content = _replay_gif_bytes(frames, scale=_REPLAY_SCALE)
    if len(content) <= _MAX_REPLAY_ARTIFACT_BYTES:
        return content, _REPLAY_SCALE
    content = _replay_gif_bytes(frames, scale=1)
    if len(content) > _MAX_REPLAY_ARTIFACT_BYTES:
        raise ValueError(
            "CarRacing replay GIF exceeds its bounded artifact limit"
        )
    return content, 1


def _replay_gif_bytes(
    frames: Sequence[_ReplayFrame],
    *,
    scale: int,
) -> bytes:
    if not frames:
        raise ValueError("CarRacing replay GIF requires at least one frame")
    rendered = [_render_replay_frame(frame, scale=scale) for frame in frames]
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
    scale: int,
) -> Image.Image:
    game_width = _FRAME_SHAPE[1] * scale
    game_height = _FRAME_SHAPE[0] * scale
    image = Image.fromarray(frame.frame, mode="RGB").resize(
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
    if frame.step_index is not None and frame.reward is not None:
        label = (
            f"step {frame.step_index} {_short_action(frame.action)} "
            f"r={frame.reward:g} total={frame.cumulative_return:g} "
            f"track={100.0 * frame.inferred_track_coverage:.1f}%"
        )
    ImageDraw.Draw(rendered).text(
        (4, game_height + 6),
        label,
        fill=(255, 255, 255),
    )
    return rendered


def _trace_action(
    action: PolicyValue,
    *,
    continuous: bool,
) -> PolicyValue:
    if not continuous:
        if type(action) is not int or action not in {0, 1, 2, 3, 4}:
            raise ValueError("CarRacing trace Action is invalid")
        return action
    if type(action) is not list or len(action) != 3:
        raise ValueError("CarRacing trace Action is invalid")
    bounds = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    traced: list[PolicyValue] = []
    for value, (minimum, maximum) in zip(action, bounds, strict=True):
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError("CarRacing trace Action is invalid")
        traced.append(value)
    return traced


def _action_meaning(action: PolicyValue, *, continuous: bool) -> str:
    if not continuous:
        if not isinstance(action, int):
            raise ValueError("CarRacing trace Action meaning is invalid")
        return _DISCRETE_ACTIONS[action]
    if not isinstance(action, list) or len(action) != 3:
        raise ValueError("CarRacing trace Action meaning is invalid")
    if not all(isinstance(value, float) for value in action):
        raise ValueError("CarRacing trace Action meaning is invalid")
    steering, gas, brake = cast(list[float], action)
    return f"steering={steering:g},gas={gas:g},brake={brake:g}"


def _short_action(action: PolicyValue) -> str:
    if action is None:
        return "initial"
    if isinstance(action, int):
        return _DISCRETE_ACTIONS[action]
    traced = _trace_action(action, continuous=True)
    assert isinstance(traced, list)
    values = cast(list[float], traced)
    return f"s={values[0]:g} g={values[1]:g} b={values[2]:g}"


def _progress_bonus(reward: float) -> float:
    return max(0.0, reward + _FRAME_PENALTY)


def _inferred_track_coverage(record: EpisodeRecord) -> float:
    tile_reward = sum(
        _progress_bonus(transition.step.reward)
        for transition in record.transitions
    )
    return min(1.0, tile_reward / _TRACK_REWARD_TOTAL)


def _cumulative_metrics(
    record: EpisodeRecord,
) -> tuple[tuple[float, float], ...]:
    cumulative_return = 0.0
    tile_reward = 0.0
    metrics: list[tuple[float, float]] = []
    for transition in record.transitions:
        cumulative_return += transition.step.reward
        tile_reward += _progress_bonus(transition.step.reward)
        metrics.append(
            (
                cumulative_return,
                min(1.0, tile_reward / _TRACK_REWARD_TOTAL),
            )
        )
    return tuple(metrics)


def _off_playfield_transition(transition: Transition) -> bool:
    return bool(
        transition.step.terminated
        and transition.step.reward <= -100.0
    )


def _trace_frame(observation: PolicyValue) -> NDArray[numpy.uint8]:
    if (
        type(observation) is not TensorValue
        or observation.dtype != "uint8"
        or observation.shape != _FRAME_SHAPE
        or len(observation.data) != _FRAME_BYTES
    ):
        raise ValueError("CarRacing trace observation is invalid")
    return numpy.frombuffer(observation.data, dtype=numpy.uint8).reshape(
        _FRAME_SHAPE
    )


def _frame_array(
    frames: Sequence[NDArray[numpy.uint8]],
) -> NDArray[numpy.uint8]:
    if not frames:
        return numpy.empty((0, *_FRAME_SHAPE), dtype=numpy.uint8)
    return numpy.stack(frames)


def _png_rgb(image: NDArray[numpy.uint8]) -> bytes:
    if (
        image.dtype != numpy.dtype("uint8")
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("CarRacing contact sheet image is invalid")
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


__all__ = ["CarRacingBenchmark"]
