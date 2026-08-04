"""Redistributable Stable-Retro Airstriker Benchmark."""

from __future__ import annotations

import hashlib
import io
import json
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

from .config import AirstrikerConfig
from .environment import AirstrikerEnvironment

_SEED_DOMAIN = b"evopolicygym-stable-retro-airstriker/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_EPISODE_STEPS = 18_000
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_STEPS_PER_EPISODE = 32
_MAX_REWARD_EVENT_STEPS = 12
_TRACE_EDGE_STEPS = 8
_MAX_CONTACT_SHEET_FRAMES = 12
_CONTACT_SHEET_COLUMNS = 4
_MAX_REPLAY_FRAMES = 24
_MAX_REPLAY_ARTIFACT_BYTES = 3 * 1024 * 1024
_REPLAY_FRAME_DURATION_MS = 160
_REPLAY_SCALE = 2
_REPLAY_STATUS_HEIGHT = 28
_FRAME_SHAPE = (224, 320, 3)
_FRAME_BYTES = 224 * 320 * 3
_THUMBNAIL_SHAPE = (112, 160, 3)
_CONTROLLER_BUTTONS = (
    "B",
    "A",
    "MODE",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "C",
    "Y",
    "X",
    "Z",
)
_VERTICAL_ACTIONS = ((), ("UP",), ("DOWN",))
_HORIZONTAL_ACTIONS = ((), ("LEFT",), ("RIGHT",))
_BUTTON_ACTIONS = (
    (),
    ("B",),
    ("A",),
    ("B", "A"),
    ("C",),
    ("B", "C"),
    ("Y",),
    ("B", "Y"),
    ("X",),
    ("A", "X"),
    ("Y", "X"),
    ("Z",),
    ("C", "Z"),
    ("Y", "Z"),
)
_ACTION_BUTTONS = tuple(
    tuple(
        button
        for button in _CONTROLLER_BUTTONS
        if button in (*vertical, *horizontal, *face_buttons)
    )
    for face_buttons in _BUTTON_ACTIONS
    for horizontal in _HORIZONTAL_ACTIONS
    for vertical in _VERTICAL_ACTIONS
)
_ACTIONS = tuple(
    "noop" if not buttons else "+".join(button.lower() for button in buttons)
    for buttons in _ACTION_BUTTONS
)


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    step_indices: tuple[int, ...]

    @property
    def frame_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/observations.npz"

    @property
    def contact_sheet_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/contact-sheet.png"

    @property
    def replay_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/replay.gif"


class AirstrikerBenchmark:
    """Mean score delta on Stable-Retro's redistributable Airstriker game."""

    def __init__(self, config: AirstrikerConfig | None = None) -> None:
        if config is None:
            config = AirstrikerConfig()
        if type(config) is not AirstrikerConfig:
            raise TypeError("config must be AirstrikerConfig")
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
        return AirstrikerEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        floor = -float(_MAX_EPISODE_STEPS)
        returns = tuple(
            record.total_reward
            if record.policy_failure is None
            else floor
            for record in records
        )
        score = statistics.fmean(returns)
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        replayed = tuple(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                step_indices=_trace_step_indices(record),
            )
            for episode_index, record in enumerate(records)
        )
        traced = replayed[:_MAX_TRACED_EPISODES]
        trace_artifact = _trace_artifact(traced, failure_score=floor)
        visual_artifacts: list[Artifact] = []
        frame_manifests: list[PolicyValue] = []
        replay_manifests: list[PolicyValue] = []
        raw_frame_bytes = 0
        for episode in replayed:
            replay, replay_timeline, replay_scale = _replay_artifact(episode)
            replay_manifests.append(
                {
                    "episode_index": episode.episode_index,
                    "status": "available",
                    "artifact": episode.replay_artifact_name,
                    "encoded_frames": len(replay_timeline),
                    "frames_omitted": (
                        1 + episode.record.steps - len(replay_timeline)
                    ),
                    "sampled_steps": len(episode.step_indices),
                    "steps_omitted": (
                        episode.record.steps - len(episode.step_indices)
                    ),
                    "scale": replay_scale,
                    "timeline": replay_timeline,
                }
            )
            if episode.episode_index >= len(traced):
                visual_artifacts.append(replay)
                continue
            frame_artifact = _frame_artifact(episode)
            contact_sheet, contact_sheet_tiles = _contact_sheet_artifact(
                episode
            )
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
                    f"Mean Airstriker score delta {score:.3f} across "
                    f"{len(records)} Episodes."
                ),
                "mean_score_delta": score,
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_score": floor,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        episode_index=episode_index,
                        failure_score=floor,
                        traced_steps=traced_steps.get(episode_index, 0),
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
                    "the first and last steps, a bounded sample of non-zero "
                    "reward events, and an even sample of remaining steps."
                ),
                "trace_format": (
                    "trace.jsonl references lossless decision/result RGB "
                    "arrays for the traced Episodes in per-Episode "
                    "observations.npz artifacts. Contact sheets cover those "
                    "traced Episodes. Every Episode has a bounded "
                    "nearest-neighbor replay GIF; omitted trace and replay "
                    "steps are reported explicitly."
                ),
                "replay_frame_cap_per_episode": _MAX_REPLAY_FRAMES,
                "replay_frame_duration_ms": _REPLAY_FRAME_DURATION_MS,
                "replay_artifact_byte_cap": _MAX_REPLAY_ARTIFACT_BYTES,
                "replay_episodes": len(replay_manifests),
                "replay_episodes_without_gif": (
                    len(records) - len(replay_manifests)
                ),
                "replay_artifacts": replay_manifests,
                "frame_artifacts": frame_manifests,
            },
            artifacts=(trace_artifact, *visual_artifacts),
        )


def _spec(config: AirstrikerConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="stable-retro/Airstriker-Genesis-v0/mean-score-delta-v1",
        description=(
            "Play the bundled Airstriker Level 1 from RGB observations using "
            "Stable-Retro's restricted discrete controller actions. Maximize "
            "mean score delta."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [224, 320, 3],
            "color_space": "RGB",
        },
        action_space={
            "type": "discrete",
            "start": 0,
            "count": 126,
            "meaning": {
                str(action): meaning
                for action, meaning in enumerate(_ACTIONS)
            },
            "controller_buttons": list(_CONTROLLER_BUTTONS),
            "restricted_actions": "DISCRETE",
        },
        metadata={
            "environment": config.game,
            "provider": "Stable-Retro",
            "upstream_version": "1.0.1",
            "failure_score": -float(_MAX_EPISODE_STEPS),
        },
        environment_parameters={
            "game": config.game,
            "state": config.state,
            "restricted_actions": "DISCRETE",
            "max_emulator_frames": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_score_delta",
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


def _episode_summary(
    record: EpisodeRecord,
    *,
    episode_index: int,
    failure_score: float,
    traced_steps: int,
) -> PolicyValue:
    action_counts: dict[str, int] = {}
    button_counts = {button: 0 for button in _CONTROLLER_BUTTONS}
    positive_reward_events = 0
    negative_reward_events = 0
    for transition in record.transitions:
        action = _trace_action(transition.action)
        action_key = str(action)
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        for button in _ACTION_BUTTONS[action]:
            button_counts[button] += 1
        if transition.step.reward > 0.0:
            positive_reward_events += 1
        elif transition.step.reward < 0.0:
            negative_reward_events += 1
    used_action_counts: dict[str, PolicyValue] = {
        action: count
        for action, count in sorted(
            action_counts.items(), key=lambda item: int(item[0])
        )
    }
    used_button_counts: dict[str, PolicyValue] = {
        button: count for button, count in button_counts.items() if count > 0
    }
    return {
        "episode_index": episode_index,
        "status": (
            "completed" if record.policy_failure is None else "policy_failed"
        ),
        "score_delta": (
            record.total_reward if record.policy_failure is None else None
        ),
        "scored_score_delta": (
            record.total_reward
            if record.policy_failure is None
            else failure_score
        ),
        "steps": record.steps,
        "failure": record.policy_failure,
        "action_counts": used_action_counts,
        "button_counts": used_button_counts,
        "positive_reward_events": positive_reward_events,
        "negative_reward_events": negative_reward_events,
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _trace_step_indices(record: EpisodeRecord) -> tuple[int, ...]:
    if record.steps <= _MAX_TRACED_STEPS_PER_EPISODE:
        return tuple(range(record.steps))

    selected = set(range(_TRACE_EDGE_STEPS))
    selected.update(range(record.steps - _TRACE_EDGE_STEPS, record.steps))
    reward_steps = tuple(
        step_index
        for step_index, transition in enumerate(record.transitions)
        if transition.step.reward != 0.0 and step_index not in selected
    )
    selected.update(_even_sample(reward_steps, _MAX_REWARD_EVENT_STEPS))
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


def _trace_artifact(
    episodes: Sequence[_TracedEpisode],
    *,
    failure_score: float,
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
                    "score_delta": (
                        record.total_reward
                        if record.policy_failure is None
                        else None
                    ),
                    "scored_score_delta": (
                        record.total_reward
                        if record.policy_failure is None
                        else failure_score
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
            action = _trace_action(transition.action)
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode.episode_index,
                        "step_index": step_index,
                        "action": action,
                        "action_meaning": _ACTIONS[action],
                        "action_buttons": list(_ACTION_BUTTONS[action]),
                        "reward": transition.step.reward,
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
    candidates: tuple[
        tuple[int | None, NDArray[numpy.uint8], int | None, float | None],
        ...,
    ] = (
        (None, _trace_frame(episode.record.initial_observation), None, None),
        *(
            (
                step_index,
                _trace_frame(
                    episode.record.transitions[step_index].step.observation
                ),
                _trace_action(episode.record.transitions[step_index].action),
                episode.record.transitions[step_index].step.reward,
            )
            for step_index in episode.step_indices
        ),
    )
    required_indices = {0, len(candidates) - 1}
    required_indices.update(
        index
        for index, (_, _, _, reward) in enumerate(candidates)
        if reward is not None and reward != 0.0
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
    timeline: list[PolicyValue] = []
    for frame_index, (step_index, _, action, reward) in enumerate(selected):
        timeline.append(
            {
                "frame_index": frame_index,
                "kind": "initial" if step_index is None else "result",
                "step_index": step_index,
                "action": action,
                "action_meaning": (
                    None if action is None else _ACTIONS[action]
                ),
                "action_buttons": (
                    None
                    if action is None
                    else list(_ACTION_BUTTONS[action])
                ),
                "reward": reward,
            }
        )
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


def _encode_replay_gif(
    frames: Sequence[
        tuple[int | None, NDArray[numpy.uint8], int | None, float | None]
    ],
) -> tuple[bytes, int]:
    content = _replay_gif_bytes(frames, scale=_REPLAY_SCALE)
    if len(content) <= _MAX_REPLAY_ARTIFACT_BYTES:
        return content, _REPLAY_SCALE
    content = _replay_gif_bytes(frames, scale=1)
    if len(content) > _MAX_REPLAY_ARTIFACT_BYTES:
        raise ValueError(
            "Airstriker replay GIF exceeds its bounded artifact limit"
        )
    return content, 1


def _replay_gif_bytes(
    frames: Sequence[
        tuple[int | None, NDArray[numpy.uint8], int | None, float | None]
    ],
    *,
    scale: int,
) -> bytes:
    if not frames:
        raise ValueError("Airstriker replay GIF requires at least one frame")
    rendered = [
        _render_replay_frame(
            frame,
            step_index=step_index,
            action=action,
            reward=reward,
            scale=scale,
        )
        for step_index, frame, action, reward in frames
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
    frame: NDArray[numpy.uint8],
    *,
    step_index: int | None,
    action: int | None,
    reward: float | None,
    scale: int,
) -> Image.Image:
    game_width = _FRAME_SHAPE[1] * scale
    game_height = _FRAME_SHAPE[0] * scale
    image = Image.fromarray(frame, mode="RGB").resize(
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
    if step_index is not None and action is not None and reward is not None:
        label = (
            f"step {step_index}  action {_ACTIONS[action]}  "
            f"reward {reward:g}"
        )
    ImageDraw.Draw(rendered).text(
        (4, game_height + 6),
        label,
        fill=(255, 255, 255),
    )
    return rendered


def _trace_action(value: PolicyValue) -> int:
    if type(value) is not int or not 0 <= value < len(_ACTIONS):
        raise ValueError("Airstriker trace Action is invalid")
    return value


def _trace_frame(value: PolicyValue) -> NDArray[numpy.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != _FRAME_SHAPE
        or len(value.data) != _FRAME_BYTES
    ):
        raise ValueError("Airstriker trace observation is invalid")
    return numpy.frombuffer(value.data, dtype=numpy.uint8).reshape(
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
        raise ValueError("Airstriker contact sheet image is invalid")
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


__all__ = ["AirstrikerBenchmark"]
