"""ARC-AGI-3 collection planning and official scorecard publication."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import zipfile
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
    Transition,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ._upstream import ArcadeLike, create_arcade
from .config import ArcAgi3Config
from .environment import ArcAgi3Environment

_SEED_DOMAIN = b"evopolicygym-arc-agi-3/episode-seed/v1\0"
_OFFSET_DOMAIN = b"evopolicygym-arc-agi-3/game-offset/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_FRAME_SIZE = 64 * 64
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 8
_MAX_TRACED_STEPS = 128
_MAX_TRACED_FRAME_BYTES_PER_EPISODE = 8 * 1024 * 1024
_MAX_TRACED_FRAME_BYTES_TOTAL = 12 * 1024 * 1024
_MAX_VIDEO_FRAMES = 512
_MAX_VIDEO_ARTIFACT_BYTES = 12 * 1024 * 1024
_VIDEO_FRAME_DURATION_MS = 100
_VIDEO_SCALE = 4
_VIDEO_STATUS_HEIGHT = 32

# ARC-AGI toolkit 0.9.9's official 16-color rendering palette, flattened as
# RGB. GIFs remain palette-indexed and use nearest-neighbor scaling.
_ARC_RGB_PALETTE = bytes(
    (
        255,
        255,
        255,
        204,
        204,
        204,
        153,
        153,
        153,
        102,
        102,
        102,
        51,
        51,
        51,
        0,
        0,
        0,
        229,
        58,
        163,
        255,
        123,
        204,
        249,
        60,
        49,
        30,
        147,
        255,
        136,
        216,
        241,
        255,
        220,
        0,
        255,
        133,
        27,
        146,
        18,
        49,
        79,
        204,
        48,
        163,
        86,
        214,
    )
    + (0,) * (256 * 3 - 16 * 3)
)


@dataclass(frozen=True, slots=True)
class _VideoFrame:
    pixels: bytes
    observation_index: int
    source_frame_index: int
    source_frame_count: int
    state: str
    levels_completed: int
    win_levels: int
    decision_step: int | None


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    transitions: tuple[Transition, ...]
    observations: tuple[PolicyValue, ...]
    frame_bytes: int

    @property
    def frames_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/observations.npz"


class ArcAgi3Benchmark:
    """Official ARC-AGI-3 scoring over one fixed collection of games."""

    def __init__(
        self,
        config: ArcAgi3Config | None = None,
        *,
        arc_api_key: str = "",
        arc_base_url: str = "https://three.arcprize.org",
        environments_dir: str = "runs/arc-agi-3/environments",
        recordings_dir: str = "runs/arc-agi-3/recordings",
        _arcade: ArcadeLike | None = None,
    ) -> None:
        if config is None:
            config = ArcAgi3Config()
        if type(config) is not ArcAgi3Config:
            raise TypeError("config must be ArcAgi3Config")
        for name, value in (
            ("arc_api_key", arc_api_key),
            ("arc_base_url", arc_base_url),
            ("environments_dir", environments_dir),
            ("recordings_dir", recordings_dir),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact string")
        if not arc_base_url:
            raise ValueError("arc_base_url must be non-empty")
        if not environments_dir or not recordings_dir:
            raise ValueError("runtime directories must be non-empty")

        self._config = config
        self._arc_api_key = arc_api_key
        self._arc_base_url = arc_base_url
        self._environments_dir = environments_dir
        self._recordings_dir = recordings_dir
        self._arcade = _arcade
        self._scorecard_id: str | None = None
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
        game_ids = self._config.game_ids
        offset = _offset(split, seed, len(game_ids))
        return tuple(
            EpisodeSpec(
                environment_seed=_seed(split, seed, index),
                scenario={"game_id": game_ids[(offset + index) % len(game_ids)]},
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        game_id = _game_id(episode, allowed=self._config.game_ids)

        arcade = self._get_arcade()
        scorecard_id = self._get_scorecard_id(arcade)
        wrapper = arcade.make(
            game_id,
            seed=episode.environment_seed,
            scorecard_id=scorecard_id,
            save_recording=False,
            include_frame_data=False,
        )
        if wrapper is None:
            raise RuntimeError(f"ARC-AGI-3 could not create {game_id}")
        return ArcAgi3Environment(
            wrapper,
            game_id=game_id,
            max_episode_steps=self._config.max_episode_steps,
        )

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        if self._scorecard_id is None:
            raise RuntimeError("ARC-AGI-3 scorecard was not opened")

        scorecard_id = self._scorecard_id
        arcade = self._get_arcade()
        try:
            scorecard = arcade.close_scorecard(scorecard_id)
        finally:
            self._scorecard_id = None
        if scorecard is None:
            raise RuntimeError("ARC-AGI-3 did not return a final scorecard")

        score = _number(scorecard.score, name="score")
        if not 0.0 <= score <= 115.0:
            raise RuntimeError("ARC-AGI-3 returned an out-of-range score")
        total_environments = _non_negative_int(
            scorecard.total_environments,
            name="total_environments",
        )
        completed_environments = _non_negative_int(
            scorecard.total_environments_completed,
            name="total_environments_completed",
        )
        total_levels = _non_negative_int(
            scorecard.total_levels,
            name="total_levels",
        )
        completed_levels = _non_negative_int(
            scorecard.total_levels_completed,
            name="total_levels_completed",
        )
        total_actions = _non_negative_int(
            scorecard.total_actions,
            name="total_actions",
        )
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        traced = _traced_episodes(records[:_MAX_TRACED_EPISODES])
        traced_steps = {
            episode.episode_index: len(episode.transitions) for episode in traced
        }
        observation_artifacts = tuple(
            artifact
            for episode in traced
            if (artifact := _observation_artifact(episode)) is not None
        )
        video_artifacts, video_manifests = _video_artifacts(
            _video_episodes(records)
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Official ARC-AGI-3 score {score:.3f}; completed "
                    f"{completed_environments}/{total_environments} games and "
                    f"{completed_levels}/{total_levels} levels."
                ),
                "official_score": score,
                "episodes": len(records),
                "mean_steps": statistics.fmean(record.steps for record in records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "completed_environments": completed_environments,
                "total_environments": total_environments,
                "completed_levels": completed_levels,
                "total_levels": total_levels,
                "total_actions": total_actions,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        index=index,
                        traced_steps=traced_steps.get(index, 0),
                    )
                    for index, record in enumerate(summarized)
                ],
                "summarized_episodes": len(summarized),
                "summary_episodes_omitted": len(records) - len(summarized),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "trace_frame_bytes": sum(episode.frame_bytes for episode in traced),
                "trace_frame_bytes_cap_per_episode": (
                    _MAX_TRACED_FRAME_BYTES_PER_EPISODE
                ),
                "trace_frame_bytes_cap_total": _MAX_TRACED_FRAME_BYTES_TOTAL,
                "trace_format": (
                    "trace.jsonl stores every traced observation once; transitions "
                    "reference their decision and result observation_index values. "
                    "Each frames descriptor losslessly references the complete int8 "
                    "TensorValue array in that Episode's observations.npz Artifact. "
                    "Capacity limits omit only whole trailing steps or Episodes; they "
                    "never sample frames from a traced observation."
                ),
                "videos": video_manifests,
                "video_episodes": len(video_artifacts),
                "video_episode_results": len(video_manifests),
                "video_episodes_without_gif": (
                    len(records) - len(video_artifacts)
                ),
                "video_frame_cap_per_episode": _MAX_VIDEO_FRAMES,
                "video_frame_duration_ms": _VIDEO_FRAME_DURATION_MS,
                "video_format": (
                    "Every Episode publishes an animated GIF using the official "
                    "ARC-AGI-3 palette when it has a visual observation. Every "
                    "selected observation contributes its final decision frame; "
                    "additional animation frames are uniformly sampled when the "
                    "cap is reached."
                ),
            },
            artifacts=(
                _trace_artifact(traced),
                *observation_artifacts,
                *video_artifacts,
            ),
        )

    def _get_arcade(self) -> ArcadeLike:
        if self._arcade is None:
            self._arcade = create_arcade(
                arc_api_key=self._arc_api_key,
                arc_base_url=self._arc_base_url,
                environments_dir=self._environments_dir,
                recordings_dir=self._recordings_dir,
            )
        return self._arcade

    def _get_scorecard_id(self, arcade: ArcadeLike) -> str:
        if self._scorecard_id is None:
            scorecard_id = arcade.create_scorecard(tags=["agent", "evopolicygym"])
            if type(scorecard_id) is not str or not scorecard_id:
                raise RuntimeError("ARC-AGI-3 returned an invalid scorecard ID")
            self._scorecard_id = scorecard_id
        return self._scorecard_id


def _spec(config: ArcAgi3Config) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=f"arcprize/ARC-AGI-3/{config.profile}/official-score-v1",
        description=(
            "Interact with complete ARC-AGI-3 games and maximize the official "
            "Relative Human Action Efficiency score. One game is one Episode; "
            "level resets remain inside that Episode."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "frames": {
                    "type": "tensor",
                    "dtype": "int8",
                    "shape": ["animation_frames", 64, 64],
                    "palette": [0, 15],
                },
                "state": {
                    "type": "enum",
                    "values": [
                        "NOT_PLAYED",
                        "NOT_FINISHED",
                        "WIN",
                        "GAME_OVER",
                    ],
                },
                "levels_completed": {"type": "integer", "minimum": 0},
                "win_levels": {"type": "integer", "minimum": 0},
                "available_actions": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 7},
                },
            },
        },
        action_space={
            "type": "tagged_object",
            "reset": {"action": 0},
            "simple": {"action": [1, 2, 3, 4, 5, 7]},
            "coordinate": {
                "action": 6,
                "x": [0, 63],
                "y": [0, 63],
            },
            "notes": ("Non-reset actions must appear in the observation's available_actions list."),
        },
        metadata={
            "environment": "ARC-AGI-3",
            "provider": "ARC Prize Foundation",
            "upstream_version": "0.9.9",
            "arcengine_version": "0.9.3",
            "scoring": "official_rhae_2026",
            "execution": "local",
            "seed_note": (
                "Each Episode seed is passed to Arcade.make(); the toolkit forwards it "
                "only when the selected game constructor declares a seed parameter."
            ),
        },
        environment_parameters={
            "profile": config.profile,
            "game_count": len(config.game_ids),
            "one_game_per_episode": True,
            "level_resets_within_episode": True,
            "episode_seed_supplied": True,
            "seed_effect_game_defined": True,
            "max_animation_frames": 1_001,
            "palette_size": 16,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="official_score",
        score_direction="maximize",
    )


def _game_id(episode: EpisodeSpec, *, allowed: tuple[str, ...]) -> str:
    scenario = episode.scenario
    if type(scenario) is not dict or set(scenario) != {"game_id"}:
        raise ValueError("ARC-AGI-3 Episode scenario must contain only game_id")
    game_id = scenario["game_id"]
    if type(game_id) is not str or game_id not in allowed:
        raise ValueError("ARC-AGI-3 Episode game_id is outside the collection")
    return game_id


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _offset(split: str, seed: int, game_count: int) -> int:
    digest = hashlib.sha256()
    digest.update(_OFFSET_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big") % game_count


def _number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"ARC-AGI-3 returned invalid {name}")
    return float(value)


def _non_negative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"ARC-AGI-3 returned invalid {name}")
    return value


def _episode_summary(
    record: EpisodeRecord,
    *,
    index: int,
    traced_steps: int,
) -> dict[str, PolicyValue]:
    final_observation = (
        record.transitions[-1].step.observation
        if record.transitions
        else record.initial_observation
    )
    final = _compact_observation(final_observation)
    action_counts: dict[str, int] = {}
    for transition in record.transitions:
        action_id = _trace_action(transition.action)["action"]
        assert type(action_id) is int
        key = str(action_id)
        action_counts[key] = action_counts.get(key, 0) + 1
    public_action_counts: dict[str, PolicyValue] = dict(action_counts)
    return {
        "episode_index": index,
        "status": "completed" if record.policy_failure is None else "policy_failed",
        "steps": record.steps,
        "return": record.total_reward,
        "failure": record.policy_failure,
        "action_counts": public_action_counts,
        "final_observation": final,
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _traced_episodes(records: Sequence[EpisodeRecord]) -> tuple[_TracedEpisode, ...]:
    traced: list[_TracedEpisode] = []
    total_frame_bytes = 0
    for episode_index, record in enumerate(records):
        initial_frame_bytes = _observation_frame_bytes(record.initial_observation)
        if total_frame_bytes + initial_frame_bytes > _MAX_TRACED_FRAME_BYTES_TOTAL:
            break

        transitions: list[Transition] = []
        observations: list[PolicyValue] = [record.initial_observation]
        episode_frame_bytes = initial_frame_bytes
        for transition in record.transitions[:_MAX_TRACED_STEPS]:
            result_frame_bytes = _observation_frame_bytes(
                transition.step.observation
            )
            if (
                episode_frame_bytes + result_frame_bytes
                > _MAX_TRACED_FRAME_BYTES_PER_EPISODE
                or total_frame_bytes + episode_frame_bytes + result_frame_bytes
                > _MAX_TRACED_FRAME_BYTES_TOTAL
            ):
                break
            transitions.append(transition)
            observations.append(transition.step.observation)
            episode_frame_bytes += result_frame_bytes

        traced.append(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                transitions=tuple(transitions),
                observations=tuple(observations),
                frame_bytes=episode_frame_bytes,
            )
        )
        total_frame_bytes += episode_frame_bytes
    return tuple(traced)


def _video_episodes(records: Sequence[EpisodeRecord]) -> tuple[_TracedEpisode, ...]:
    episodes: list[_TracedEpisode] = []
    for episode_index, record in enumerate(records):
        candidates = _traced_episodes((record,))
        if candidates:
            candidate = candidates[0]
            episodes.append(
                _TracedEpisode(
                    episode_index=episode_index,
                    record=record,
                    transitions=candidate.transitions,
                    observations=candidate.observations,
                    frame_bytes=candidate.frame_bytes,
                )
            )
            continue
        episodes.append(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                transitions=(),
                observations=(record.initial_observation,),
                frame_bytes=_observation_frame_bytes(record.initial_observation),
            )
        )
    return tuple(episodes)


def _observation_frame_bytes(value: PolicyValue) -> int:
    validated = _validated_observation(value)
    return 0 if validated is None else len(validated[0].data)


def _observation_key(observation_index: int) -> str:
    return f"observation_{observation_index:06d}"


def _observation_artifact(episode: _TracedEpisode) -> Artifact | None:
    arrays: dict[str, NDArray[numpy.int8]] = {}
    for observation_index, value in enumerate(episode.observations):
        validated = _validated_observation(value)
        if validated is None:
            continue
        frames = validated[0]
        arrays[_observation_key(observation_index)] = numpy.frombuffer(
            frames.data,
            dtype=numpy.int8,
        ).reshape(frames.shape)
    if not arrays:
        return None

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for key, array in arrays.items():
            encoded = io.BytesIO()
            numpy.save(encoded, array, allow_pickle=False)
            member = zipfile.ZipInfo(
                f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, encoded.getvalue())
    return Artifact(
        name=episode.frames_artifact_name,
        media_type="application/x-npz",
        content=stream.getvalue(),
    )


def _trace_artifact(records: Sequence[_TracedEpisode]) -> Artifact:
    lines: list[bytes] = []
    for traced_episode in records:
        episode_index = traced_episode.episode_index
        record = traced_episode.record
        traced = traced_episode.transitions
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": record.total_reward,
                    "failure": record.policy_failure,
                    "traced_steps": len(traced),
                    "trace_steps_omitted": record.steps - len(traced),
                }
            )
        )
        lines.append(
            _json_line(
                {
                    "type": "observation",
                    "episode_index": episode_index,
                    "observation_index": 0,
                    "after_step_index": None,
                    "decision_for_step_index": 0 if traced else None,
                    "observation": _trace_observation(
                        record.initial_observation,
                        artifact_name=traced_episode.frames_artifact_name,
                        observation_index=0,
                    ),
                }
            )
        )
        for step_index, transition in enumerate(traced):
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "decision_observation_index": step_index,
                        "action": _trace_action(transition.action),
                        "reward": transition.step.reward,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "result_observation_index": step_index + 1,
                    }
                )
            )
            lines.append(
                _json_line(
                    {
                        "type": "observation",
                        "episode_index": episode_index,
                        "observation_index": step_index + 1,
                        "after_step_index": step_index,
                        "decision_for_step_index": (
                            step_index + 1
                            if step_index + 1 < len(traced)
                            else None
                        ),
                        "observation": _trace_observation(
                            transition.step.observation,
                            artifact_name=traced_episode.frames_artifact_name,
                            observation_index=step_index + 1,
                        ),
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _video_artifacts(
    records: Sequence[_TracedEpisode],
) -> tuple[tuple[Artifact, ...], list[PolicyValue]]:
    artifacts: list[Artifact] = []
    manifests: list[PolicyValue] = []
    for traced_episode in records:
        episode_index = traced_episode.episode_index
        record = traced_episode.record
        traced = traced_episode.transitions
        observations = traced_episode.observations
        validated = [
            (observation_index, _validated_observation(observation))
            for observation_index, observation in enumerate(observations)
        ]
        visual = [
            (observation_index, observation)
            for observation_index, observation in validated
            if observation is not None
        ]
        if not visual:
            manifests.append(
                {
                    "episode_index": episode_index,
                    "status": "unavailable",
                    "reason": "no_visual_observation",
                    "artifact": None,
                    "traced_steps": len(traced),
                    "trace_steps_omitted": record.steps - len(traced),
                    "source_animation_frames": 0,
                    "encoded_frames": 0,
                    "sampled": False,
                    "frame_duration_ms": _VIDEO_FRAME_DURATION_MS,
                    "scale": None,
                    "timeline": [],
                }
            )
            continue

        allocations = _video_frame_allocations(
            [observation[0].shape[0] for _, observation in visual]
        )
        video_frames: list[_VideoFrame] = []
        timeline: list[PolicyValue] = []
        source_frame_total = 0
        sampled_any = False
        for (observation_index, observation), allocation in zip(
            visual,
            allocations,
            strict=True,
        ):
            frames, state, levels_completed, win_levels, available_actions = observation
            source_frames = frames.shape[0]
            selected = _sample_frame_indices(source_frames, allocation)
            video_frame_start = len(video_frames)
            preceding_step = observation_index - 1
            decision_step = (
                observation_index if observation_index < len(traced) else None
            )
            for frame_index in selected:
                offset = frame_index * _FRAME_SIZE
                video_frames.append(
                    _VideoFrame(
                        pixels=frames.data[offset : offset + _FRAME_SIZE],
                        observation_index=observation_index,
                        source_frame_index=frame_index,
                        source_frame_count=source_frames,
                        state=state,
                        levels_completed=levels_completed,
                        win_levels=win_levels,
                        decision_step=decision_step,
                    )
                )
            source_frame_total += source_frames
            sampled_any = sampled_any or len(selected) < source_frames
            timeline.append(
                {
                    "observation_index": observation_index,
                    "after_step_index": (
                        preceding_step if preceding_step >= 0 else None
                    ),
                    "decision_for_step_index": decision_step,
                    "preceding_action": (
                        _trace_action(traced[preceding_step].action)
                        if preceding_step >= 0
                        else None
                    ),
                    "state": state,
                    "levels_completed": levels_completed,
                    "win_levels": win_levels,
                    "available_actions": available_actions,
                    "source_animation_frames": source_frames,
                    "video_frame_start": video_frame_start,
                    "video_frame_count": len(selected),
                    "sampled": len(selected) < source_frames,
                }
            )

        name = f"episode-{episode_index:03d}/playback.gif"
        content, scale = _encode_gif(video_frames)
        artifacts.append(
            Artifact(
                name=name,
                media_type="image/gif",
                content=content,
            )
        )
        manifests.append(
            {
                "episode_index": episode_index,
                "status": "available",
                "artifact": name,
                "traced_steps": len(traced),
                "trace_steps_omitted": record.steps - len(traced),
                "source_animation_frames": source_frame_total,
                "encoded_frames": len(video_frames),
                "sampled": sampled_any,
                "frame_duration_ms": _VIDEO_FRAME_DURATION_MS,
                "scale": scale,
                "timeline": timeline,
            }
        )
    return tuple(artifacts), manifests


def _video_frame_allocations(source_frames: Sequence[int]) -> list[int]:
    if not source_frames:
        return []
    if len(source_frames) > _MAX_VIDEO_FRAMES:
        raise ValueError("ARC-AGI-3 video contains too many observations")
    allocations = [1] * len(source_frames)
    remaining = _MAX_VIDEO_FRAMES - len(source_frames)
    while remaining:
        active = [
            index
            for index, source_count in enumerate(source_frames)
            if allocations[index] < source_count
        ]
        if not active:
            break
        share = max(remaining // len(active), 1)
        for index in active:
            added = min(source_frames[index] - allocations[index], share, remaining)
            allocations[index] += added
            remaining -= added
            if remaining == 0:
                break
    return allocations


def _sample_frame_indices(source_frames: int, selected_frames: int) -> tuple[int, ...]:
    if selected_frames >= source_frames:
        return tuple(range(source_frames))
    if selected_frames == 1:
        return (source_frames - 1,)
    return tuple(
        index * (source_frames - 1) // (selected_frames - 1)
        for index in range(selected_frames)
    )


def _palette_image(frame: bytes) -> Image.Image:
    image = Image.frombytes("P", (64, 64), frame)
    image.putpalette(_ARC_RGB_PALETTE)
    return image


def _encode_gif(frames: Sequence[_VideoFrame]) -> tuple[bytes, int]:
    content = _gif_bytes(frames, scale=_VIDEO_SCALE)
    if len(content) <= _MAX_VIDEO_ARTIFACT_BYTES:
        return content, _VIDEO_SCALE
    content = _gif_bytes(frames, scale=1)
    if len(content) > _MAX_VIDEO_ARTIFACT_BYTES:
        raise ValueError("ARC-AGI-3 video exceeds its bounded artifact limit")
    return content, 1


def _gif_bytes(frames: Sequence[_VideoFrame], *, scale: int) -> bytes:
    if not frames:
        raise ValueError("ARC-AGI-3 video requires at least one frame")
    rendered = [_render_video_frame(frame, scale=scale) for frame in frames]
    stream = io.BytesIO()
    rendered[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=_VIDEO_FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    return stream.getvalue()


def _render_video_frame(frame: _VideoFrame, *, scale: int) -> Image.Image:
    game_size = 64 * scale
    image = _palette_image(frame.pixels).resize(
        (game_size, game_size),
        resample=Image.Resampling.NEAREST,
    )
    rendered = Image.new(
        "P",
        (game_size, game_size + _VIDEO_STATUS_HEIGHT),
        color=5,
    )
    rendered.putpalette(_ARC_RGB_PALETTE)
    rendered.paste(image, (0, 0))
    decision = "-" if frame.decision_step is None else str(frame.decision_step)
    draw = ImageDraw.Draw(rendered)
    draw.text(
        (2, game_size + 2),
        (
            f"obs {frame.observation_index}  "
            f"frame {frame.source_frame_index + 1}/{frame.source_frame_count}  "
            f"decision {decision}"
        ),
        fill=0,
    )
    draw.text(
        (2, game_size + 16),
        (
            f"{frame.state}  "
            f"levels {frame.levels_completed}/{frame.win_levels}"
        ),
        fill=0,
    )
    return rendered


def _compact_observation(value: PolicyValue) -> PolicyValue:
    validated = _validated_observation(value)
    if validated is None:
        return None
    frames, state, levels_completed, win_levels, available_actions = validated
    last_frame = frames.data[-_FRAME_SIZE:]
    return {
        "state": state,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "available_actions": available_actions,
        "animation_frames": frames.shape[0],
        "last_frame_sha256": hashlib.sha256(last_frame).hexdigest(),
    }


def _trace_observation(
    value: PolicyValue,
    *,
    artifact_name: str,
    observation_index: int,
) -> PolicyValue:
    validated = _validated_observation(value)
    if validated is None:
        return None
    frames, state, levels_completed, win_levels, available_actions = validated
    return {
        "frames": {
            "type": "tensor",
            "dtype": frames.dtype,
            "shape": list(frames.shape),
            "encoding": "numpy-npz",
            "artifact": artifact_name,
            "key": _observation_key(observation_index),
            "sha256": hashlib.sha256(frames.data).hexdigest(),
        },
        "state": state,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "available_actions": available_actions,
    }


def _validated_observation(
    value: PolicyValue,
) -> tuple[TensorValue, str, int, int, list[PolicyValue]] | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {
        "frames",
        "state",
        "levels_completed",
        "win_levels",
        "available_actions",
    }:
        raise ValueError("ARC-AGI-3 trace observation is invalid")
    frames = value["frames"]
    state = value["state"]
    levels_completed = value["levels_completed"]
    win_levels = value["win_levels"]
    available_actions = value["available_actions"]
    if (
        type(frames) is not TensorValue
        or frames.dtype != "int8"
        or len(frames.shape) != 3
        or not 1 <= frames.shape[0] <= 1_001
        or frames.shape[1:] != (64, 64)
        or len(frames.data) != frames.shape[0] * _FRAME_SIZE
        or any(pixel > 15 for pixel in frames.data)
        or type(state) is not str
        or state not in {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}
        or type(levels_completed) is not int
        or levels_completed < 0
        or type(win_levels) is not int
        or win_levels < 0
        or type(available_actions) is not list
        or any(type(action) is not int or not 1 <= action <= 7 for action in available_actions)
    ):
        raise ValueError("ARC-AGI-3 trace observation is invalid")
    return (
        frames,
        state,
        levels_completed,
        win_levels,
        list(available_actions),
    )


def _trace_action(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict or "action" not in value:
        raise ValueError("ARC-AGI-3 trace Action is invalid")
    action = value["action"]
    if type(action) is not int or not 0 <= action <= 7:
        raise ValueError("ARC-AGI-3 trace Action is invalid")
    expected = {"action", "x", "y"} if action == 6 else {"action"}
    if set(value) != expected:
        raise ValueError("ARC-AGI-3 trace Action is invalid")
    if action != 6:
        return {"action": action}
    x = value["x"]
    y = value["y"]
    if type(x) is not int or type(y) is not int or not 0 <= x <= 63 or not 0 <= y <= 63:
        raise ValueError("ARC-AGI-3 trace Action is invalid")
    return {"action": action, "x": x, "y": y}


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


__all__ = ["ArcAgi3Benchmark"]
