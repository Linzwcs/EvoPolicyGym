"""Canonical Crafter achievement scoring with bounded public diagnostics."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import subprocess
import tempfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import cast

import imageio_ffmpeg
from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue
from PIL import Image

from .config import CrafterConfig
from .constants import ACHIEVEMENTS, ACTIONS
from .environment import CrafterEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-crafter/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_TRANSITIONS = 2_048
_MAX_CONTACT_SHEET_FRAMES = 16
_CONTACT_SHEET_COLUMNS = 4
_CONTACT_SHEET_FRAME_SIZE = 256
_MAX_REPLAY_FRAMES = 300
_REPLAY_FRAME_SIZE = 512
_REPLAY_FPS = 10
_MAX_TRACE_BYTES = 15 * 1024 * 1024
_MAX_REPLAY_BYTES = 15 * 1024 * 1024
_AGENT_SKILL_NAME = "optimize-crafter-policy"


def _agent_skill() -> str:
    packaged = files("crafter_benchmarks").joinpath(
        "skills",
        _AGENT_SKILL_NAME,
        "SKILL.md",
    )
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).parents[2]
        / "skills"
        / _AGENT_SKILL_NAME
        / "SKILL.md"
    )
    return source.read_text(encoding="utf-8")


class CrafterBenchmark:
    """Official shifted-geometric achievement score over seeded Episodes."""

    def __init__(self, config: CrafterConfig | None = None) -> None:
        selected = CrafterConfig() if config is None else config
        if type(selected) is not CrafterConfig:
            raise TypeError("config must be CrafterConfig or None")
        self._config = selected
        self._spec = _spec(selected)

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
            EpisodeSpec(environment_seed=_episode_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return CrafterEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        achievement_sets = tuple(_scored_achievements(record) for record in records)
        success_rates: dict[str, PolicyValue] = {
            name: 100.0
            * sum(name in achievements for achievements in achievement_sets)
            / len(records)
            for name in ACHIEVEMENTS
        }
        score = _crafter_score(
            tuple(cast(float, success_rates[name]) for name in ACHIEVEMENTS)
        )
        returns = tuple(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        traced = records[:_MAX_TRACED_EPISODES]
        artifacts: list[Artifact] = [_trace_artifact(traced)]
        artifact_manifest: list[dict[str, object]] = []
        for episode_index, record in enumerate(traced):
            contact_indices = _milestone_indices(
                record,
                limit=_MAX_CONTACT_SHEET_FRAMES,
            )
            contact_sheet = _contact_sheet_artifact(
                record,
                episode_index=episode_index,
                indices=contact_indices,
            )
            artifacts.append(contact_sheet)
            manifest_entry: dict[str, object] = {
                "episode_index": episode_index,
                "status": (
                    "completed"
                    if record.policy_failure is None
                    else "policy_failed"
                ),
                "steps": record.steps,
                "return": record.total_reward,
                "failure": record.policy_failure,
                "unlocked_achievements": sorted(
                    _scored_achievements(record)
                ),
                "contact_sheet": {
                    "artifact": contact_sheet.name,
                    "columns": _CONTACT_SHEET_COLUMNS,
                    "frame_size": [
                        _CONTACT_SHEET_FRAME_SIZE,
                        _CONTACT_SHEET_FRAME_SIZE,
                    ],
                    "frames": [
                        _frame_manifest(record, index)
                        for index in contact_indices
                    ],
                    "resampling": "nearest",
                },
            }
            if episode_index == 0:
                replay_indices = _milestone_indices(
                    record,
                    limit=_MAX_REPLAY_FRAMES,
                )
                replay = _replay_artifact(record, replay_indices)
                artifacts.append(replay)
                manifest_entry["replay"] = {
                    "artifact": replay.name,
                    "codec": "h264",
                    "fps": _REPLAY_FPS,
                    "frame_size": [
                        _REPLAY_FRAME_SIZE,
                        _REPLAY_FRAME_SIZE,
                    ],
                    "frames": [
                        {
                            "video_frame": frame,
                            **_frame_manifest(record, observation),
                        }
                        for frame, observation in enumerate(replay_indices)
                    ],
                    "resampling": "nearest",
                }
            artifact_manifest.append(manifest_entry)
        artifacts.append(
            _manifest_artifact(
                artifact_manifest,
                traced_episodes=len(traced),
                omitted_episodes=len(records) - len(traced),
            )
        )

        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Crafter score {score:.3f}% across "
                    f"{len(records)} Episodes."
                ),
                "crafter_score_percent": score,
                "achievement_success_percent": success_rates,
                "mean_return": statistics.fmean(returns),
                "mean_steps": statistics.fmean(record.steps for record in records),
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "policy_failures": sum(
                    record.policy_failure is not None for record in records
                ),
                "failure_achievement_credit": 0.0,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "max_traced_transitions_per_episode": _MAX_TRACED_TRANSITIONS,
            },
            artifacts=tuple(artifacts),
        )


def _spec(config: CrafterConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="crafter/CrafterReward-v1/achievement-score-v1",
        description=(
            "Survive and unlock Crafter's 22 achievements from canonical "
            "64x64 RGB observations. Maximize the official shifted-geometric "
            "achievement success score."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [64, 64, 3],
            "color_space": "RGB",
        },
        action_space={
            "type": "discrete",
            "values": list(range(len(ACTIONS))),
            "meaning": {
                str(index): name for index, name in enumerate(ACTIONS)
            },
        },
        metadata={
            "environment": "CrafterReward-v1",
            "provider": "danijar/crafter",
            "upstream_version": "1.8.3",
            "upstream_url": "https://github.com/danijar/crafter",
            "upstream_license": "MIT",
            "achievements": list(ACHIEVEMENTS),
            "official_score_formula": (
                "exp(mean(log(1 + success_percent))) - 1"
            ),
            "privileged_information_exposed": False,
        },
        environment_parameters={
            "area": [64, 64],
            "view": [9, 9],
            "image_size": [64, 64],
            "reward": True,
            "max_episode_steps": config.max_episode_steps,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="crafter_score_percent",
        score_direction="maximize",
        agent_skill=_agent_skill(),
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _scored_achievements(record: EpisodeRecord) -> frozenset[str]:
    if record.policy_failure is not None:
        return frozenset()
    unlocked: set[str] = set()
    for transition in record.transitions:
        unlocked.update(_transition_achievements(transition.step.metrics))
    return frozenset(unlocked)


def _transition_achievements(metrics: PolicyValue) -> tuple[str, ...]:
    if type(metrics) is not dict or set(metrics) != {"achievements_unlocked"}:
        raise ValueError("Crafter transition metrics are invalid")
    value = metrics["achievements_unlocked"]
    if type(value) is not list:
        raise ValueError("Crafter transition achievements are invalid")
    if any(type(name) is not str or name not in ACHIEVEMENTS for name in value):
        raise ValueError("Crafter transition achievements are invalid")
    names = cast(list[str], value)
    if len(names) != len(set(names)):
        raise ValueError("Crafter transition achievements contain duplicates")
    return tuple(names)


def _crafter_score(success_rates: Sequence[float]) -> float:
    if len(success_rates) != len(ACHIEVEMENTS):
        raise ValueError("Crafter score requires every achievement")
    if any(not math.isfinite(rate) or not 0.0 <= rate <= 100.0 for rate in success_rates):
        raise ValueError("Crafter success rates are invalid")
    return math.expm1(statistics.fmean(math.log1p(rate) for rate in success_rates))


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


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        selected = _trace_indices(len(record.transitions))
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
                    "return": record.total_reward,
                    "failure": record.policy_failure,
                    "unlocked_achievements": sorted(
                        _scored_achievements(record)
                    ),
                    "traced_transitions": len(selected),
                    "transitions_omitted": record.steps - len(selected),
                }
            )
        )
        for step_index in selected:
            transition = record.transitions[step_index]
            if type(transition.action) is not int or transition.action not in range(
                len(ACTIONS)
            ):
                raise ValueError("Crafter trace Action is invalid")
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": transition.action,
                        "action_name": ACTIONS[transition.action],
                        "reward": transition.step.reward,
                        "achievements_unlocked": list(
                            _transition_achievements(transition.step.metrics)
                        ),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
    content = b"".join(lines)
    if len(content) > _MAX_TRACE_BYTES:
        raise RuntimeError("bounded Crafter trace exceeded its byte budget")
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=content,
    )


def _trace_indices(length: int) -> tuple[int, ...]:
    if length <= _MAX_TRACED_TRANSITIONS:
        return tuple(range(length))
    half = _MAX_TRACED_TRANSITIONS // 2
    return tuple(range(half)) + tuple(range(length - half, length))


def _contact_sheet_artifact(
    record: EpisodeRecord,
    *,
    episode_index: int,
    indices: Sequence[int],
) -> Artifact:
    observations = [record.initial_observation]
    observations.extend(
        transition.step.observation for transition in record.transitions
    )
    images = [
        _policy_image(observations[index]).resize(
            (_CONTACT_SHEET_FRAME_SIZE, _CONTACT_SHEET_FRAME_SIZE),
            Image.Resampling.NEAREST,
        )
        for index in indices
    ]
    rows = math.ceil(_MAX_CONTACT_SHEET_FRAMES / _CONTACT_SHEET_COLUMNS)
    sheet = Image.new(
        "RGB",
        (
            _CONTACT_SHEET_FRAME_SIZE * _CONTACT_SHEET_COLUMNS,
            _CONTACT_SHEET_FRAME_SIZE * rows,
        ),
    )
    for index, image in enumerate(images):
        column = index % _CONTACT_SHEET_COLUMNS
        row = index // _CONTACT_SHEET_COLUMNS
        sheet.paste(
            image,
            (
                _CONTACT_SHEET_FRAME_SIZE * column,
                _CONTACT_SHEET_FRAME_SIZE * row,
            ),
        )
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=True)
    return Artifact(
        name=f"episode-{episode_index}-frames.png",
        media_type="image/png",
        content=output.getvalue(),
    )


def _milestone_indices(
    record: EpisodeRecord,
    *,
    limit: int,
) -> tuple[int, ...]:
    length = len(record.transitions) + 1
    if length <= limit:
        return tuple(range(length))
    achievements = tuple(
        index + 1
        for index, transition in enumerate(record.transitions)
        if _transition_achievements(transition.step.metrics)
    )
    required = {0, length - 1}
    available_for_achievements = max(0, limit - len(required))
    required.update(
        _sample_values(achievements, limit=available_for_achievements)
    )
    for index in _sample_indices(length, limit=limit):
        if len(required) == limit:
            break
        required.add(index)
    if len(required) < limit:
        for index in range(length):
            required.add(index)
            if len(required) == limit:
                break
    return tuple(sorted(required))


def _sample_values(
    values: Sequence[int],
    *,
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0 or not values:
        return ()
    if len(values) <= limit:
        return tuple(values)
    positions = _sample_indices(len(values), limit=limit)
    return tuple(values[index] for index in positions)


def _sample_indices(length: int, *, limit: int) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("sample length must be positive")
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    if length <= limit:
        return tuple(range(length))
    if limit == 1:
        return (0,)
    return tuple(
        round(index * (length - 1) / (limit - 1))
        for index in range(limit)
    )


def _replay_artifact(
    record: EpisodeRecord,
    indices: Sequence[int],
) -> Artifact:
    observations = [record.initial_observation]
    observations.extend(
        transition.step.observation for transition in record.transitions
    )
    with tempfile.TemporaryDirectory(prefix="crafter-replay-") as temporary:
        target = Path(temporary) / "episode-0-replay.mp4"
        command = (
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{_REPLAY_FRAME_SIZE}x{_REPLAY_FRAME_SIZE}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(_REPLAY_FPS),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stderr is not None
        try:
            for index in indices:
                image = _policy_image(observations[index]).resize(
                    (_REPLAY_FRAME_SIZE, _REPLAY_FRAME_SIZE),
                    Image.Resampling.NEAREST,
                )
                process.stdin.write(image.tobytes())
            process.stdin.close()
            error = process.stderr.read()
            process.stderr.close()
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if returncode != 0:
            raise RuntimeError(
                "Crafter replay encoding failed: "
                + error.decode("utf-8", errors="replace")[:1_000]
            )
        content = target.read_bytes()
    if len(content) > _MAX_REPLAY_BYTES:
        raise RuntimeError("bounded Crafter replay exceeded its byte budget")
    return Artifact(
        name="episode-0-replay.mp4",
        media_type="video/mp4",
        content=content,
    )


def _frame_manifest(
    record: EpisodeRecord,
    observation_index: int,
) -> dict[str, object]:
    achievements: tuple[str, ...] = ()
    step_index: int | None = None
    if observation_index > 0:
        step_index = observation_index - 1
        achievements = _transition_achievements(
            record.transitions[step_index].step.metrics
        )
    return {
        "observation_index": observation_index,
        "step_index": step_index,
        "achievements_unlocked": list(achievements),
    }


def _manifest_artifact(
    episodes: Sequence[dict[str, object]],
    *,
    traced_episodes: int,
    omitted_episodes: int,
) -> Artifact:
    content = json.dumps(
        {
            "schema": "crafter/artifact-manifest/v1",
            "source_observation": {
                "color_space": "RGB",
                "dtype": "uint8",
                "shape": [64, 64, 3],
            },
            "trace": {
                "artifact": "trace.jsonl",
                "max_transitions_per_episode": _MAX_TRACED_TRANSITIONS,
            },
            "traced_episodes": traced_episodes,
            "omitted_episodes": omitted_episodes,
            "episodes": list(episodes),
        },
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    return Artifact(
        name="artifact-manifest.json",
        media_type="application/json",
        content=content,
    )


def _policy_image(value: PolicyValue) -> Image.Image:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != (64, 64, 3)
        or len(value.data) != 64 * 64 * 3
    ):
        raise ValueError("Crafter replay observation is invalid")
    return Image.frombytes("RGB", (64, 64), value.data)


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


__all__ = ["CrafterBenchmark"]
