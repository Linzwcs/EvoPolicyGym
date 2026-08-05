"""Lossless RGB evidence and derived MP4 video for BipedalWalker."""

from __future__ import annotations

import io
import math
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
from evopolicygym.authoring import Artifact, EpisodeRecord
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

VISUAL_MAX_FRAMES_PER_EPISODE = 42
VISUAL_INITIAL_FRAME_METRIC = "feedback_visual_initial_rgb"
VISUAL_FRAME_METRIC = "feedback_visual_rgb"
VISUAL_CAPTURE_FAILED_METRIC = "feedback_visual_capture_failed"
VISUAL_FRAME_SHAPE = (400, 600, 3)

_SCHEMA = "gymnasium-bipedal-walker/rendered-frame-evidence/v1"
_VIDEO_FRAMES_PER_SECOND = 5
_MAX_VIDEO_ARTIFACT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _RenderedFrame:
    step_index: int | None
    frame: NDArray[np.uint8]
    reward: float | None
    cumulative_return: float


def visual_feedback(
    records: Sequence[EpisodeRecord],
    *,
    capture_interval: int,
) -> tuple[tuple[Artifact, ...], list[PolicyValue], int]:
    """Publish raw captured frames and a derived MP4 for each Episode."""

    artifacts: list[Artifact] = []
    manifests: list[PolicyValue] = []
    unavailable = 0
    for episode_index, record in enumerate(records):
        frames, capture_failed = _episode_frames(record)
        if capture_failed or not frames:
            unavailable += 1
        if not frames:
            manifests.append(
                _unavailable_manifest(
                    episode_index=episode_index,
                    record=record,
                    capture_interval=capture_interval,
                    capture_failed=capture_failed,
                )
            )
            continue

        evidence_name = f"episode-{episode_index:03d}/rendered-frames.npz"
        evidence = _frame_evidence_artifact(frames, name=evidence_name)
        artifacts.append(evidence)
        video_name = f"episode-{episode_index:03d}/behavior.mp4"
        try:
            video = _video_artifact(frames, name=video_name)
        except Exception:
            video = None
        if video is not None:
            artifacts.append(video)
        manifests.append(
            {
                "schema": _SCHEMA,
                "episode_index": episode_index,
                "status": "partial" if capture_failed else "available",
                "evidence_artifact": evidence_name,
                "evidence_media_type": evidence.media_type,
                "evidence_artifact_bytes": evidence.size,
                "video_status": "available" if video is not None else "unavailable",
                "video_reason": None if video is not None else "encoding_failed",
                "video_artifact": video_name if video is not None else None,
                "video_media_type": video.media_type if video is not None else None,
                "video_artifact_bytes": video.size if video is not None else None,
                "renderer": "Gymnasium rgb_array",
                "source": "upstream_render",
                "color_space": "RGB",
                "frame_dtype": "uint8",
                "frame_shape": list(VISUAL_FRAME_SHAPE),
                "capture_interval_steps": capture_interval,
                "recorded_frames": len(frames),
                "steps_without_rendered_frame": max(
                    0,
                    record.steps - (len(frames) - 1),
                ),
                "complete_for_capture_schedule": not capture_failed,
                "complete_for_episode": (
                    not capture_failed
                    and capture_interval == 1
                    and len(frames) == record.steps + 1
                ),
                "video_codec": "H.264",
                "video_pixel_format": "yuv420p",
                "video_frames_per_second": _VIDEO_FRAMES_PER_SECOND,
                "video_frames": len(frames),
                "timeline": [
                    {
                        "frame_index": frame_index,
                        "step_index": frame.step_index,
                        "reward": frame.reward,
                        "cumulative_return": frame.cumulative_return,
                    }
                    for frame_index, frame in enumerate(frames)
                ],
            }
        )
    return tuple(artifacts), manifests, unavailable


def trace_metrics(value: PolicyValue) -> PolicyValue:
    """Remove RGB tensors from JSON traces while retaining capture status."""

    if type(value) is not dict:
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in {VISUAL_INITIAL_FRAME_METRIC, VISUAL_FRAME_METRIC}
    }


def visual_capture_interval(max_episode_steps: int) -> int:
    return max(
        1,
        math.ceil((max_episode_steps + 1) / (VISUAL_MAX_FRAMES_PER_EPISODE - 2)),
    )


def _unavailable_manifest(
    *,
    episode_index: int,
    record: EpisodeRecord,
    capture_interval: int,
    capture_failed: bool,
) -> PolicyValue:
    return {
        "schema": _SCHEMA,
        "episode_index": episode_index,
        "status": "unavailable",
        "reason": "capture_failed" if capture_failed else "no_recorded_frames",
        "evidence_artifact": None,
        "video_status": "unavailable",
        "video_reason": "no_recorded_frames",
        "video_artifact": None,
        "renderer": "Gymnasium rgb_array",
        "source": "upstream_render",
        "color_space": "RGB",
        "frame_dtype": "uint8",
        "frame_shape": list(VISUAL_FRAME_SHAPE),
        "capture_interval_steps": capture_interval,
        "recorded_frames": 0,
        "steps_without_rendered_frame": record.steps,
        "complete_for_capture_schedule": False,
        "complete_for_episode": False,
    }


def _episode_frames(
    record: EpisodeRecord,
) -> tuple[tuple[_RenderedFrame, ...], bool]:
    frames: list[_RenderedFrame] = []
    cumulative_return = 0.0
    capture_failed = False
    for step_index, transition in enumerate(record.transitions):
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            continue
        capture_failed = (
            capture_failed
            or metrics.get(VISUAL_CAPTURE_FAILED_METRIC) is True
        )
        if step_index == 0 and VISUAL_INITIAL_FRAME_METRIC in metrics:
            frames.append(
                _RenderedFrame(
                    step_index=None,
                    frame=_frame(metrics[VISUAL_INITIAL_FRAME_METRIC]),
                    reward=None,
                    cumulative_return=0.0,
                )
            )
        cumulative_return += transition.step.reward
        if VISUAL_FRAME_METRIC in metrics:
            frames.append(
                _RenderedFrame(
                    step_index=step_index + 1,
                    frame=_frame(metrics[VISUAL_FRAME_METRIC]),
                    reward=transition.step.reward,
                    cumulative_return=cumulative_return,
                )
            )
    if len(frames) > VISUAL_MAX_FRAMES_PER_EPISODE:
        raise ValueError("BipedalWalker rendered frame count exceeds its public bound")
    return tuple(frames), capture_failed


def _frame(value: PolicyValue) -> NDArray[np.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != VISUAL_FRAME_SHAPE
    ):
        raise ValueError("BipedalWalker visual metric contains an invalid RGB frame")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(VISUAL_FRAME_SHAPE)


def _frame_evidence_artifact(
    frames: Sequence[_RenderedFrame],
    *,
    name: str,
) -> Artifact:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        frames=np.stack([frame.frame for frame in frames]),
        step_indices=np.asarray(
            [-1 if frame.step_index is None else frame.step_index for frame in frames],
            dtype=np.int32,
        ),
        rewards=np.asarray(
            [0.0 if frame.reward is None else frame.reward for frame in frames],
            dtype=np.float64,
        ),
        reward_present=np.asarray(
            [frame.reward is not None for frame in frames],
            dtype=np.bool_,
        ),
        cumulative_returns=np.asarray(
            [frame.cumulative_return for frame in frames],
            dtype=np.float64,
        ),
    )
    return Artifact(
        name=name,
        media_type="application/x-npz",
        content=output.getvalue(),
        retention="bulk",
    )


def _video_artifact(
    frames: Sequence[_RenderedFrame],
    *,
    name: str,
) -> Artifact:
    width = VISUAL_FRAME_SHAPE[1]
    height = VISUAL_FRAME_SHAPE[0]
    with tempfile.TemporaryDirectory(prefix="evopolicygym-bipedal-walker-video-") as tmp:
        path = Path(tmp, "behavior.mp4")
        writer = cast(
            Generator[None, NDArray[np.uint8] | None, None],
            imageio_ffmpeg.write_frames(
                str(path),
                (width, height),
                fps=_VIDEO_FRAMES_PER_SECOND,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart", "-threads", "1"],
            ),
        )
        try:
            writer.send(None)
            for frame in frames:
                writer.send(np.ascontiguousarray(frame.frame))
        finally:
            writer.close()
        content = path.read_bytes()
    if len(content) > _MAX_VIDEO_ARTIFACT_BYTES:
        raise ValueError("BipedalWalker MP4 exceeds its public byte bound")
    if len(content) < 12 or content[4:8] != b"ftyp":
        raise ValueError("BipedalWalker MP4 encoder returned an invalid container")
    return Artifact(
        name=name,
        media_type="video/mp4",
        content=content,
        retention="bulk",
    )


__all__ = [
    "VISUAL_CAPTURE_FAILED_METRIC",
    "VISUAL_FRAME_METRIC",
    "VISUAL_FRAME_SHAPE",
    "VISUAL_INITIAL_FRAME_METRIC",
    "VISUAL_MAX_FRAMES_PER_EPISODE",
    "trace_metrics",
    "visual_capture_interval",
    "visual_feedback",
]
