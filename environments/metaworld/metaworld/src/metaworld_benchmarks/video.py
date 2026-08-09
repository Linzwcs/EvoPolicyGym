"""Lossless camera evidence and derived previews from Host-only Step metrics."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from evopolicygym.authoring import Artifact, EpisodeRecord
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class VideoCamera:
    """Versioned free-camera parameters for comparable visual evidence."""

    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


VIDEO_CAMERA = VideoCamera(
    name="overview-v1",
    lookat=(0.0, 0.65, 0.12),
    distance=1.35,
    azimuth=160.0,
    elevation=-22.0,
)
VIDEO_FRAME_HEIGHT = 128
VIDEO_FRAME_WIDTH = 128
VIDEO_FRAME_SHAPE = (VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH, 3)
VIDEO_MAX_FRAMES_PER_EPISODE = 42
VIDEO_INITIAL_FRAME_METRIC = "feedback_video_initial_rgb"
VIDEO_FRAME_METRIC = "feedback_video_rgb"
VIDEO_CAPTURE_FAILED_METRIC = "feedback_video_capture_failed"

_MAX_VIDEO_ARTIFACT_BYTES = 4 * 1024 * 1024
_VIDEO_FRAME_DURATION_MS = 200
_VIDEO_STATUS_HEIGHT = 36


@dataclass(frozen=True, slots=True)
class _ReplayFrame:
    step_index: int | None
    frame: NDArray[np.uint8]
    reward: float | None
    cumulative_return: float


def video_feedback(
    records: Sequence[EpisodeRecord],
    *,
    profile: str,
    capture_interval: int,
) -> tuple[tuple[Artifact, ...], list[PolicyValue], int]:
    """Build one replay GIF and one manifest for every Episode."""

    artifacts: list[Artifact] = []
    manifests: list[PolicyValue] = []
    unavailable = 0
    for episode_index, record in enumerate(records):
        frames, capture_failed = _episode_frames(record)
        if capture_failed or not frames:
            unavailable += 1
        if not frames:
            manifests.append(
                {
                    "schema": "metaworld/rendered-frame-evidence/v1",
                    "episode_index": episode_index,
                    "status": "unavailable",
                    "reason": "no_recorded_frames",
                    "artifact": None,
                    "frames_artifact": None,
                    "evidence_artifact": None,
                    "preview_artifact": None,
                    "camera": VIDEO_CAMERA.name,
                    "source": "host_camera",
                    "color_space": "RGB",
                    "frame_dtype": "uint8",
                    "frame_shape": list(VIDEO_FRAME_SHAPE),
                    "capture_interval_steps": capture_interval,
                    "recorded_frames": 0,
                    "steps_without_video_frame": record.steps,
                    "complete_for_capture_schedule": False,
                    "complete_for_episode": False,
                }
            )
            continue
        frames_name = f"episode-{episode_index:03d}/rendered-frames.npz"
        frame_evidence = _frame_evidence_artifact(frames, name=frames_name)
        artifacts.append(frame_evidence)
        selected, content, scale = _encode_bounded_gif(frames, profile=profile)
        name = f"episode-{episode_index:03d}/{VIDEO_CAMERA.name}.gif"
        artifacts.append(
            Artifact(
                name=name,
                media_type="image/gif",
                content=content,
                retention="bulk",
            )
        )
        manifests.append(
            {
                "schema": "metaworld/rendered-frame-evidence/v1",
                "episode_index": episode_index,
                "status": "partial" if capture_failed else "available",
                "artifact": name,
                "frames_artifact": frames_name,
                "evidence_artifact": frames_name,
                "evidence_media_type": frame_evidence.media_type,
                "evidence_artifact_bytes": frame_evidence.size,
                "preview_artifact": name,
                "preview_media_type": "image/gif",
                "camera": VIDEO_CAMERA.name,
                "source": "host_camera",
                "color_space": "RGB",
                "frame_dtype": "uint8",
                "frame_shape": list(VIDEO_FRAME_SHAPE),
                "capture_interval_steps": capture_interval,
                "recorded_frames": len(frames),
                "steps_without_video_frame": max(0, record.steps - (len(frames) - 1)),
                "complete_for_capture_schedule": not capture_failed,
                "complete_for_episode": (
                    not capture_failed
                    and capture_interval == 1
                    and len(frames) == record.steps + 1
                ),
                "encoded_frames": len(selected),
                "encoded_frames_omitted": len(frames) - len(selected),
                "frame_duration_ms": _VIDEO_FRAME_DURATION_MS,
                "display_scale": scale,
                "timeline": [
                    {
                        "frame_index": frame_index,
                        "step_index": frame.step_index,
                        "reward": frame.reward,
                        "cumulative_return": frame.cumulative_return,
                    }
                    for frame_index, frame in enumerate(selected)
                ],
            }
        )
    return tuple(artifacts), manifests, unavailable


def _frame_evidence_artifact(
    frames: Sequence[_ReplayFrame],
    *,
    name: str,
) -> Artifact:
    """Preserve every captured RGB frame without presentation transforms."""

    if not frames:
        raise ValueError("MetaWorld frame evidence requires at least one frame")
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


def trace_metrics(value: PolicyValue) -> PolicyValue:
    """Remove raw RGB tensors from JSON traces while retaining capture status."""

    if type(value) is not dict:
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in {VIDEO_INITIAL_FRAME_METRIC, VIDEO_FRAME_METRIC}
    }


def video_capture_interval(max_episode_steps: int) -> int:
    """Choose a fixed stride that bounds initial/result replay evidence."""

    return max(
        1,
        math.ceil((max_episode_steps + 1) / (VIDEO_MAX_FRAMES_PER_EPISODE - 2)),
    )


def _episode_frames(record: EpisodeRecord) -> tuple[tuple[_ReplayFrame, ...], bool]:
    frames: list[_ReplayFrame] = []
    cumulative_return = 0.0
    capture_failed = False
    for step_index, transition in enumerate(record.transitions):
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            continue
        capture_failed = capture_failed or metrics.get(VIDEO_CAPTURE_FAILED_METRIC) is True
        if step_index == 0 and VIDEO_INITIAL_FRAME_METRIC in metrics:
            frames.append(
                _ReplayFrame(
                    step_index=None,
                    frame=_frame(metrics[VIDEO_INITIAL_FRAME_METRIC]),
                    reward=None,
                    cumulative_return=0.0,
                )
            )
        cumulative_return += transition.step.reward
        if VIDEO_FRAME_METRIC in metrics:
            frames.append(
                _ReplayFrame(
                    step_index=step_index + 1,
                    frame=_frame(metrics[VIDEO_FRAME_METRIC]),
                    reward=transition.step.reward,
                    cumulative_return=cumulative_return,
                )
            )
    if len(frames) > VIDEO_MAX_FRAMES_PER_EPISODE:
        raise ValueError("MetaWorld video frame count exceeds its public bound")
    return tuple(frames), capture_failed


def _frame(value: PolicyValue) -> NDArray[np.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != VIDEO_FRAME_SHAPE
    ):
        raise ValueError("MetaWorld video metric contains an invalid RGB frame")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(VIDEO_FRAME_SHAPE)


def _encode_bounded_gif(
    frames: Sequence[_ReplayFrame],
    *,
    profile: str,
) -> tuple[tuple[_ReplayFrame, ...], bytes, int]:
    if not frames:
        raise ValueError("MetaWorld replay GIF requires at least one frame")
    selected = tuple(frames)
    content = _gif_bytes(selected, profile=profile, scale=2)
    if len(content) <= _MAX_VIDEO_ARTIFACT_BYTES:
        return selected, content, 2
    content = _gif_bytes(selected, profile=profile, scale=1)
    while len(content) > _MAX_VIDEO_ARTIFACT_BYTES and len(selected) > 2:
        selected = (selected[0], *selected[1:-1:2], selected[-1])
        content = _gif_bytes(selected, profile=profile, scale=1)
    if len(content) > _MAX_VIDEO_ARTIFACT_BYTES:
        raise ValueError("MetaWorld replay GIF exceeds its public byte bound")
    return selected, content, 1


def _gif_bytes(
    frames: Sequence[_ReplayFrame],
    *,
    profile: str,
    scale: int,
) -> bytes:
    rendered = [_render_frame(frame, profile=profile, scale=scale) for frame in frames]
    stream = io.BytesIO()
    rendered[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=_VIDEO_FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return stream.getvalue()


def _render_frame(
    frame: _ReplayFrame,
    *,
    profile: str,
    scale: int,
) -> Image.Image:
    width = VIDEO_FRAME_WIDTH * scale
    height = VIDEO_FRAME_HEIGHT * scale
    camera = Image.fromarray(frame.frame).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    rendered = Image.new("RGB", (width, height + _VIDEO_STATUS_HEIGHT), color=(0, 0, 0))
    rendered.paste(camera, (0, 0))
    label = f"{profile} | initial"
    if frame.step_index is not None and frame.reward is not None:
        label = (
            f"{profile} | step {frame.step_index} | r={frame.reward:.3g} "
            f"| total={frame.cumulative_return:.3g}"
        )
    ImageDraw.Draw(rendered).text((4, height + 10), label, fill=(255, 255, 255))
    return rendered


__all__ = [
    "VIDEO_CAMERA",
    "VIDEO_CAPTURE_FAILED_METRIC",
    "VIDEO_FRAME_HEIGHT",
    "VIDEO_FRAME_METRIC",
    "VIDEO_FRAME_SHAPE",
    "VIDEO_FRAME_WIDTH",
    "VIDEO_INITIAL_FRAME_METRIC",
    "VIDEO_MAX_FRAMES_PER_EPISODE",
    "VideoCamera",
    "trace_metrics",
    "video_capture_interval",
    "video_feedback",
]
