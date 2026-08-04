"""Bounded camera replay evidence derived from Host-only Step metrics."""

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


def video_camera(profile: str) -> str | int:
    """Select one stable public camera for each Robotics task family."""

    if profile.startswith("fetch-"):
        return "external_camera_0"
    if profile == "ant-maze":
        return "track"
    if profile == "point-maze":
        return -1
    if profile.startswith("adroit-hand-"):
        return "fixed"
    if profile == "franka-kitchen":
        return "left_cap"
    return -1


def video_camera_label(camera: str | int) -> str:
    return camera if type(camera) is str else "free"


def video_feedback(
    records: Sequence[EpisodeRecord],
    *,
    profile: str,
    camera: str,
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
                    "episode_index": episode_index,
                    "status": "unavailable",
                    "reason": "no_recorded_frames",
                    "artifact": None,
                    "camera": camera,
                    "frame_shape": list(VIDEO_FRAME_SHAPE),
                    "capture_interval_steps": capture_interval,
                    "recorded_frames": 0,
                    "steps_without_video_frame": record.steps,
                }
            )
            continue
        selected, content, scale = _encode_bounded_gif(frames, profile=profile)
        name = f"episode-{episode_index:03d}/robot-camera.gif"
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
                "episode_index": episode_index,
                "status": "partial" if capture_failed else "available",
                "artifact": name,
                "camera": camera,
                "frame_shape": list(VIDEO_FRAME_SHAPE),
                "capture_interval_steps": capture_interval,
                "recorded_frames": len(frames),
                "steps_without_video_frame": max(0, record.steps - (len(frames) - 1)),
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
        raise ValueError("Gymnasium-Robotics video frame count exceeds its public bound")
    return tuple(frames), capture_failed


def _frame(value: PolicyValue) -> NDArray[np.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != VIDEO_FRAME_SHAPE
    ):
        raise ValueError("Gymnasium-Robotics video metric contains an invalid RGB frame")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(VIDEO_FRAME_SHAPE)


def _encode_bounded_gif(
    frames: Sequence[_ReplayFrame],
    *,
    profile: str,
) -> tuple[tuple[_ReplayFrame, ...], bytes, int]:
    if not frames:
        raise ValueError("Gymnasium-Robotics replay GIF requires at least one frame")
    selected = tuple(frames)
    content = _gif_bytes(selected, profile=profile, scale=2)
    if len(content) <= _MAX_VIDEO_ARTIFACT_BYTES:
        return selected, content, 2
    content = _gif_bytes(selected, profile=profile, scale=1)
    while len(content) > _MAX_VIDEO_ARTIFACT_BYTES and len(selected) > 2:
        selected = (selected[0], *selected[1:-1:2], selected[-1])
        content = _gif_bytes(selected, profile=profile, scale=1)
    if len(content) > _MAX_VIDEO_ARTIFACT_BYTES:
        raise ValueError("Gymnasium-Robotics replay GIF exceeds its public byte bound")
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
    "VIDEO_CAPTURE_FAILED_METRIC",
    "VIDEO_FRAME_HEIGHT",
    "VIDEO_FRAME_METRIC",
    "VIDEO_FRAME_SHAPE",
    "VIDEO_FRAME_WIDTH",
    "VIDEO_INITIAL_FRAME_METRIC",
    "VIDEO_MAX_FRAMES_PER_EPISODE",
    "trace_metrics",
    "video_camera",
    "video_camera_label",
    "video_capture_interval",
    "video_feedback",
]
