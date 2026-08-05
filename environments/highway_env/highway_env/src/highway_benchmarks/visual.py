"""Lossless rendered-frame evidence and derived previews for HighwayEnv."""

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

VISUAL_MAX_FRAMES_PER_EPISODE = 42
VISUAL_INITIAL_FRAME_METRIC = "feedback_visual_initial_rgb"
VISUAL_FRAME_METRIC = "feedback_visual_rgb"
VISUAL_CAPTURE_FAILED_METRIC = "feedback_visual_capture_failed"

_MAX_PREVIEW_ARTIFACT_BYTES = 4 * 1024 * 1024
_PREVIEW_FRAME_DURATION_MS = 200
_PREVIEW_MAX_WIDTH = 320
_PREVIEW_FALLBACK_MAX_WIDTH = 192
_PREVIEW_STATUS_HEIGHT = 36


@dataclass(frozen=True, slots=True)
class _RenderedFrame:
    step_index: int | None
    frame: NDArray[np.uint8]
    reward: float | None
    cumulative_return: float


def visual_feedback(
    records: Sequence[EpisodeRecord],
    *,
    profile: str,
    capture_interval: int,
) -> tuple[tuple[Artifact, ...], list[PolicyValue], int]:
    """Publish lossless captured frames and a derived GIF for each Episode."""

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
                    "schema": "highway-env/rendered-frame-evidence/v1",
                    "episode_index": episode_index,
                    "status": "unavailable",
                    "reason": (
                        "capture_unavailable"
                        if capture_failed
                        else "no_recorded_frames"
                    ),
                    "evidence_artifact": None,
                    "preview_artifact": None,
                    "renderer": "HighwayEnv rgb_array",
                    "source": "upstream_render",
                    "color_space": "RGB",
                    "frame_dtype": "uint8",
                    "frame_shape": None,
                    "capture_interval_steps": capture_interval,
                    "recorded_frames": 0,
                    "steps_without_rendered_frame": record.steps,
                    "complete_for_capture_schedule": False,
                    "complete_for_episode": False,
                }
            )
            continue

        evidence_name = f"episode-{episode_index:03d}/rendered-frames.npz"
        evidence = _frame_evidence_artifact(frames, name=evidence_name)
        artifacts.append(evidence)
        preview_frames, preview_content, preview_width = _encode_bounded_gif(
            frames,
            profile=profile,
        )
        preview_name = f"episode-{episode_index:03d}/road-scene.gif"
        artifacts.append(
            Artifact(
                name=preview_name,
                media_type="image/gif",
                content=preview_content,
                retention="bulk",
            )
        )
        manifests.append(
            {
                "schema": "highway-env/rendered-frame-evidence/v1",
                "episode_index": episode_index,
                "status": "partial" if capture_failed else "available",
                "evidence_artifact": evidence_name,
                "evidence_media_type": evidence.media_type,
                "evidence_artifact_bytes": evidence.size,
                "preview_artifact": preview_name,
                "preview_media_type": "image/gif",
                "renderer": "HighwayEnv rgb_array",
                "source": "upstream_render",
                "color_space": "RGB",
                "frame_dtype": "uint8",
                "frame_shape": list(frames[0].frame.shape),
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
                "preview_frames": len(preview_frames),
                "preview_frames_omitted": len(frames) - len(preview_frames),
                "preview_frame_duration_ms": _PREVIEW_FRAME_DURATION_MS,
                "preview_width": preview_width,
                "timeline": [
                    {
                        "frame_index": frame_index,
                        "step_index": frame.step_index,
                        "reward": frame.reward,
                        "cumulative_return": frame.cumulative_return,
                    }
                    for frame_index, frame in enumerate(preview_frames)
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
        if key not in {VISUAL_INITIAL_FRAME_METRIC, VISUAL_FRAME_METRIC}
    }


def visual_capture_interval(max_episode_steps: int) -> int:
    """Choose a fixed stride that bounds initial/result camera evidence."""

    return max(
        1,
        math.ceil((max_episode_steps + 1) / (VISUAL_MAX_FRAMES_PER_EPISODE - 2)),
    )


def _episode_frames(
    record: EpisodeRecord,
) -> tuple[tuple[_RenderedFrame, ...], bool]:
    frames: list[_RenderedFrame] = []
    cumulative_return = 0.0
    capture_failed = False
    frame_shape: tuple[int, ...] | None = None
    for step_index, transition in enumerate(record.transitions):
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            continue
        capture_failed = (
            capture_failed
            or metrics.get(VISUAL_CAPTURE_FAILED_METRIC) is True
        )
        if step_index == 0 and VISUAL_INITIAL_FRAME_METRIC in metrics:
            initial = _frame(metrics[VISUAL_INITIAL_FRAME_METRIC])
            frame_shape = initial.shape
            frames.append(
                _RenderedFrame(
                    step_index=None,
                    frame=initial,
                    reward=None,
                    cumulative_return=0.0,
                )
            )
        cumulative_return += transition.step.reward
        if VISUAL_FRAME_METRIC in metrics:
            frame = _frame(metrics[VISUAL_FRAME_METRIC])
            if frame_shape is not None and frame.shape != frame_shape:
                raise ValueError("HighwayEnv rendered frame shape changed within an Episode")
            frame_shape = frame.shape
            frames.append(
                _RenderedFrame(
                    step_index=step_index + 1,
                    frame=frame,
                    reward=transition.step.reward,
                    cumulative_return=cumulative_return,
                )
            )
    if len(frames) > VISUAL_MAX_FRAMES_PER_EPISODE:
        raise ValueError("HighwayEnv rendered frame count exceeds its public bound")
    return tuple(frames), capture_failed


def _frame(value: PolicyValue) -> NDArray[np.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or len(value.shape) != 3
        or value.shape[2] != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
    ):
        raise ValueError("HighwayEnv visual metric contains an invalid RGB frame")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(value.shape)


def _frame_evidence_artifact(
    frames: Sequence[_RenderedFrame],
    *,
    name: str,
) -> Artifact:
    if not frames:
        raise ValueError("HighwayEnv frame evidence requires at least one frame")
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


def _encode_bounded_gif(
    frames: Sequence[_RenderedFrame],
    *,
    profile: str,
) -> tuple[tuple[_RenderedFrame, ...], bytes, int]:
    if not frames:
        raise ValueError("HighwayEnv GIF preview requires at least one frame")
    selected = tuple(frames)
    width = _PREVIEW_MAX_WIDTH
    content = _gif_bytes(selected, profile=profile, max_width=width)
    if len(content) <= _MAX_PREVIEW_ARTIFACT_BYTES:
        return selected, content, width
    width = _PREVIEW_FALLBACK_MAX_WIDTH
    content = _gif_bytes(selected, profile=profile, max_width=width)
    while len(content) > _MAX_PREVIEW_ARTIFACT_BYTES and len(selected) > 2:
        selected = (selected[0], *selected[1:-1:2], selected[-1])
        content = _gif_bytes(selected, profile=profile, max_width=width)
    if len(content) > _MAX_PREVIEW_ARTIFACT_BYTES:
        raise ValueError("HighwayEnv GIF preview exceeds its public byte bound")
    return selected, content, width


def _gif_bytes(
    frames: Sequence[_RenderedFrame],
    *,
    profile: str,
    max_width: int,
) -> bytes:
    rendered = [
        _preview_frame(frame, profile=profile, max_width=max_width)
        for frame in frames
    ]
    stream = io.BytesIO()
    rendered[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=_PREVIEW_FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return stream.getvalue()


def _preview_frame(
    frame: _RenderedFrame,
    *,
    profile: str,
    max_width: int,
) -> Image.Image:
    source_height, source_width, _ = frame.frame.shape
    width = min(source_width, max_width)
    height = max(1, round(source_height * width / source_width))
    camera = Image.fromarray(frame.frame)
    if (width, height) != (source_width, source_height):
        camera = camera.resize((width, height), resample=Image.Resampling.LANCZOS)
    rendered = Image.new(
        "RGB",
        (width, height + _PREVIEW_STATUS_HEIGHT),
        color=(0, 0, 0),
    )
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
    "VISUAL_CAPTURE_FAILED_METRIC",
    "VISUAL_FRAME_METRIC",
    "VISUAL_INITIAL_FRAME_METRIC",
    "VISUAL_MAX_FRAMES_PER_EPISODE",
    "trace_metrics",
    "visual_capture_interval",
    "visual_feedback",
]
