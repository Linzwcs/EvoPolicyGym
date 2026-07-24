"""Versioned Agent Session protocol constants and framing."""

from ._framing import JsonFrameCodec

SESSION_PROTOCOL = "agent-session/v1"
SESSION_MAX_FRAME_BYTES = 64 * 1024

SESSION_FRAMES = JsonFrameCodec(
    label="Session",
    max_bytes=SESSION_MAX_FRAME_BYTES,
)

__all__: list[str] = []
