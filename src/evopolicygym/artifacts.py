"""Bounded public Artifact values published by a Benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
FEEDBACK_MAX_ARTIFACTS = 64
FEEDBACK_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Artifact:
    """One bounded public file represented without a Host path."""

    name: str
    media_type: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _validate_artifact_name(self.name)
        if not self.media_type or "\r" in self.media_type or "\n" in self.media_type:
            raise ValueError("artifact media_type must be a non-empty single line")
        if type(self.content) is not bytes:
            raise TypeError("artifact content must be exact bytes")
        if len(self.content) > ARTIFACT_MAX_BYTES:
            raise ValueError("artifact content exceeds the public byte limit")

    @property
    def size(self) -> int:
        return len(self.content)

    def read_bytes(self) -> bytes:
        return bytes(self.content)


def _validate_artifact_name(name: str) -> None:
    if type(name) is not str:
        raise TypeError("artifact name must be text")
    if not name or name.startswith("/") or "\\" in name or "\0" in name:
        raise ValueError("artifact name must be a relative POSIX path")
    if any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("artifact name contains an unsafe path component")


__all__ = [
    "ARTIFACT_MAX_BYTES",
    "Artifact",
    "FEEDBACK_MAX_ARTIFACTS",
    "FEEDBACK_MAX_ARTIFACT_BYTES",
]
