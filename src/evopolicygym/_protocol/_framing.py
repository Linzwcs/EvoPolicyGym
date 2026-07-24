"""Shared bounded length-prefixed JSON framing mechanism."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Never, cast


@dataclass(frozen=True, slots=True)
class JsonFrameCodec:
    """Pure codec for one named bounded JSON-object frame."""

    label: str
    max_bytes: int

    def encode(self, message: dict[str, object]) -> bytes:
        if type(message) is not dict or any(
            type(key) is not str for key in message
        ):
            raise TypeError(f"{self.label} frame must be an object")
        payload = json.dumps(
            message,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
        if not payload or len(payload) > self.max_bytes:
            raise ValueError(f"{self.label} frame exceeds its byte limit")
        return len(payload).to_bytes(4, "big") + payload

    def decode_header(self, header: bytes) -> int:
        if type(header) is not bytes:
            raise TypeError(f"{self.label} frame header must be exact bytes")
        if len(header) != 4:
            raise EOFError(f"{self.label} frame header ended early")
        length = int.from_bytes(header, "big")
        if length <= 0 or length > self.max_bytes:
            raise ValueError(f"{self.label} frame length is invalid")
        return length

    def decode_payload(self, payload: bytes) -> dict[str, object]:
        if type(payload) is not bytes:
            raise TypeError(f"{self.label} frame payload must be exact bytes")
        if not payload or len(payload) > self.max_bytes:
            raise ValueError(f"{self.label} frame payload length is invalid")
        try:
            value = json.loads(
                payload.decode("utf-8", errors="strict"),
                parse_constant=_reject_non_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise ValueError(f"{self.label} frame JSON is invalid") from None
        if type(value) is not dict or any(type(key) is not str for key in value):
            raise ValueError(f"{self.label} frame must be an object")
        return cast(dict[str, object], value)

    def decode(self, frame: bytes) -> dict[str, object]:
        if type(frame) is not bytes:
            raise TypeError(f"{self.label} frame must be exact bytes")
        if len(frame) < 4:
            raise EOFError(f"{self.label} frame header ended early")
        length = self.decode_header(frame[:4])
        expected = 4 + length
        if len(frame) < expected:
            raise EOFError(f"{self.label} frame payload ended early")
        if len(frame) > expected:
            raise ValueError(f"{self.label} frame contains trailing bytes")
        return self.decode_payload(frame[4:])


def _reject_non_json_constant(value: str) -> Never:
    raise ValueError(f"{value} is not valid JSON")


__all__: list[str] = []
