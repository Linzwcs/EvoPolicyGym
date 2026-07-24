"""Binary-stream I/O for complete Policy process frames."""

from __future__ import annotations

from typing import BinaryIO

from ...._protocol.policy import POLICY_FRAMES


def write_policy_message(
    stream: BinaryIO,
    message: dict[str, object],
) -> None:
    stream.write(POLICY_FRAMES.encode(message))
    stream.flush()


def read_policy_message(stream: BinaryIO) -> dict[str, object]:
    length = POLICY_FRAMES.decode_header(_read_exact(stream, 4))
    return POLICY_FRAMES.decode_payload(_read_exact(stream, length))


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Policy frame ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__: list[str] = []
