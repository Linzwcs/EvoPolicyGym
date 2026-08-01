"""Agent-side framed transport primitives for the active Session client."""

from __future__ import annotations

import socket

from ..._protocol.session import SESSION_FRAMES


def send_session_message(
    connection: socket.socket,
    message: dict[str, object],
) -> None:
    connection.sendall(SESSION_FRAMES.encode(message))


def receive_session_message(connection: socket.socket) -> dict[str, object]:
    length = SESSION_FRAMES.decode_header(_receive_exact(connection, 4))
    return SESSION_FRAMES.decode_payload(_receive_exact(connection, length))


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Session frame ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__: list[str] = []
