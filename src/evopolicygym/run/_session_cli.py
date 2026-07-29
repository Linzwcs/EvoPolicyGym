"""Agent-facing command-line presentation for an active local Session."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import cast

from .._protocol.session import (
    SESSION_MAX_EPISODE_INDICES,
    SESSION_PROTOCOL,
)
from .._version import __version__
from ._socket import (
    receive_session_message,
    send_session_message,
)

_SESSION_SOCKET_VARIABLE = "EVOPOLICYGYM_SESSION_SOCKET"
_WORKSPACE_VARIABLE = "EVOPOLICYGYM_WORKSPACE"
_UNSIGNED_DECIMAL = re.compile(r"[0-9]+")
_MAX_EPISODE_INDEX = 2**64 - 1


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    try:
        request = _request(namespace)
        response = _call_session(request)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "client_error",
                        "message": str(error),
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    serialized = json.dumps(
        response,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if response.get("ok") is True:
        print(serialized)
        return 0
    print(serialized, file=sys.stderr)
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evopolicygym-session")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    submit = subcommands.add_parser(
        "submit",
        help="evaluate the fixed workspace Policy Program directory",
    )
    submit.add_argument(
        "program",
        help="must resolve to $EVOPOLICYGYM_WORKSPACE/program",
    )
    submit.add_argument(
        "--episodes",
        type=_parse_episode_selector,
        required=True,
        metavar="SELECTOR",
        help='Run-local Episode indices, for example "0:2,4:8"',
    )

    finish = subcommands.add_parser(
        "finish",
        help="finish search with one or more published candidates",
    )
    finish.add_argument("submission_ids", nargs="+")
    return parser


def _request(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.command == "submit":
        workspace = _required_path(_WORKSPACE_VARIABLE)
        expected = (workspace / "program").resolve(strict=True)
        supplied = Path(cast(str, namespace.program)).resolve(strict=True)
        if supplied != expected:
            raise ValueError("submitted Program must be workspace/program")
        return {
            "protocol": SESSION_PROTOCOL,
            "method": "submit",
            "episode_indices": list(
                cast(tuple[int, ...], namespace.episodes)
            ),
        }
    if namespace.command == "finish":
        return {
            "protocol": SESSION_PROTOCOL,
            "method": "finish",
            "submission_ids": cast(list[str], namespace.submission_ids),
        }
    raise RuntimeError("unknown command")


def _call_session(request: dict[str, object]) -> dict[str, object]:
    socket_path = _required_path(_SESSION_SOCKET_VARIABLE)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(socket_path))
        send_session_message(connection, request)
        response = receive_session_message(connection)
        if (
            response.get("protocol") != SESSION_PROTOCOL
            or type(response.get("ok")) is not bool
        ):
            raise ValueError("Session returned an invalid protocol response")
        return response
    finally:
        connection.close()


def _required_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} is not set; no Agent Session is active")
    return Path(raw)


def _parse_episode_selector(value: str) -> tuple[int, ...]:
    if not value or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError(
            "Episode selector must not be empty or contain whitespace"
        )
    indices: list[int] = []
    for item in value.split(","):
        if not item:
            raise argparse.ArgumentTypeError(
                "Episode selector contains an empty item"
            )
        parts = item.split(":")
        if len(parts) == 1:
            index = _parse_unsigned(parts[0], endpoint=False)
            _append_index(indices, index)
            continue
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "Episode ranges must use START:END"
            )
        start = _parse_unsigned(parts[0], endpoint=False)
        end = _parse_unsigned(parts[1], endpoint=True)
        if start >= end:
            raise argparse.ArgumentTypeError(
                "Episode ranges must be non-empty and increasing"
            )
        count = end - start
        if count > SESSION_MAX_EPISODE_INDICES - len(indices):
            raise argparse.ArgumentTypeError(
                "Episode selector contains too many indices"
            )
        if indices and start <= indices[-1]:
            raise argparse.ArgumentTypeError(
                "Episode selector must be strictly increasing without overlap"
            )
        indices.extend(range(start, end))
    if len(indices) > SESSION_MAX_EPISODE_INDICES:
        raise argparse.ArgumentTypeError(
            "Episode selector contains too many indices"
        )
    return tuple(indices)


def _parse_unsigned(value: str, *, endpoint: bool) -> int:
    if _UNSIGNED_DECIMAL.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "Episode indices must use unsigned decimal integers"
        )
    parsed = int(value)
    limit = 2**64 if endpoint else _MAX_EPISODE_INDEX
    if parsed > limit:
        raise argparse.ArgumentTypeError(
            "Episode index exceeds the unsigned 64-bit range"
        )
    return parsed


def _append_index(indices: list[int], index: int) -> None:
    if len(indices) == SESSION_MAX_EPISODE_INDICES:
        raise argparse.ArgumentTypeError(
            "Episode selector contains too many indices"
        )
    if indices and index <= indices[-1]:
        raise argparse.ArgumentTypeError(
            "Episode selector must be strictly increasing without overlap"
        )
    indices.append(index)


if __name__ == "__main__":
    raise SystemExit(main())
