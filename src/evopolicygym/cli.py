"""Agent-facing command-line presentation for an active local Session."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import cast

from ._protocol.session import SESSION_PROTOCOL
from ._version import __version__
from .run._socket import (
    receive_session_message,
    send_session_message,
)

_SESSION_SOCKET_VARIABLE = "EVOPOLICYGYM_SESSION_SOCKET"
_WORKSPACE_VARIABLE = "EVOPOLICYGYM_WORKSPACE"


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
    parser = argparse.ArgumentParser(prog="evopolicygym")
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
    submit.add_argument("--episodes", type=int, default=1)

    finish = subcommands.add_parser(
        "finish",
        help="select one previously published submission",
    )
    finish.add_argument("submission_id")
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
            "episodes": cast(int, namespace.episodes),
        }
    if namespace.command == "finish":
        return {
            "protocol": SESSION_PROTOCOL,
            "method": "finish",
            "submission_id": cast(str, namespace.submission_id),
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


if __name__ == "__main__":
    raise SystemExit(main())
