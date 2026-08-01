"""Retained Coding Agent invocation record."""

from __future__ import annotations

from pathlib import Path

from ...agents import AgentInvocation
from .._workspace import RunDirectoryPaths
from .writer import write_json_atomic, write_read_only_text_file

_INVOCATION_SCHEMA = "evopolicygym/agent-invocation/v1"
_SESSION_SOCKET_VARIABLE = "EVOPOLICYGYM_SESSION_SOCKET"
_WORKSPACE_VARIABLE = "EVOPOLICYGYM_WORKSPACE"


def retain_agent_invocation(
    paths: RunDirectoryPaths,
    invocation: AgentInvocation,
) -> None:
    if invocation.instructions is not None:
        write_read_only_text_file(
            paths.agent / "instructions.md",
            invocation.instructions,
            error_message="Agent instructions could not be retained",
        )
    _write_invocation(paths.agent / "invocation.json", invocation)


def _write_invocation(path: Path, invocation: AgentInvocation) -> None:
    write_json_atomic(
        path,
        {
            "schema": _INVOCATION_SCHEMA,
            "agent": dict(invocation.identity),
            "command": list(invocation.recorded_command),
            "cwd": "workspace",
            "environment": {
                "fixed_names": [
                    "PATH",
                    "PYTHONPATH",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONUNBUFFERED",
                    _SESSION_SOCKET_VARIABLE,
                    _WORKSPACE_VARIABLE,
                ],
                "inherited_allowlist": list(
                    invocation.inherited_environment
                ),
            },
            "instructions": (
                "agent/instructions.md"
                if invocation.instructions is not None
                else None
            ),
            "stdout": "agent/stdout.log",
            "stdout_media_type": invocation.stdout_media_type,
            "stderr": "agent/stderr.log",
        },
    )


__all__: list[str] = []
