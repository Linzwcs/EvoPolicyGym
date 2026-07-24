"""Public template for command-line Coding Agent integrations."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..errors import AgentRunError


@dataclass(frozen=True, slots=True)
class AgentTask:
    """One Host-authored task delivered unchanged to a Coding Agent."""

    instructions: str = field(repr=False)

    def __post_init__(self) -> None:
        instructions = _instructions(self.instructions)
        assert instructions is not None
        object.__setattr__(self, "instructions", instructions)


@dataclass(frozen=True, slots=True)
class AgentInvocation:
    """Validated command invocation returned by an Agent integration."""

    command: tuple[str, ...]
    recorded_command: tuple[str, ...]
    identity: Mapping[str, str]
    instructions: str | None = field(default=None, repr=False)
    inherited_environment: tuple[str, ...] = ()
    stdout_media_type: str = "text/plain"

    def __post_init__(self) -> None:
        command = _command(self.command, name="command")
        recorded = _command(self.recorded_command, name="recorded_command")
        identity = _identity(self.identity)
        instructions = _instructions(self.instructions)
        environment = _environment_names(self.inherited_environment)
        media_type = _media_type(self.stdout_media_type)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "recorded_command", recorded)
        object.__setattr__(self, "identity", MappingProxyType(identity))
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "inherited_environment", environment)
        object.__setattr__(self, "stdout_media_type", media_type)


@runtime_checkable
class CodingAgent(Protocol):
    """Structural template implemented by Coding Agent providers."""

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        """Translate one Host task into a validated command invocation."""
        ...


def command_invocation(
    command: Sequence[str],
    *,
    recorded_command: Sequence[str] | None = None,
    identity: Mapping[str, str] | None = None,
    instructions: str | None = None,
    inherited_environment: Sequence[str] = (),
    stdout_media_type: str = "text/plain",
) -> AgentInvocation:
    """Build a validated invocation for a simple command-based integration."""

    selected_command = _command(command, name="command")
    return AgentInvocation(
        command=selected_command,
        recorded_command=(
            selected_command
            if recorded_command is None
            else _command(recorded_command, name="recorded_command")
        ),
        identity={"provider": "command"} if identity is None else identity,
        instructions=instructions,
        inherited_environment=tuple(inherited_environment),
        stdout_media_type=stdout_media_type,
    )


def resolve_executable(value: str) -> str:
    """Resolve one command name or path to an executable absolute path."""

    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8", errors="strict")) > 4_096
        or "\0" in value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError("executable must be a bounded command or path")
    has_separator = os.sep in value or (
        os.altsep is not None and os.altsep in value
    )
    if has_separator:
        candidate = Path(value)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise AgentRunError("Agent executable does not exist") from None
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise AgentRunError("Agent executable is not executable")
        return str(resolved)
    found = shutil.which(value)
    if found is None:
        raise AgentRunError("Agent executable was not found on PATH")
    return str(Path(found).resolve())


def _command(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(type(item) is not str or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty sequence of text arguments")
    return tuple(value)


def _identity(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("identity must be a mapping")
    identity: dict[str, str] = {}
    for key, item in value.items():
        if (
            type(key) is not str
            or not key
            or type(item) is not str
            or not item
        ):
            raise ValueError("identity must contain non-empty text")
        identity[key] = item
    if "provider" not in identity:
        raise ValueError("identity requires provider")
    return identity


def _environment_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("inherited_environment must be a sequence")
    names = tuple(value)
    if any(
        type(name) is not str
        or not name
        or "=" in name
        or "\0" in name
        for name in names
    ):
        raise ValueError("inherited_environment contains an invalid name")
    return tuple(dict.fromkeys(names))


def _instructions(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("instructions must be text or None")
    if not value or len(value.encode("utf-8", errors="strict")) > 256 * 1024:
        raise ValueError("instructions must be non-empty and bounded")
    return value


def _media_type(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8", errors="strict")) > 256
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(
            "stdout_media_type must be non-empty bounded single-line text"
        )
    return value


__all__ = [
    "AgentInvocation",
    "AgentTask",
    "CodingAgent",
    "command_invocation",
    "resolve_executable",
]
