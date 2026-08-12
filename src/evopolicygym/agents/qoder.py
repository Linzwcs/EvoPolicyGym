"""Public caller-owned Qoder Agent selection."""

from __future__ import annotations

from dataclasses import dataclass

from .base import AgentInvocation, AgentTask, resolve_executable

_QODER_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "QODER_CONFIG_DIR",
    "QODER_PERSONAL_ACCESS_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "USER",
    "LOGNAME",
    "SHELL",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Qoder:
    """Select the Qoder model, optional reasoning effort, and CLI used by a Run."""

    model: str
    reasoning_effort: str | None = None
    executable: str = "qodercli"

    def __post_init__(self) -> None:
        _validate_identifier(self.model, name="model", max_bytes=128)
        if self.reasoning_effort is not None:
            _validate_identifier(
                self.reasoning_effort,
                name="reasoning_effort",
                max_bytes=64,
            )
        _validate_executable(self.executable)

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        """Translate one Host-authored task into a Qoder CLI invocation."""

        if type(task) is not AgentTask:
            raise TypeError("task must be AgentTask")
        resolved_executable = resolve_executable(self.executable)
        command_prefix = (
            resolved_executable,
            "--print",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            *(
                ("--reasoning-effort", self.reasoning_effort)
                if self.reasoning_effort is not None
                else ()
            ),
            "--permission-mode",
            "bypass_permissions",
        )
        identity = {
            "provider": "qoder",
            "model": self.model,
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort is not None
                else {}
            ),
        }
        return AgentInvocation(
            command=(*command_prefix, task.instructions),
            recorded_command=(*command_prefix, "@agent/instructions.md"),
            identity=identity,
            instructions=task.instructions,
            inherited_environment=_QODER_ENVIRONMENT_ALLOWLIST,
            stdout_media_type="application/x-ndjson",
        )


def _validate_identifier(value: str, *, name: str, max_bytes: int) -> None:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8", errors="strict")) > max_bytes
        or any(character.isspace() for character in value)
        or "\0" in value
    ):
        raise ValueError(f"{name} must be a non-empty bounded identifier")


def _validate_executable(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8", errors="strict")) > 4_096
        or "\0" in value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError("executable must be a bounded command or path")


__all__ = ["Qoder"]
