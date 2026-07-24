"""Public caller-owned Codex Agent selection."""

from __future__ import annotations

from dataclasses import dataclass

from .base import AgentInvocation, AgentTask, resolve_executable

_CODEX_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "CODEX_HOME",
    "CODEX_API_KEY",
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
class Codex:
    """Select the Codex model and CLI executable used by a Run."""

    model: str
    executable: str = "codex"

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or len(self.model.encode("utf-8", errors="strict")) > 128
            or any(character.isspace() for character in self.model)
            or "\0" in self.model
        ):
            raise ValueError("model must be a non-empty bounded identifier")
        if (
            type(self.executable) is not str
            or not self.executable
            or len(self.executable.encode("utf-8", errors="strict")) > 4_096
            or "\0" in self.executable
            or "\r" in self.executable
            or "\n" in self.executable
        ):
            raise ValueError("executable must be a bounded command or path")

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        """Translate one Host-authored task into a Codex CLI invocation."""

        if type(task) is not AgentTask:
            raise TypeError("task must be AgentTask")
        resolved_executable = resolve_executable(self.executable)
        command_prefix = (
            resolved_executable,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--json",
            "--model",
            self.model,
            "--sandbox",
            "danger-full-access",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--color",
            "never",
        )
        return AgentInvocation(
            command=(*command_prefix, task.instructions),
            recorded_command=(*command_prefix, "@agent/instructions.md"),
            identity={
                "provider": "codex",
                "model": self.model,
            },
            instructions=task.instructions,
            inherited_environment=_CODEX_ENVIRONMENT_ALLOWLIST,
            stdout_media_type="application/x-ndjson",
        )


__all__ = ["Codex"]
