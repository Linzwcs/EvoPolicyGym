"""Public caller-owned Claude Code Agent selection."""

from __future__ import annotations

from dataclasses import dataclass

from .base import AgentInvocation, AgentTask, resolve_executable

_CLAUDE_CODE_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_TMPDIR",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
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
class ClaudeCode:
    """Select the Claude Code model, effort, and CLI used by a Run."""

    model: str
    effort: str
    executable: str = "claude"

    def __post_init__(self) -> None:
        _validate_identifier(self.model, name="model", max_bytes=128)
        _validate_identifier(self.effort, name="effort", max_bytes=64)
        _validate_executable(self.executable)

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        """Translate one Host-authored task into a Claude Code invocation."""

        if type(task) is not AgentTask:
            raise TypeError("task must be AgentTask")
        resolved_executable = resolve_executable(self.executable)
        command_prefix = (
            resolved_executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--permission-mode",
            "bypassPermissions",
            "--no-session-persistence",
            "--no-chrome",
            "--bare",
            "--strict-mcp-config",
        )
        return AgentInvocation(
            command=(*command_prefix, task.instructions),
            recorded_command=(*command_prefix, "@agent/instructions.md"),
            identity={
                "provider": "claude-code",
                "model": self.model,
                "effort": self.effort,
            },
            instructions=task.instructions,
            inherited_environment=_CLAUDE_CODE_ENVIRONMENT_ALLOWLIST,
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


__all__ = ["ClaudeCode"]
