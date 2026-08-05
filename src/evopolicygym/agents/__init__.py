"""Public Coding Agent integration template and first-party providers.

The Host owns the task, a provider translates it into a validated invocation,
and generic process execution remains provider-neutral.
"""

from .base import (
    AgentInvocation,
    AgentTask,
    CodingAgent,
    command_invocation,
    resolve_executable,
)
from .claude_code import ClaudeCode
from .codex import Codex
from .kimi_code import KimiCode

__all__ = [
    "AgentInvocation",
    "AgentTask",
    "CodingAgent",
    "ClaudeCode",
    "Codex",
    "KimiCode",
    "command_invocation",
    "resolve_executable",
]
