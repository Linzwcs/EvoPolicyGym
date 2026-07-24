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
from .codex import Codex

__all__ = [
    "AgentInvocation",
    "AgentTask",
    "CodingAgent",
    "Codex",
    "command_invocation",
    "resolve_executable",
]
