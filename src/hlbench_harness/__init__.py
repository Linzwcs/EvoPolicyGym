"""hlbench-pro automated evaluation harness.

A consumer package that drives a coding-agent CLI session through one
init → submit → finalize loop on a registered hlbench env, preserving
the agent's conversation across iterations via session resume.

Three backends are bundled:

- :class:`hlbench_harness.claude_agent.ClaudeAgent` — Claude Code
  (``claude --print --resume <uuid>``).
- :class:`hlbench_harness.codex_agent.CodexAgent` — OpenAI Codex CLI
  (``codex exec`` then ``codex exec resume <session-id>``).
- :class:`hlbench_harness.kimi_agent.KimiAgent` — Moonshot Kimi Code
  CLI (``kimi -p`` then ``kimi -S <session-id> -p``).

Lives outside ``src/hlbench/`` per the lib/consumer separation rule —
this module imports ``hlbench.core.Server`` but the lib does not depend
on it.
"""

from hlbench_harness.claude_agent import (
    ClaudeAgent,
    ClaudeAgentConfig,
    find_claude_binary,
)
from hlbench_harness.codex_agent import (
    CodexAgent,
    CodexAgentConfig,
    find_codex_binary,
)
from hlbench_harness.kimi_agent import (
    KimiAgent,
    KimiAgentConfig,
    find_kimi_binary,
)

__version__ = "0.1.0"

__all__ = [
    "ClaudeAgent",
    "ClaudeAgentConfig",
    "CodexAgent",
    "CodexAgentConfig",
    "KimiAgent",
    "KimiAgentConfig",
    "find_claude_binary",
    "find_codex_binary",
    "find_kimi_binary",
]
