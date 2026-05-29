"""hlbench-pro automated evaluation harness.

A consumer package that drives a Claude Code session through a complete
init → submit → finalize loop on a registered hlbench env, preserving
the agent's conversation context across iterations via
``claude --print --resume <session_id>``.

Lives outside ``src/hlbench/`` per the lib/consumer separation rule —
this module imports ``hlbench.core.Server`` but the lib does not depend
on it.
"""

__version__ = "0.1.0"
