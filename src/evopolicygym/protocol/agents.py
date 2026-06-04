"""Agent-facing workspace rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

_SOURCE = Path(__file__).with_name("AGENTS.md")
_FALLBACK = """# EvoPolicyGym Agent Rules

Implement `system/policy.py`, read `feedback/`, and submit through the server
API. Do not modify feedback, checkpoints, logs, hidden data, or `AGENTS.md`.
Do not run local environment rollouts or copied environment simulators; all
rollout data must come from `/submit` and prior `feedback/`.
Do not call `/finalize`.
"""


def body() -> str:
    """Return the packaged AGENTS.md body."""

    if _SOURCE.exists():
        return _SOURCE.read_text(encoding="utf-8")
    return _FALLBACK


def digest(text: str) -> str:
    """Return the protocol hash for an AGENTS.md body."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage(path: str | Path) -> str:
    """Write AGENTS.md to `path` and return its SHA-256 version string."""

    target = Path(path)
    text = body()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return digest(text)
