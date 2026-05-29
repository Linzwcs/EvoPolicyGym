"""Operator-side ``agent.jsonl`` writer.

Per ``docs/output.md §6.2``: the harness running the agent (i.e.,
``hlbench_harness`` — we are the operator) writes
``<run_dir>/logs/agent.jsonl`` with one JSON object per line capturing
agent-harness activity (run start/end, per-turn completions with token
+ cost metrics, etc.).

Schema (one event per line):

    {"t": "<ISO8601>", "event": "<name>", ...event-specific fields}

Event types emitted by ``hlbench_harness``:

    agent_start  — run begins; model + session_id
    completion   — one turn finished; cost_usd + tokens + latency
    agent_end    — run ends; termination_reason

Tool-call events (``output.md §6.2`` shows them as an example) are NOT
emitted by ``hlbench_harness``: the inner ``claude --print`` runs all
tool calls inside its own subprocess and we don't see them as discrete
events. Operators with deeper instrumentation can extend the writer
themselves; the file format is append-only.

Stdlib only — no logging module. Failures to write are non-fatal;
the run continues if disk is full / read-only / etc.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Match ``HarnessLog`` (output.md §6.1): millisecond-precision UTC.
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S."


def _now_iso() -> str:
    now = datetime.now(UTC)
    return now.strftime(_ISO_FORMAT) + f"{now.microsecond // 1000:03d}Z"


class AgentLog:
    """Append-only JSON-lines writer for ``runs/<...>/logs/agent.jsonl``.

    Cheap to construct; safe to ignore (``disabled()`` returns a no-op
    instance for tests / lib usage that doesn't want a log file)."""

    SCHEMA_VERSION = "0.1"

    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Touch so readers can tail even before the first event.
            path.touch(exist_ok=True)

    @classmethod
    def disabled(cls) -> AgentLog:
        return cls(path=None)

    @property
    def path(self) -> Path | None:
        return self._path

    def event(self, name: str, /, **fields: Any) -> None:
        """Append one JSON event. ``t`` is auto-stamped; pass any other
        SPEC §6.2 fields as kwargs.

        Failures to write are swallowed — observability shouldn't be
        load-bearing for correctness."""
        if self._path is None:
            return
        payload: dict[str, Any] = {"t": _now_iso(), "event": name}
        # SPEC §6.2 doesn't require schema_version on every line, but
        # ``HarnessLog`` does it elsewhere; mirror that for parity.
        payload["schema_version"] = self.SCHEMA_VERSION
        # Drop None values so the JSONL stays compact.
        for k, v in fields.items():
            if v is not None:
                payload[k] = v
        line = json.dumps(payload, default=str) + "\n"
        with (
            contextlib.suppress(OSError),
            self._path.open("a", encoding="utf-8") as f,
        ):
            f.write(line)

    # -------- typed helpers (so callers don't have to remember field names)

    def agent_start(self, *, model: str, session_id: str, **extra: Any) -> None:
        """Emit ``{"event": "agent_start", "model": ..., "session_id": ...}``."""
        self.event("agent_start", model=model, session_id=session_id, **extra)

    def completion(
        self,
        *,
        turn_index: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        **extra: Any,
    ) -> None:
        """Emit ``{"event": "completion", "turn_index": ..., ...}``.

        Mirrors the SPEC §6.2 example. Token / cost fields are optional
        because test stubs and aborted turns may not surface them."""
        self.event(
            "completion",
            turn_index=turn_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            **extra,
        )

    def agent_end(self, *, reason: str, **extra: Any) -> None:
        """Emit ``{"event": "agent_end", "reason": ..., ...}``."""
        self.event("agent_end", reason=reason, **extra)


__all__ = ["AgentLog"]
