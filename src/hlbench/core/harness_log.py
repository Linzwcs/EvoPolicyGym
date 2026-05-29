"""Plain-text harness lifecycle log → ``run_dir/logs/harness.log``.

One line per event, format per output.md §6.1::

    <ISO timestamp>Z LEVEL  event key1=val1 key2=val2 ...

Stdlib only — we deliberately do not use the ``logging`` module here:
every Server is per-run and the log file is uniquely owned, so a
direct file handle keeps things simple and avoids global-logger
interactions with consumer code.

What we LOG (high signal, low chatter):
- ``run_start`` — Server.__init__ done
- ``submit_received`` — entering Server.submit
- ``snapshot_taken`` — SubmitHandler Phase 2 done
- ``episode_start`` / ``episode_end`` — per-episode bracket inside Phase 6
- ``submit_completed`` — Server.submit returns
- ``finalize_start`` — Server.finalize entered
- ``run_end`` — Server.finalize returns

What we DON'T LOG:
- Held-out seed values, held-out per-episode returns
- Anything that could leak the spec's "Hidden held-out" invariant
- Trajectory contents (those go to feedback/, not logs)

Failures to write are non-fatal: harness.log is observability, never
load-bearing for correctness.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _format_kv(kv: dict[str, Any]) -> str:
    """Format a kv mapping as space-separated ``key=value`` pairs.

    Values containing whitespace are JSON-quoted to keep one event = one
    line. Booleans/None render as Python repr (True/False/None). Floats
    are formatted with ``g`` to avoid trailing zeros."""
    parts: list[str] = []
    for k, v in kv.items():
        if v is None:
            s = "None"
        elif isinstance(v, bool):
            s = "True" if v else "False"
        elif isinstance(v, float):
            s = f"{v:g}"
        elif isinstance(v, str):
            if any(c.isspace() for c in v) or '"' in v:
                # quote-and-escape
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                s = f'"{escaped}"'
            else:
                s = v
        else:
            s = str(v)
        parts.append(f"{k}={s}")
    return " ".join(parts)


class HarnessLog:
    """Append-only writer for harness.log. Cheap to construct;
    safe to ignore (a no-op writer is returned by ``disabled()`` for
    tests / lib usage that doesn't want a log file)."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure the file exists even if no events fire (so consumers
            # can always tail it).
            path.touch(exist_ok=True)

    @classmethod
    def disabled(cls) -> HarnessLog:
        """A no-op log writer (used by tests + lib mode without run dir)."""
        return cls(path=None)

    def event(self, name: str, level: str = "INFO", /, **kv: Any) -> None:
        """Append one event. Failures are swallowed — a log write going
        wrong should never break the run."""
        if self._path is None:
            return
        line = f"{_now_iso()} {level:5s} {name}"
        if kv:
            line += " " + _format_kv(kv)
        with (
            contextlib.suppress(OSError),
            self._path.open("a", encoding="utf-8") as f,
        ):
            f.write(line + "\n")

    @property
    def path(self) -> Path | None:
        return self._path
