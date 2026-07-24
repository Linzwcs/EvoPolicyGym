"""Non-authoritative observation of one active Program-Evolution Run."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Literal, Protocol, TextIO, runtime_checkable

type RunEventValue = str | int | float | bool | None
type ProgressMode = Literal["auto", "plain"]

_RESERVED_FIELDS = frozenset(
    {"schema", "time_unix_ns", "monotonic_ns", "event"}
)


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One immutable Host-side lifecycle event delivered after persistence."""

    name: str
    time_unix_ns: int
    monotonic_ns: int
    fields: Mapping[str, RunEventValue]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("RunEvent name must be non-empty text")
        for name in ("time_unix_ns", "monotonic_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"RunEvent {name} must be a non-negative integer")

        fields: dict[str, RunEventValue] = {}
        for key, value in self.fields.items():
            if type(key) is not str or not key:
                raise ValueError("RunEvent field names must be non-empty text")
            if key in _RESERVED_FIELDS:
                raise ValueError(f"RunEvent field name is reserved: {key}")
            if type(value) not in {str, int, float, bool, type(None)}:
                raise TypeError("RunEvent fields must contain JSON scalar values")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("RunEvent float fields must be finite")
            fields[key] = value
        object.__setattr__(self, "fields", MappingProxyType(fields))


@runtime_checkable
class RunObserver(Protocol):
    """Receive persisted Run events without participating in Run semantics."""

    def on_event(self, event: RunEvent, /) -> None:
        ...


class ConsoleProgress:
    """Render concise Run progress to a Host-owned text stream."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        mode: ProgressMode = "auto",
    ) -> None:
        if mode not in {"auto", "plain"}:
            raise ValueError("progress mode must be 'auto' or 'plain'")
        self._stream = sys.stderr if stream is None else stream
        self._interactive = mode == "auto" and self._is_terminal()
        self._evaluation_started: dict[str, int] = {}
        self._active_line = False
        self._lock = Lock()

    def on_event(self, event: RunEvent, /) -> None:
        """Render one recognized event and ignore unknown future events."""

        if type(event) is not RunEvent:
            raise TypeError("event must be RunEvent")
        with self._lock:
            rendered = self._render(event)
            if rendered is None:
                return
            message, transient = rendered
            if self._interactive:
                self._render_interactive(message, transient=transient)
            else:
                self._stream.write(message + "\n")
                self._stream.flush()

    def _render(self, event: RunEvent) -> tuple[str, bool] | None:
        fields = event.fields
        if event.name == "agent_started":
            benchmark = _text(fields, "benchmark_id")
            return (f"Agent started · {benchmark}", False)

        if event.name == "evaluation_started":
            submission = _text(fields, "submission_id")
            episodes = _integer(fields, "episodes")
            remaining = _integer(fields, "episodes_remaining")
            self._evaluation_started[submission] = event.monotonic_ns
            return (
                f"{submission} · evaluating {episodes} Episodes"
                f" · budget {remaining} remaining",
                True,
            )

        if event.name == "episode_completed":
            submission = _text(fields, "submission_id")
            completed = _integer(fields, "completed")
            total = _integer(fields, "total")
            status = _text(fields, "status")
            started = self._evaluation_started.get(submission)
            elapsed = (
                ""
                if started is None
                else f" · {(event.monotonic_ns - started) / 1_000_000_000:.1f}s"
            )
            return (
                f"{submission} · Episodes {completed}/{total}"
                f"{elapsed} · {status}",
                True,
            )

        if event.name == "submission_published":
            submission = _text(fields, "submission_id")
            score = _number(fields, "score")
            remaining = _integer(fields, "episodes_remaining")
            self._evaluation_started.pop(submission, None)
            return (
                f"{submission} · score {score:g}"
                f" · budget {remaining} remaining",
                False,
            )

        if event.name in {"evaluation_failed", "publication_failed"}:
            submission = _text(fields, "submission_id")
            self._evaluation_started.pop(submission, None)
            label = event.name.replace("_", " ")
            return (f"{submission} · {label}", False)

        if event.name == "run_finished":
            submission = _text(fields, "submission_id")
            return (f"Run finished · selected {submission}", False)

        if event.name == "agent_timeout":
            return ("Agent timed out", False)

        if event.name == "agent_start_failed":
            return ("Agent failed to start", False)

        if event.name == "agent_stopped_after_terminal":
            return ("Agent stopped after terminal Run state", False)

        if event.name == "agent_exited":
            return (
                f"Agent exited · return code {fields.get('returncode')}",
                False,
            )

        return None

    def _render_interactive(self, message: str, *, transient: bool) -> None:
        if transient:
            self._stream.write("\r\x1b[2K" + message)
            self._stream.flush()
            self._active_line = True
            return
        if self._active_line:
            self._stream.write("\r\x1b[2K")
            self._active_line = False
        self._stream.write(message + "\n")
        self._stream.flush()

    def _is_terminal(self) -> bool:
        try:
            return self._stream.isatty()
        except OSError:
            return False


def _text(fields: Mapping[str, RunEventValue], name: str) -> str:
    value = fields.get(name)
    return value if type(value) is str else "unknown"


def _integer(fields: Mapping[str, RunEventValue], name: str) -> int:
    value = fields.get(name)
    return value if type(value) is int else 0


def _number(fields: Mapping[str, RunEventValue], name: str) -> float:
    value = fields.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


__all__ = [
    "ConsoleProgress",
    "ProgressMode",
    "RunEvent",
    "RunEventValue",
    "RunObserver",
]
