"""Write feedback artifacts to ``workspace/feedback/submit_NNN/``.

Schemas live in:
- ``summary.json``                         → SPEC.md §4.1
- ``errors.txt`` (submit-level)            → SPEC.md §4.4.2
- ``episodes/ep_<XXX>/trajectory.jsonl``   → SPEC.md §4.2
- ``episodes/ep_<XXX>/stdout.txt``         → SPEC.md §4.5
- ``episodes/ep_<XXX>/stderr.txt``         → SPEC.md §4.5
- ``episodes/ep_<XXX>/error.txt``          → SPEC.md §4.4.3

Atomicity contract: ``summary.json`` is written via temp-file + rename, so
the agent never observes a partial file. Other files (trajectory, error,
stdout, stderr) are written before ``summary.json`` appears, so the
convention "if summary.json is there, the rest is too" holds end-to-end.

Error files (``errors.txt`` and per-episode ``error.txt``) accept multiple
appended events (SPEC §4.4.2/§4.4.3) and are capped at 64KB cumulative.
Once the cap is hit, subsequent events are dropped and a single
``category: "truncated"`` sentinel line is appended (SPEC §4.4.5).

Stream files (``stdout.txt`` / ``stderr.txt``) are written once per
episode and capped at 64KB with a ``... [truncated at 64KB] ...`` marker
line (SPEC §4.5).

MVP omissions (deferred to post-MVP):
- ``observations.npy`` (external obs storage)
- ``video.mp4``
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "0.1"

#: SPEC §4.4.5 — each error file capped at 64 KB cumulative across entries.
ERROR_FILE_CAP_BYTES = 64 * 1024

#: SPEC §4.5 — each per-episode stdout/stderr file capped at 64 KB.
STREAM_FILE_CAP_BYTES = 64 * 1024

#: Single-line marker we look for to decide "truncated sentinel already
#: written" without re-parsing JSON.
_TRUNCATED_MARKER = b'"category":"truncated"'

#: Appended to stdout.txt / stderr.txt when capture exceeds the cap
#: (SPEC §4.5 specifies this exact spelling for analyst grep).
_STREAM_TRUNCATED_MARKER = "\n... [truncated at 64KB] ...\n"


# ----------------------- directory / name helpers -------------------------


def dir_width(episode_budget: int) -> int:
    """Per SPEC.md §4.0: width = ``max(3, len(str(episode_budget)))``."""
    return max(3, len(str(episode_budget)))


def submit_dir_name(submit_index: int, width: int) -> str:
    return f"submit_{submit_index:0{width}d}"


def episode_dir_name(global_episode_index: int, width: int) -> str:
    return f"ep_{global_episode_index:0{width}d}"


def now_iso_utc() -> str:
    """ISO-8601 UTC with millisecond precision (matches SPEC examples)."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# ----------------------- JSON encoding ------------------------------------


def _jsonify_floats(value: Any) -> Any:
    """Encode NaN/Inf as strings per SPEC.md §4.2.

    Walks dict/list leaves and replaces NaN→``"NaN"``, +Inf→``"Inf"``,
    -Inf→``"-Inf"``. Leaves other types untouched (callers are expected to
    have already converted numpy → Python scalars; see env_runner.py).
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, dict):
        return {k: _jsonify_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_floats(v) for v in value]
    return value


def _dumps(value: Any, *, indent: int | None = None) -> str:
    """JSON dump with NaN/Inf handled per SPEC §4.2; raises on bare NaN/Inf
    that slipped through ``_jsonify_floats`` (defensive)."""
    return json.dumps(
        _jsonify_floats(value),
        indent=indent,
        separators=(",", ":") if indent is None else (", ", ": "),
        allow_nan=False,
    )


# ----------------------- atomic write -------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Temp-file + ``os.replace`` so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ----------------------- public writers -----------------------------------


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Atomic write of ``summary.json`` (SPEC §4.1)."""
    _atomic_write_text(path, _dumps(summary, indent=2) + "\n")


def write_trajectory(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    """Write ``trajectory.jsonl`` (SPEC §4.2). One JSON object per line."""
    lines = [_dumps(e) for e in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _error_event(
    *,
    category: str,
    message: str,
    traceback_str: str | None,
    step_index: int | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": now_iso_utc(),
        "category": category,
        "message": message,
        "traceback": traceback_str,
    }
    if step_index is not None:
        event["step_index"] = step_index
    return event


def _append_error_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSONL line to ``path``, enforcing the SPEC §4.4.5 cap.

    Semantics:
    - First write: always succeeds (the failure that produced the event is
      important enough to record in full even if it alone exceeds the cap).
    - Subsequent writes: appended if cumulative size stays ≤ 64 KB.
      Otherwise dropped, and a single ``category: "truncated"`` sentinel
      is appended (only once per file).
    """
    line = (_dumps(event) + "\n").encode("utf-8")
    if not path.exists():
        path.write_bytes(line)
        return
    current = path.read_bytes()
    if _TRUNCATED_MARKER in current:
        # Already capped; drop silently — sentinel is already on disk.
        return
    if len(current) + len(line) <= ERROR_FILE_CAP_BYTES:
        with path.open("ab") as f:
            f.write(line)
        return
    # Would overflow; write sentinel and drop this event.
    sentinel = _error_event(
        category="truncated",
        message="additional events omitted",
        traceback_str=None,
    )
    sentinel_line = (_dumps(sentinel) + "\n").encode("utf-8")
    with path.open("ab") as f:
        f.write(sentinel_line)


def write_submit_error(
    path: Path,
    *,
    category: str,
    message: str,
    traceback_str: str | None = None,
) -> None:
    """Append one event to submit-level ``errors.txt`` (SPEC §4.4.2).

    Multiple entries are allowed (oom / submit_wall_exceeded may coexist
    with successful episodes per SPEC §3.3). Capped at 64 KB per §4.4.5.
    """
    event = _error_event(
        category=category, message=message, traceback_str=traceback_str
    )
    _append_error_event(path, event)


def write_episode_error(
    path: Path,
    *,
    category: str,
    message: str,
    step_index: int | None,
    traceback_str: str | None,
) -> None:
    """Append one event to per-episode ``error.txt`` (SPEC §4.4.3).

    Multiple entries are allowed (e.g. a non-fatal warning followed by the
    fatal error). Capped at 64 KB per §4.4.5.
    """
    event = _error_event(
        category=category,
        message=message,
        traceback_str=traceback_str,
        step_index=step_index,
    )
    _append_error_event(path, event)


def write_episode_stream(path: Path, text: str) -> None:
    """Write captured ``stdout.txt`` / ``stderr.txt`` for one episode
    (SPEC §4.5).

    Always creates the file (may be zero-byte if the policy printed
    nothing). Truncates at ``STREAM_FILE_CAP_BYTES`` (64 KB) with a
    final ``... [truncated at 64KB] ...`` marker line appended.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= STREAM_FILE_CAP_BYTES:
        path.write_bytes(encoded)
        return
    marker = _STREAM_TRUNCATED_MARKER.encode("utf-8")
    keep = STREAM_FILE_CAP_BYTES - len(marker)
    truncated = encoded[:keep]
    # Defensive: a hard byte cut may land in the middle of a multi-byte
    # UTF-8 char. Back off to the last valid boundary.
    while truncated:
        try:
            truncated.decode("utf-8")
            break
        except UnicodeDecodeError as e:
            truncated = truncated[: e.start]
    path.write_bytes(truncated + marker)
