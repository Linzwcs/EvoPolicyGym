"""Write feedback artifacts to ``workspace/feedback/submit_NNN/``.

Schemas live in:
- ``summary.json``                         → SPEC.md §4.1
- ``errors.txt`` (submit-level)            → SPEC.md §4.4.2
- ``episodes/ep_<XXX>/trajectory.jsonl``   → SPEC.md §4.2
- ``episodes/ep_<XXX>/error.txt``          → SPEC.md §4.4.3

Atomicity contract: ``summary.json`` is written via temp-file + rename, so
the agent never observes a partial file. Other files (trajectory, error)
are written before ``summary.json`` appears, so the convention "if
summary.json is there, the rest is too" holds end-to-end.

MVP omissions (deferred to post-MVP):
- ``observations.npy`` (external obs storage)
- ``video.mp4``
- ``stdout.txt`` / ``stderr.txt`` capture
- 64 KB error-file truncation cap
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


def write_submit_error(
    path: Path,
    *,
    category: str,
    message: str,
    traceback_str: str | None = None,
) -> None:
    """Write submit-level ``errors.txt`` (SPEC §4.4.2). One JSON line."""
    event = _error_event(
        category=category, message=message, traceback_str=traceback_str
    )
    path.write_text(_dumps(event) + "\n")


def write_episode_error(
    path: Path,
    *,
    category: str,
    message: str,
    step_index: int | None,
    traceback_str: str | None,
) -> None:
    """Write per-episode ``error.txt`` (SPEC §4.4.3). One JSON line for MVP
    (the spec allows multiple, but env_runner produces at most one
    fatal event per episode)."""
    event = _error_event(
        category=category,
        message=message,
        traceback_str=traceback_str,
        step_index=step_index,
    )
    path.write_text(_dumps(event) + "\n")
