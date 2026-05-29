"""Per-turn state snapshot helpers.

A ``TurnObservation`` is everything the harness shows the agent at the
start of each turn: the run's current ``GET /info`` body, a list of
all ``submit_NNN/summary.json`` documents the agent has produced so
far, and a derived ``progress`` block (remaining budget, last verdict,
mean return) that's cheap to scan in the continuation prompt.

Pure functions; no I/O against the HTTP layer — we drive
``Server.info()`` directly because the harness owns the Server instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurnObservation:
    """Snapshot of run state at the start of one harness turn."""

    turn_index: int
    info: dict[str, Any]
    submit_summaries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        v = self.info["state"]["remaining_budget"]
        assert isinstance(v, int)
        return v

    @property
    def is_finalized(self) -> bool:
        v = self.info["state"]["is_finalized"]
        assert isinstance(v, bool)
        return v

    @property
    def last_submit(self) -> dict[str, Any] | None:
        return self.submit_summaries[-1] if self.submit_summaries else None

    def progress_line(self) -> str:
        """One-line scannable status for the continuation prompt."""
        info_state = self.info["state"]
        n_submits = info_state["n_submits"]
        n_ok = info_state["n_successful_submits"]
        last = self.last_submit
        if last is None:
            return (
                f"turn={self.turn_index} "
                f"remaining_budget={self.remaining_budget} "
                f"submits=0 last=none"
            )
        last_status = last.get("status", "?")
        mean = last.get("mean_return")
        mean_str = "n/a" if mean is None else f"{mean:.2f}"
        return (
            f"turn={self.turn_index} "
            f"remaining_budget={self.remaining_budget} "
            f"submits={n_submits} ({n_ok} ok) "
            f"last=#{last['submit_index']}:{last_status} "
            f"mean_return={mean_str}"
        )


def observe(*, turn_index: int, info: dict[str, Any], workspace: Path) -> TurnObservation:
    """Build a ``TurnObservation`` from the live Server info + on-disk
    feedback summaries.

    The feedback dir is the source of truth for per-submit data because
    the Server in-memory ``submit_history`` only stores aggregates."""
    summaries = _load_submit_summaries(workspace / "feedback")
    return TurnObservation(
        turn_index=turn_index,
        info=info,
        submit_summaries=summaries,
    )


def _load_submit_summaries(feedback_dir: Path) -> list[dict[str, Any]]:
    """Read every ``submit_NNN/summary.json`` in order. Robust to missing
    files (e.g., right after init) and to a submit dir without summary
    (mid-write — should not happen in practice given the atomic writer,
    but defensive)."""
    if not feedback_dir.is_dir():
        return []
    submits = sorted(p for p in feedback_dir.iterdir() if p.is_dir() and p.name.startswith("submit_"))
    out: list[dict[str, Any]] = []
    for s in submits:
        summary_path = s / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            data = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out
