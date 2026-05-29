"""AgentLog (operator-side agent.jsonl writer) tests.

Pure file I/O — no dependency on Server / Sandbox / claude. The writer
mirrors HarnessLog's structure but emits JSON lines per
``output.md §6.2``."""

from __future__ import annotations

import json
from pathlib import Path

from hlbench_harness.agent_log import AgentLog


def _read_lines(path: Path) -> list[dict]:
    """Read the JSONL file as a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_writes_each_event_as_one_jsonl_line(tmp_path: Path) -> None:
    log_path = tmp_path / "agent.jsonl"
    log = AgentLog(log_path)
    log.agent_start(model="claude-sonnet-4-6", session_id="abc-123")
    log.completion(
        turn_index=0, input_tokens=100, output_tokens=20,
        cost_usd=0.05, latency_ms=1500,
    )
    log.agent_end(reason="budget_exhausted", n_turns=2)

    events = _read_lines(log_path)
    assert [e["event"] for e in events] == ["agent_start", "completion", "agent_end"]

    # Each line has a timestamp + schema_version (mirroring HarnessLog).
    for e in events:
        assert "t" in e
        assert e["schema_version"] == "0.1"

    assert events[0]["model"] == "claude-sonnet-4-6"
    assert events[0]["session_id"] == "abc-123"

    assert events[1]["turn_index"] == 0
    assert events[1]["input_tokens"] == 100
    assert events[1]["output_tokens"] == 20
    assert events[1]["cost_usd"] == 0.05
    assert events[1]["latency_ms"] == 1500

    assert events[2]["reason"] == "budget_exhausted"
    assert events[2]["n_turns"] == 2


def test_none_fields_are_dropped(tmp_path: Path) -> None:
    """Optional fields that are None must not appear in the JSONL — keeps
    the file compact and parseable schemas don't see null clutter."""
    log_path = tmp_path / "agent.jsonl"
    log = AgentLog(log_path)
    log.completion(
        turn_index=0,
        input_tokens=None,  # dropped
        output_tokens=10,
        cost_usd=None,      # dropped
        latency_ms=500,
    )
    events = _read_lines(log_path)
    assert "input_tokens" not in events[0]
    assert "cost_usd" not in events[0]
    assert events[0]["output_tokens"] == 10
    assert events[0]["latency_ms"] == 500


def test_disabled_is_noop(tmp_path: Path) -> None:
    """``AgentLog.disabled()`` swallows all events — used by tests +
    lib mode where no run_dir exists."""
    log = AgentLog.disabled()
    log.agent_start(model="x", session_id="y")
    log.completion(turn_index=0)
    log.agent_end(reason="max_turns")
    assert log.path is None
    # No file to read; success = no exception.


def test_touches_log_file_on_construction(tmp_path: Path) -> None:
    """Empty agent.jsonl exists immediately so consumers can ``tail -f``
    it before the first event fires."""
    log_path = tmp_path / "logs" / "agent.jsonl"
    AgentLog(log_path)
    assert log_path.is_file()
    assert log_path.read_text() == ""


def test_arbitrary_event_with_kwargs(tmp_path: Path) -> None:
    """The low-level ``event()`` accepts any kwargs; useful if operators
    want to log custom event types beyond the typed helpers."""
    log_path = tmp_path / "agent.jsonl"
    log = AgentLog(log_path)
    log.event("custom_event", foo=42, bar="baz", nested=[1, 2, 3])
    events = _read_lines(log_path)
    assert len(events) == 1
    assert events[0]["event"] == "custom_event"
    assert events[0]["foo"] == 42
    assert events[0]["bar"] == "baz"
    assert events[0]["nested"] == [1, 2, 3]


def test_write_failure_is_swallowed(tmp_path: Path) -> None:
    """Observability must NEVER break the run. If the path goes
    unwritable mid-run, event() returns silently."""
    log_path = tmp_path / "agent.jsonl"
    log = AgentLog(log_path)
    log.event("first", x=1)
    # Make the parent dir non-writable.
    (tmp_path).chmod(0o555)
    try:
        # Must not raise.
        log.event("second", x=2)
    finally:
        (tmp_path).chmod(0o755)
    # First event still readable.
    events = _read_lines(log_path)
    assert events[0]["event"] == "first"
