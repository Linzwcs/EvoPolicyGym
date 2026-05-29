"""ClaudeAgent (subprocess wrapper) tests.

We don't depend on a real ``claude`` binary in CI. Instead, the test
creates a tiny stub shell script that mimics ``claude --print
--output-format=json`` — prints a JSON envelope to stdout, echoes args
to stderr, exits 0 (or with a configured code).

This is enough to exercise:
  - command-line composition (--session-id on turn 0, --resume on
    turn 1+, allowedTools quoting, model selection),
  - cwd + HLBENCH_URL env injection,
  - JSON parsing of the agent's response,
  - per-turn log files (turn_NNN.json / turn_NNN.txt / prompt.txt),
  - timeout handling,
  - error path (non-zero exit code)."""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from hlbench_harness.claude_agent import (
    DEFAULT_ALLOWED_TOOLS,
    ClaudeAgent,
    ClaudeAgentConfig,
    TurnResult,
    find_claude_binary,
    new_session_id,
)

# --------------------------- helpers -------------------------------------


def _make_stub_claude(tmp_path: Path, *, body: str) -> Path:
    """Write a tiny executable Python script at ``tmp_path/fake_claude``
    that runs ``body`` (which has full access to sys.argv and os.environ).

    Returns the path; caller passes it to ``ClaudeAgentConfig.claude_binary``."""
    script = tmp_path / "fake_claude"
    script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


_STUB_SUCCESS = """
import json, sys, os
# Echo args + env hints to stderr (useful for test assertions on cmd).
sys.stderr.write("ARGS:" + json.dumps(sys.argv[1:]) + "\\n")
sys.stderr.write("CWD:" + os.getcwd() + "\\n")
sys.stderr.write("URL:" + os.environ.get("HLBENCH_URL", "") + "\\n")
# A claude-like JSON envelope. Real one is richer but 'result' is the
# field the harness reads.
sys.stdout.write(json.dumps({
    "type": "result",
    "subtype": "success",
    "result": "ok turn done",
    "session_id": os.environ.get("HLBENCH_SESSION_ID", "unknown"),
}))
sys.stdout.write("\\n")
sys.exit(0)
"""

_STUB_FAIL = """
import sys
sys.stderr.write("simulated failure\\n")
sys.exit(2)
"""

_STUB_HANG = """
import time
time.sleep(60)   # longer than test timeout
"""


# --------------------------- tests ---------------------------------------


def test_new_session_id_returns_valid_uuid() -> None:
    import uuid
    sid = new_session_id()
    # Must round-trip through uuid.UUID without raising.
    parsed = uuid.UUID(sid)
    assert str(parsed) == sid


def test_first_turn_uses_session_id_flag(tmp_path: Path) -> None:
    """Turn 0 invocation must include ``--session-id <uuid>``; later
    turns must switch to ``--resume <uuid>``."""
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    sid = "11111111-2222-3333-4444-555555555555"
    agent = ClaudeAgent(
        workspace_dir=workspace,
        http_url="http://h:1",
        log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
        session_id=sid,
    )
    r1 = agent.run_turn("hello")
    r2 = agent.run_turn("again")

    assert r1.ok and r2.ok
    # Both turns return the same session_id (same conversation).
    assert r1.session_id == sid
    assert r2.session_id == sid
    # Stub echoed args to stderr — parse them.
    args1 = json.loads(r1.stderr.split("ARGS:", 1)[1].splitlines()[0])
    args2 = json.loads(r2.stderr.split("ARGS:", 1)[1].splitlines()[0])
    assert "--session-id" in args1
    assert "--resume" not in args1
    assert "--resume" in args2
    assert "--session-id" not in args2
    # Both must declare --print and JSON output.
    for args in (args1, args2):
        assert "--print" in args
        assert "--output-format" in args
        assert "json" in args


def test_allowed_tools_passed_through(tmp_path: Path) -> None:
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://h",
        log_dir=log_dir,
        config=ClaudeAgentConfig(
            claude_binary=str(stub),
            allowed_tools=("Read", "Bash"),
            model="haiku",
        ),
    )
    r = agent.run_turn("hi")
    args = json.loads(r.stderr.split("ARGS:", 1)[1].splitlines()[0])
    # --allowedTools value is space-separated per claude CLI convention.
    i = args.index("--allowedTools")
    assert args[i + 1] == "Read Bash"
    # Model is forwarded.
    j = args.index("--model")
    assert args[j + 1] == "haiku"


def test_cwd_is_workspace_and_url_is_in_env(tmp_path: Path) -> None:
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://harness:9999",
        log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    # Stub echoed cwd + env to stderr.
    assert f"CWD:{workspace.resolve()}" in r.stderr
    assert "URL:http://harness:9999" in r.stderr


def test_writes_per_turn_logs(tmp_path: Path) -> None:
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x",
        log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    agent.run_turn("first")
    agent.run_turn("second")

    # Three files per turn: json, text transcript, raw prompt.
    for i in (0, 1):
        for suffix in (".json", ".txt", ".prompt.txt"):
            f = log_dir / f"turn_{i:03d}{suffix}"
            assert f.is_file(), f"missing {f}"

    # The prompt file is byte-exact what we passed.
    assert (log_dir / "turn_000.prompt.txt").read_text() == "first"
    assert (log_dir / "turn_001.prompt.txt").read_text() == "second"

    # The JSON file is parseable and carries the stub's 'result' field.
    data0 = json.loads((log_dir / "turn_000.json").read_text())
    assert data0["result"] == "ok turn done"

    # The text transcript includes the agent's response text.
    transcript0 = (log_dir / "turn_000.txt").read_text()
    assert "ok turn done" in transcript0
    assert "rc=0" in transcript0


def test_nonzero_exit_is_not_ok(tmp_path: Path) -> None:
    stub = _make_stub_claude(tmp_path, body=_STUB_FAIL)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    r = agent.run_turn("x")
    assert r.ok is False
    assert r.exit_code == 2
    assert r.timed_out is False
    # No JSON to parse; raw_json should be None.
    assert r.raw_json is None
    # Stderr is preserved.
    assert "simulated failure" in r.stderr


def test_timeout_marks_timed_out(tmp_path: Path) -> None:
    stub = _make_stub_claude(tmp_path, body=_STUB_HANG)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub), timeout_seconds=1),
    )
    r = agent.run_turn("x")
    assert r.timed_out is True
    assert r.exit_code == 124
    assert r.ok is False


def test_find_claude_binary_returns_path_or_none() -> None:
    """Smoke check: function returns either a path string or None
    without raising."""
    out = find_claude_binary()
    assert out is None or os.path.isfile(out)


def test_default_allowed_tools_contains_core_set() -> None:
    """A pin so accidental removal of Bash/Read/Edit/Write is caught."""
    assert "Bash" in DEFAULT_ALLOWED_TOOLS
    assert "Read" in DEFAULT_ALLOWED_TOOLS
    assert "Edit" in DEFAULT_ALLOWED_TOOLS
    assert "Write" in DEFAULT_ALLOWED_TOOLS


def test_turn_result_dataclass_round_trip() -> None:
    """TurnResult is frozen; .ok is derived. Sanity check."""
    import dataclasses
    r = TurnResult(turn_index=0, session_id="x", exit_code=0,
                   timed_out=False, duration_seconds=0.1, text="hi")
    assert r.ok
    r2 = TurnResult(turn_index=1, session_id="x", exit_code=1,
                    timed_out=False, duration_seconds=0.2, text="bye")
    assert not r2.ok
    # Frozen — cannot mutate.
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.exit_code = 1  # type: ignore[misc]
