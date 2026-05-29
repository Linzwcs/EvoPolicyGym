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

# Like _STUB_SUCCESS but also emits cost/usage fields the real claude
# CLI returns. Lets us verify the harness parses them when present.
_STUB_SUCCESS_WITH_COST = """
import json, sys, os
sys.stdout.write(json.dumps({
    "type": "result",
    "subtype": "success",
    "result": "ok with cost",
    "session_id": os.environ.get("HLBENCH_SESSION_ID", "unknown"),
    "total_cost_usd": 0.1234,
    "num_turns": 3,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 12345,
        "cache_read_input_tokens": 0,
        "server_tool_use": {"web_search_requests": 0},
    },
}))
sys.exit(0)
"""

# Streaming stub: emits multiple JSON events (mimics what real claude
# does with --output-format=stream-json) — assistant message, tool
# use, tool result, then the terminal "result" event. Lets us verify
# the harness captures the whole stream, not just the final event.
_STUB_STREAMING = """
import json, sys, os, time
events = [
    {"type": "system", "subtype": "init", "session_id": os.environ.get("HLBENCH_SESSION_ID", "?")},
    {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "let me write a policy"}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": "system/policy.py"}}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "content": "file written"}]}},
    {"type": "result",
     "subtype": "success",
     "result": "wrote policy + submitted",
     "session_id": os.environ.get("HLBENCH_SESSION_ID", "?"),
     "total_cost_usd": 0.05,
     "num_turns": 1,
     "usage": {"input_tokens": 50, "output_tokens": 30}},
]
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\\n")
    sys.stdout.flush()
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
    # Both must declare --print, --verbose, and stream-json output
    # (stream-json is required for thought-process capture; --verbose
    # is required by the claude CLI when combining --print with
    # stream-json output).
    for args in (args1, args2):
        assert "--print" in args
        assert "--verbose" in args
        assert "--output-format" in args
        # The flag value follows --output-format directly.
        of_idx = args.index("--output-format")
        assert args[of_idx + 1] == "stream-json"


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


# --------------------------- cost / usage parsing ----------------------------


def test_cost_and_usage_extracted_from_json(tmp_path: Path) -> None:
    """When claude's JSON envelope includes total_cost_usd / num_turns /
    usage, the TurnResult surfaces them."""
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS_WITH_COST)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x",
        log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    assert r.ok
    assert r.cost_usd == 0.1234
    assert r.inner_num_turns == 3
    # usage is filtered to int-valued keys only (server_tool_use is a
    # nested dict and must be excluded).
    assert r.usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 12345,
        "cache_read_input_tokens": 0,
    }


def test_cost_absent_yields_none(tmp_path: Path) -> None:
    """The standard test stub doesn't emit cost; the harness must
    tolerate that and leave the fields as None (rather than 0.0)."""
    stub = _make_stub_claude(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x",
        log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    assert r.ok
    assert r.cost_usd is None
    assert r.inner_num_turns is None
    assert r.usage is None


def test_failed_turn_has_no_cost(tmp_path: Path) -> None:
    """A turn that exited non-zero with no JSON output also leaves the
    cost fields None — defensive against partial-write scenarios."""
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
    assert r.cost_usd is None
    assert r.usage is None


# --------------------------- streaming capture ----------------------------


def test_stream_jsonl_captures_full_event_stream(tmp_path: Path) -> None:
    """A streaming stub emits 5 events: system/init, two assistant
    messages (thinking + tool_use), tool_result, result. All must
    land verbatim in ``turn_NNN.stream.jsonl`` so analysts can replay
    the agent's thought process."""
    stub = _make_stub_claude(tmp_path, body=_STUB_STREAMING)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    assert r.ok

    # Per-turn stream file exists with one JSON object per line.
    stream_path = log_dir / "turn_000.stream.jsonl"
    assert stream_path.is_file()
    events = [json.loads(line) for line in stream_path.read_text().splitlines() if line.strip()]
    assert len(events) == 5
    assert events[0]["type"] == "system"
    assert events[1]["type"] == "assistant"
    # The thinking block is preserved (this is the whole point).
    thinking_block = events[1]["message"]["content"][0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == "let me write a policy"
    # Tool use captured.
    assert events[2]["message"]["content"][0]["type"] == "tool_use"
    assert events[3]["message"]["content"][0]["type"] == "tool_result"
    assert events[4]["type"] == "result"

    # The terminal result event is also surfaced through TurnResult.
    assert r.text == "wrote policy + submitted"
    assert r.cost_usd == 0.05
    assert r.usage == {"input_tokens": 50, "output_tokens": 30}

    # turn_000.json mirrors just the result event (backwards-compat
    # quick-access view).
    json_only = json.loads((log_dir / "turn_000.json").read_text())
    assert json_only["type"] == "result"
    assert json_only["total_cost_usd"] == 0.05


def test_stream_jsonl_exists_even_when_turn_times_out(tmp_path: Path) -> None:
    """Real motivation for streaming: when a turn times out we still
    have all events emitted up to the kill, so the agent's progress
    is recoverable."""
    # Stub that emits 3 events, then hangs forever. With timeout=1s
    # the harness will kill after 1s but the 3 events should already
    # be on disk.
    stub_body = """
import json, sys, time
for i in range(3):
    sys.stdout.write(json.dumps({"type":"assistant","msg":i}) + "\\n")
    sys.stdout.flush()
time.sleep(60)  # would block well past the harness timeout
"""
    stub = _make_stub_claude(tmp_path, body=stub_body)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = ClaudeAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=ClaudeAgentConfig(claude_binary=str(stub), timeout_seconds=2),
    )
    r = agent.run_turn("hi")
    assert r.timed_out
    # No result event ⇒ no cost.
    assert r.cost_usd is None
    # But the partial stream is preserved.
    stream_path = log_dir / "turn_000.stream.jsonl"
    assert stream_path.is_file()
    events = [json.loads(line) for line in stream_path.read_text().splitlines() if line.strip()]
    assert len(events) == 3
    assert all(e["type"] == "assistant" for e in events)
    assert [e["msg"] for e in events] == [0, 1, 2]
