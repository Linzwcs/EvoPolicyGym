"""KimiAgent (subprocess wrapper) tests.

Mirrors the structure of ``test_harness_codex_agent.py`` — tiny stub
shell scripts mimic the ``kimi --output-format stream-json``
event stream so we can exercise command composition, the two-tier
session-id resolution (stream scrape → session_index.jsonl fallback),
the resume vs. continue command flag pickers, log files, timeouts,
and the no-cost-or-usage contract without depending on a real
``kimi`` binary or network."""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from hlbench_harness.kimi_agent import (
    KimiAgent,
    KimiAgentConfig,
    TurnResult,
    _find_session_id,
    _lookup_session_in_index,
    find_kimi_binary,
)

# --------------------------- helpers -------------------------------------


def _make_stub_kimi(tmp_path: Path, *, body: str) -> Path:
    """Write a tiny executable Python script that pretends to be the
    ``kimi`` binary. The ``body`` snippet has full access to
    ``sys.argv`` and ``os.environ``.

    Returns the path; caller passes it to
    ``KimiAgentConfig.kimi_binary``."""
    script = tmp_path / "fake_kimi"
    script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


# Standard success stub: emits a stream-json event carrying a
# kimi-shape sessionId in the payload, then an assistant message.
# Echoes argv + cwd + env to stderr for command-shape assertions.
_STUB_SUCCESS = """
import json, sys, os
sys.stderr.write("ARGS:" + json.dumps(sys.argv[1:]) + "\\n")
sys.stderr.write("CWD:" + os.getcwd() + "\\n")
sys.stderr.write("URL:" + os.environ.get("HLBENCH_URL", "") + "\\n")

events = [
    {"type": "meta",
     "payload": {"sessionId": "session_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeffffff",
                 "workDir": os.getcwd(),
                 "cli_version": "0.6.0"}},
    {"type": "agent_message", "message": "wrote system/policy.py"},
    {"type": "final", "message": "ok turn done"},
]
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\\n")
    sys.stdout.flush()
sys.exit(0)
"""

# Same as success but emits no sessionId anywhere — exercises the
# session_index.jsonl fallback. A test will populate the index file
# before calling.
_STUB_NO_SESSION_ID = """
import json, sys, os
events = [
    {"type": "agent_message", "message": "no sessionId path"},
    {"type": "final", "message": "ok"},
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


def test_turn_0_no_resume_flag(tmp_path: Path) -> None:
    """Turn 0 must not include ``-S`` or ``-C`` — it's a fresh
    session. Turn 1 with a scraped id must use ``-S <id>``."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace,
        http_url="http://h:1",
        log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r1 = agent.run_turn("hello")
    r2 = agent.run_turn("again")

    assert r1.ok and r2.ok

    args1 = json.loads(r1.stderr.split("ARGS:", 1)[1].splitlines()[0])
    args2 = json.loads(r2.stderr.split("ARGS:", 1)[1].splitlines()[0])

    # Turn 0: no -S, no -C.
    assert "-S" not in args1
    assert "-C" not in args1
    # Turn 1: -S with the scraped id.
    assert "-S" in args2
    sid_idx = args2.index("-S")
    assert args2[sid_idx + 1] == "session_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeffffff"
    # Both turns specify stream-json and yolo.
    for args in (args1, args2):
        assert "--output-format" in args
        of_idx = args.index("--output-format")
        assert args[of_idx + 1] == "stream-json"
        assert "-y" in args


def test_session_id_scraped_from_stream(tmp_path: Path) -> None:
    """After turn 0, ``agent.kimi_session_id`` exposes the scraped
    full kimi-internal id. Public ``session_id`` is the harness-minted
    UUID4 and is DIFFERENT from the kimi id."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://h",
        log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    assert agent.kimi_session_id is None
    harness_label = agent.session_id

    agent.run_turn("first")
    assert agent.kimi_session_id == "session_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeffffff"
    assert agent.session_id == harness_label
    assert agent.session_id != agent.kimi_session_id


def test_falls_back_to_session_index_when_stream_lacks_id(
    tmp_path: Path,
) -> None:
    """If turn 0 emits no sessionId in stream-json but
    session_index.jsonl has an entry for our workspace, use that.
    Turn 1 must then use ``-S <indexed-id>`` (NOT ``-C``)."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_NO_SESSION_ID)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    index_path = tmp_path / "session_index.jsonl"

    # Pre-populate the index with an entry matching our workspace.
    # Use the *resolved* workspace path (KimiAgent compares against that).
    index_path.write_text(json.dumps({
        "sessionId": "session_11111111-2222-4333-8444-555555555555",
        "workDir": str(workspace.resolve()),
        "sessionDir": "/dummy/dir",
    }) + "\n")

    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://h",
        log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=index_path,
    )
    agent.run_turn("first")
    assert agent.kimi_session_id == "session_11111111-2222-4333-8444-555555555555"

    r2 = agent.run_turn("second")
    assert r2.ok  # the stub doesn't echo ARGS for ``_STUB_NO_SESSION_ID``,
    # so we verify via the persisted transcript.
    transcript = (log_dir / "turn_001.txt").read_text()
    assert "session_11111111" in transcript


def test_no_session_id_no_index_falls_back_to_continue(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """If both scrape AND index lookup fail, turn 1 uses ``-C``
    (continue most-recent for cwd) and logs a warning."""
    import logging
    caplog.set_level(logging.WARNING, logger="hlbench_harness.kimi_agent")

    # Stub that echoes ARGS and emits NO sessionId.
    stub_body = """
import json, sys, os
sys.stderr.write("ARGS:" + json.dumps(sys.argv[1:]) + "\\n")
sys.stdout.write(json.dumps({"type": "agent_message", "message": "no id"}) + "\\n")
sys.stdout.write(json.dumps({"type": "final", "message": "ok"}) + "\\n")
sys.exit(0)
"""
    stub = _make_stub_kimi(tmp_path, body=stub_body)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://h", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    agent.run_turn("first")
    assert agent.kimi_session_id is None
    # Warning emitted on the "no resolvable id" path.
    assert any(
        "session id" in rec.getMessage() and "fallback" in rec.getMessage()
        for rec in caplog.records
    )

    r2 = agent.run_turn("second")
    args2 = json.loads(r2.stderr.split("ARGS:", 1)[1].splitlines()[0])
    # Turn 1 falls back to -C; no -S because we never resolved an id.
    assert "-C" in args2
    assert "-S" not in args2


def test_cwd_workspace_url_in_env(tmp_path: Path) -> None:
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://harness:9999",
        log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("hi")
    assert f"CWD:{workspace.resolve()}" in r.stderr
    assert "URL:http://harness:9999" in r.stderr


def test_writes_per_turn_logs(tmp_path: Path) -> None:
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x",
        log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    agent.run_turn("first")
    agent.run_turn("second")

    # Same four-file layout as the claude / codex backends.
    for i in (0, 1):
        for suffix in (".stream.jsonl", ".json", ".txt", ".prompt.txt"):
            f = log_dir / f"turn_{i:03d}{suffix}"
            assert f.is_file(), f"missing {f}"

    assert (log_dir / "turn_000.prompt.txt").read_text() == "first"
    assert (log_dir / "turn_001.prompt.txt").read_text() == "second"

    stream0 = (log_dir / "turn_000.stream.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in stream0 if line.strip()]
    assert parsed[0]["type"] == "meta"
    assert parsed[-1]["type"] == "final"

    transcript0 = (log_dir / "turn_000.txt").read_text()
    assert "wrote system/policy.py" in transcript0 or "ok turn done" in transcript0
    assert "rc=0" in transcript0


def test_nonzero_exit_is_not_ok(tmp_path: Path) -> None:
    stub = _make_stub_kimi(tmp_path, body=_STUB_FAIL)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("x")
    assert r.ok is False
    assert r.exit_code == 2
    assert r.timed_out is False
    assert r.raw_json is None
    assert "simulated failure" in r.stderr


def test_timeout_marks_timed_out(tmp_path: Path) -> None:
    stub = _make_stub_kimi(tmp_path, body=_STUB_HANG)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub), timeout_seconds=1),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("x")
    assert r.timed_out is True
    assert r.exit_code == 124
    assert r.ok is False


def test_find_kimi_binary_returns_path_or_none() -> None:
    """Smoke check: function returns either a path string or None
    without raising."""
    out = find_kimi_binary()
    assert out is None or os.path.isfile(out)


def test_turn_result_dataclass_round_trip() -> None:
    """TurnResult is frozen; .ok is derived. Matches the claude /
    codex agents' TurnResult contract."""
    import dataclasses
    r = TurnResult(turn_index=0, session_id="x", exit_code=0,
                   timed_out=False, duration_seconds=0.1, text="hi")
    assert r.ok
    r2 = TurnResult(turn_index=1, session_id="x", exit_code=1,
                    timed_out=False, duration_seconds=0.2, text="bye")
    assert not r2.ok
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.exit_code = 1  # type: ignore[misc]


def test_cost_and_usage_stay_none_for_kimi(tmp_path: Path) -> None:
    """Kimi 0.6's stream-json doesn't surface token usage or dollar
    cost. The TurnResult must keep both None — agent.jsonl relies on
    this to drop the keys cleanly."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub)),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("hi")
    assert r.ok
    assert r.cost_usd is None
    assert r.usage is None
    assert r.inner_num_turns is None


def test_command_includes_model_and_yolo(tmp_path: Path) -> None:
    """``-m <model>`` and ``-y`` (yolo) must be on every turn."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub), model="kimi-k2"),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("hi")
    args = json.loads(r.stderr.split("ARGS:", 1)[1].splitlines()[0])
    i = args.index("-m")
    assert args[i + 1] == "kimi-k2"
    assert "-y" in args


def test_yolo_can_be_disabled(tmp_path: Path) -> None:
    """``KimiAgentConfig(yolo=False)`` drops the -y flag."""
    stub = _make_stub_kimi(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = KimiAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=KimiAgentConfig(kimi_binary=str(stub), yolo=False),
        session_index_path=tmp_path / "no-such-index.jsonl",
    )
    r = agent.run_turn("hi")
    args = json.loads(r.stderr.split("ARGS:", 1)[1].splitlines()[0])
    assert "-y" not in args


def test_session_id_property_is_uuid4_string(tmp_path: Path) -> None:
    """Public session_id is a stable harness-minted UUID4."""
    import uuid as _uuid
    agent = KimiAgent(
        workspace_dir=tmp_path, http_url="http://x",
        log_dir=tmp_path / "logs",
        config=KimiAgentConfig(kimi_binary="/nonexistent"),
    )
    sid = agent.session_id
    assert sid
    parsed = _uuid.UUID(sid)
    assert str(parsed) == sid


# --------------------------- helper unit tests ----------------------------


def test_find_session_id_nested_payload() -> None:
    """``_find_session_id`` finds the id at any depth."""
    obj = {"type": "meta", "payload": {"sessionId": "session_dead-beef"}}
    assert _find_session_id(obj) == "session_dead-beef"


def test_find_session_id_in_list() -> None:
    obj = [{"x": 1}, {"sessionId": "session_aaaa"}]
    assert _find_session_id(obj) == "session_aaaa"


def test_find_session_id_returns_none_when_absent() -> None:
    obj = {"type": "agent_message", "message": "no id here"}
    assert _find_session_id(obj) is None


def test_lookup_session_in_index_filters_by_workdir(tmp_path: Path) -> None:
    """Picks the most-recent entry whose ``workDir`` matches."""
    idx = tmp_path / "session_index.jsonl"
    ws_a = tmp_path / "ws-a"
    ws_a.mkdir()
    ws_b = tmp_path / "ws-b"
    ws_b.mkdir()

    idx.write_text(
        json.dumps({"sessionId": "session_aaa", "workDir": str(ws_a)}) + "\n" +
        json.dumps({"sessionId": "session_bbb", "workDir": str(ws_b)}) + "\n" +
        json.dumps({"sessionId": "session_ccc", "workDir": str(ws_a)}) + "\n",
    )
    # ws_a has two entries — must return the latest (ccc), not aaa.
    assert _lookup_session_in_index(idx, ws_a) == "session_ccc"
    assert _lookup_session_in_index(idx, ws_b) == "session_bbb"
    # Workspace with no entries → None.
    assert _lookup_session_in_index(idx, tmp_path / "no-such-ws") is None


def test_lookup_session_in_index_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    assert _lookup_session_in_index(tmp_path / "no-such.jsonl", tmp_path) is None
