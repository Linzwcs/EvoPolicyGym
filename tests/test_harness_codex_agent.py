"""CodexAgent (subprocess wrapper) tests.

Mirrors the structure of ``test_harness_claude_agent.py`` — a tiny
stub shell script mimics the ``codex exec --json`` JSONL stream so we
can exercise command composition, session-id scraping, the
turn-0/turn-1 split, log files, timeouts, and the no-cost-or-usage
contract without depending on a real ``codex`` binary or network."""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from hlbench_harness.codex_agent import (
    CodexAgent,
    CodexAgentConfig,
    TurnResult,
    find_codex_binary,
)

# --------------------------- helpers -------------------------------------


def _make_stub_codex(tmp_path: Path, *, body: str) -> Path:
    """Write a tiny executable Python script that pretends to be the
    ``codex`` binary. The ``body`` snippet has full access to
    ``sys.argv`` and ``os.environ``.

    Returns the path; caller passes it to
    ``CodexAgentConfig.codex_binary``."""
    script = tmp_path / "fake_codex"
    script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


# Standard success stub: emits session_meta first, then a couple of
# task / agent_message events, and exits 0. Also echoes argv + cwd +
# env hints to stderr so tests can pin command shape.
_STUB_SUCCESS = """
import json, sys, os
sys.stderr.write("ARGS:" + json.dumps(sys.argv[1:]) + "\\n")
sys.stderr.write("CWD:" + os.getcwd() + "\\n")
sys.stderr.write("URL:" + os.environ.get("HLBENCH_URL", "") + "\\n")

events = [
    {"type": "session_meta",
     "payload": {"id": "01998dc0-aaaa-7bbb-8ccc-ddddeeeeffff",
                 "cwd": os.getcwd(),
                 "cli_version": "0.133.0"}},
    {"type": "task_started", "payload": {"model": "gpt-5-codex"}},
    {"type": "agent_message", "payload": {"message": "wrote system/policy.py"}},
    {"type": "task_complete",
     "payload": {"last_agent_message": "ok turn done"}},
]
for ev in events:
    sys.stdout.write(json.dumps(ev) + "\\n")
    sys.stdout.flush()
sys.exit(0)
"""

# Same as success but the stub OMITS session_meta — used to verify the
# fallback path where turn 1 still does a fresh ``codex exec`` rather
# than ``codex exec resume`` against a bogus id.
_STUB_NO_SESSION_META = """
import json, sys, os
events = [
    {"type": "task_started", "payload": {}},
    {"type": "agent_message", "payload": {"message": "no session_meta path"}},
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


def test_first_turn_uses_exec_subcommand_no_resume(tmp_path: Path) -> None:
    """Turn 0 must invoke ``codex exec ...`` (no ``resume`` subcommand,
    no positional session id). Turn 1 must switch to
    ``codex exec resume <scraped-id> ...``."""
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace,
        http_url="http://h:1",
        log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    r1 = agent.run_turn("hello")
    r2 = agent.run_turn("again")

    assert r1.ok and r2.ok

    args1 = json.loads(r1.stderr.split("ARGS:", 1)[1].splitlines()[0])
    args2 = json.loads(r2.stderr.split("ARGS:", 1)[1].splitlines()[0])

    # Turn 0: bare ``exec``, no positional session id.
    assert args1[0] == "exec"
    assert "resume" not in args1
    # Turn 1: ``exec resume`` plus the scraped id as a positional arg.
    assert args2[:2] == ["exec", "resume"]
    # The scraped id appears just before the prompt (last position).
    assert args2[-2] == "01998dc0-aaaa-7bbb-8ccc-ddddeeeeffff"
    assert args2[-1] == "again"

    # --json + --skip-git-repo-check + bypass-approvals on both turns.
    for args in (args1, args2):
        assert "--json" in args
        assert "--skip-git-repo-check" in args
        assert "--dangerously-bypass-approvals-and-sandbox" in args


def test_session_id_scraped_from_session_meta(tmp_path: Path) -> None:
    """After turn 0, ``agent.codex_session_id`` exposes the scraped
    UUID. The public ``session_id`` is the harness-minted UUID4 and is
    DIFFERENT from the codex internal id."""
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://h",
        log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    # Pre-turn-0 the codex id isn't known yet.
    assert agent.codex_session_id is None
    # The harness-side label is set immediately, though.
    harness_label = agent.session_id
    assert harness_label  # truthy

    agent.run_turn("first")

    # Now we have the codex id, and it differs from the harness label.
    assert agent.codex_session_id == "01998dc0-aaaa-7bbb-8ccc-ddddeeeeffff"
    assert agent.session_id == harness_label  # unchanged
    assert agent.session_id != agent.codex_session_id


def test_no_session_meta_falls_back_to_fresh_exec_on_turn_1(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If turn 0 emits no session_meta event, turn 1 must NOT use
    ``exec resume`` (there's nothing to resume against). It should
    instead launch a fresh ``codex exec`` and log a warning."""
    import logging
    caplog.set_level(logging.WARNING, logger="hlbench_harness.codex_agent")

    stub = _make_stub_codex(tmp_path, body=_STUB_NO_SESSION_META)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://h", log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    agent.run_turn("first")
    assert agent.codex_session_id is None
    # Warning emitted on the missing session_meta path.
    assert any(
        "no session_meta" in rec.getMessage()
        for rec in caplog.records
    )

    r2 = agent.run_turn("second")
    assert r2.ok
    # The no_session_meta stub doesn't echo ARGS to stderr — verify
    # via the recorded transcript file instead.
    transcript = (log_dir / "turn_001.txt").read_text()
    assert " exec " in transcript or transcript.count(" exec") >= 1
    # ``resume`` must NOT appear (we have nothing to resume).
    assert " resume " not in transcript and "exec resume" not in transcript


def test_cwd_workspace_url_in_env(tmp_path: Path) -> None:
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://harness:9999",
        log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    assert f"CWD:{workspace.resolve()}" in r.stderr
    assert "URL:http://harness:9999" in r.stderr


def test_writes_per_turn_logs(tmp_path: Path) -> None:
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://x",
        log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    agent.run_turn("first")
    agent.run_turn("second")

    # Same four-file layout as the claude backend.
    for i in (0, 1):
        for suffix in (".stream.jsonl", ".json", ".txt", ".prompt.txt"):
            f = log_dir / f"turn_{i:03d}{suffix}"
            assert f.is_file(), f"missing {f}"

    assert (log_dir / "turn_000.prompt.txt").read_text() == "first"
    assert (log_dir / "turn_001.prompt.txt").read_text() == "second"

    # Stream file has one JSON object per line.
    stream0 = (log_dir / "turn_000.stream.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in stream0 if line.strip()]
    assert parsed[0]["type"] == "session_meta"
    assert parsed[-1]["type"] == "task_complete"

    # Transcript carries the assistant text.
    transcript0 = (log_dir / "turn_000.txt").read_text()
    assert "ok turn done" in transcript0 or "wrote system/policy.py" in transcript0
    assert "rc=0" in transcript0


def test_nonzero_exit_is_not_ok(tmp_path: Path) -> None:
    stub = _make_stub_codex(tmp_path, body=_STUB_FAIL)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    r = agent.run_turn("x")
    assert r.ok is False
    assert r.exit_code == 2
    assert r.timed_out is False
    # No JSON events on the stream → raw_json stays None.
    assert r.raw_json is None
    assert "simulated failure" in r.stderr


def test_timeout_marks_timed_out(tmp_path: Path) -> None:
    stub = _make_stub_codex(tmp_path, body=_STUB_HANG)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub), timeout_seconds=1),
    )
    r = agent.run_turn("x")
    assert r.timed_out is True
    assert r.exit_code == 124
    assert r.ok is False


def test_find_codex_binary_returns_path_or_none() -> None:
    """Smoke check: function returns either a path string or None
    without raising."""
    out = find_codex_binary()
    assert out is None or os.path.isfile(out)


def test_turn_result_dataclass_round_trip() -> None:
    """TurnResult is frozen; .ok is derived. Sanity check matches the
    claude_agent.TurnResult contract."""
    import dataclasses
    r = TurnResult(turn_index=0, session_id="x", exit_code=0,
                   timed_out=False, duration_seconds=0.1, text="hi")
    assert r.ok
    r2 = TurnResult(turn_index=1, session_id="x", exit_code=1,
                    timed_out=False, duration_seconds=0.2, text="bye")
    assert not r2.ok
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.exit_code = 1  # type: ignore[misc]


def test_cost_and_usage_stay_none_for_codex(tmp_path: Path) -> None:
    """Codex 0.133's --json stream doesn't surface token usage or
    dollar cost. The TurnResult must keep both None — agent.jsonl
    relies on this to drop the keys cleanly."""
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub)),
    )
    r = agent.run_turn("hi")
    assert r.ok
    assert r.cost_usd is None
    assert r.usage is None
    assert r.inner_num_turns is None


def test_command_includes_model_and_workspace_cd(tmp_path: Path) -> None:
    """``-m <model>`` and ``-C <workspace>`` must be on every turn's
    invocation."""
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    agent = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir,
        config=CodexAgentConfig(codex_binary=str(stub), model="gpt-5"),
    )
    r = agent.run_turn("hi")
    args = json.loads(r.stderr.split("ARGS:", 1)[1].splitlines()[0])
    # -m gpt-5
    i = args.index("-m")
    assert args[i + 1] == "gpt-5"
    # -C <workspace>
    j = args.index("-C")
    assert args[j + 1] == str(workspace.resolve())


def test_sandbox_mode_only_when_not_bypassing(tmp_path: Path) -> None:
    """``--dangerously-bypass-approvals-and-sandbox`` strictly
    subsumes ``-s``. When both would be present, codex 0.133 errors;
    we must drop ``-s`` while bypass is on."""
    stub = _make_stub_codex(tmp_path, body=_STUB_SUCCESS)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_dir = tmp_path / "logs"

    # bypass=True (default): no -s.
    agent_bypass = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir / "a",
        config=CodexAgentConfig(
            codex_binary=str(stub), bypass_approvals=True,
            sandbox_mode="workspace-write",
        ),
    )
    r = agent_bypass.run_turn("hi")
    args = json.loads(r.stderr.split("ARGS:", 1)[1].splitlines()[0])
    assert "--dangerously-bypass-approvals-and-sandbox" in args
    assert "-s" not in args

    # bypass=False: -s is forwarded.
    agent_strict = CodexAgent(
        workspace_dir=workspace, http_url="http://x", log_dir=log_dir / "b",
        config=CodexAgentConfig(
            codex_binary=str(stub), bypass_approvals=False,
            sandbox_mode="read-only",
        ),
    )
    r2 = agent_strict.run_turn("hi")
    args2 = json.loads(r2.stderr.split("ARGS:", 1)[1].splitlines()[0])
    assert "--dangerously-bypass-approvals-and-sandbox" not in args2
    i = args2.index("-s")
    assert args2[i + 1] == "read-only"


def test_session_id_property_is_uuid4_string(tmp_path: Path) -> None:
    """Public session_id is a stable harness-minted UUID4. Generated
    eagerly at construction time so agent.jsonl's agent_start event
    has something to record before turn 0 runs."""
    import uuid as _uuid
    agent = CodexAgent(
        workspace_dir=tmp_path, http_url="http://x",
        log_dir=tmp_path / "logs",
        config=CodexAgentConfig(codex_binary="/nonexistent"),
    )
    # Non-empty + parseable.
    sid = agent.session_id
    assert sid
    parsed = _uuid.UUID(sid)
    assert str(parsed) == sid
