"""OpenAI Codex CLI agent backend.

Drives the inner agent by spawning ``codex exec`` non-interactively
(``codex-cli`` >= 0.133). Conversation context is preserved across
turns via ``codex exec resume <session-id>``.

Unlike Claude Code (which lets callers pre-allocate a UUID via
``--session-id``), Codex auto-generates the session id internally and
emits it as the first JSONL event (``type:"session_meta"``) when
``--json`` is on. We scrape that id on turn 0 and pass it to
``codex exec resume`` on every subsequent turn.

The public ``session_id`` property is a harness-side UUID4 that we
mint upfront — it's what lands in ``agent.jsonl`` and
``harness_runner.json`` so downstream tools have a stable label even
before turn 0 has run. The scraped Codex-internal session id is held
privately and used only to build the ``exec resume`` command.

Per-turn artifacts (one set per turn) — same shape as the Claude
backend so analyst tools can be backend-agnostic::

    run_dir/logs/agent_turns/turn_NNN.stream.jsonl   full --json stream
    run_dir/logs/agent_turns/turn_NNN.json           mirror of final event(s)
    run_dir/logs/agent_turns/turn_NNN.txt            human-readable transcript
    run_dir/logs/agent_turns/turn_NNN.prompt.txt     exact prompt sent

Codex 0.133's ``--json`` stream does NOT carry per-turn token usage
or dollar cost. Both ``TurnResult.cost_usd`` and ``TurnResult.usage``
stay ``None`` for this backend; if upstream adds them, surface them
the same way ``claude_agent`` does.

Heads-up for operators: Codex persists every session as
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` outside the hlbench
``run_dir``. The ``--ephemeral`` flag would suppress that, but it's
mutually exclusive with ``resume``, so we don't ship it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one ``codex exec`` invocation.

    Mirrors ``claude_agent.TurnResult`` field-for-field so the
    ``HarnessRunner`` (which reads via ``getattr``) works against both
    backends without changes. ``cost_usd`` and ``usage`` are always
    ``None`` for codex 0.133 — its ``--json`` event stream doesn't
    carry token / cost information.
    """

    turn_index: int
    session_id: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    text: str
    raw_json: dict[str, Any] | None = None
    stderr: str = ""
    cost_usd: float | None = None
    inner_num_turns: int | None = None
    usage: dict[str, int] | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class CodexAgentConfig:
    """Knobs for the Codex CLI agent backend.

    Defaults err on the side of "let the agent work without nagging":
    ``bypass_approvals=True`` adds
    ``--dangerously-bypass-approvals-and-sandbox`` (the harness is
    the actual sandbox), ``sandbox_mode="workspace-write"`` matches
    the ``codex exec`` default, and the timeout is generous because
    each turn may include several shell + edit calls plus codex's
    first-call latency.
    """

    model: str = "gpt-5-codex"
    sandbox_mode: str = "workspace-write"
    bypass_approvals: bool = True
    timeout_seconds: int = 900  # codex first-call latency can blow 600
    extra_args: tuple[str, ...] = ()  # passthrough for -c key=val etc.
    codex_binary: str = "codex"


class CodexAgent:
    """Stateful wrapper around ``codex exec`` (and ``codex exec resume``).

    Holds two ids: a harness-minted ``session_id`` (UUID4, stable for
    the whole run, exposed publicly) and the Codex-internal session
    id scraped from the first turn's ``session_meta`` JSONL event
    (private, used to build subsequent ``exec resume`` commands).

    If turn 0 dies before emitting ``session_meta`` (binary missing,
    auth failure, etc.), ``_codex_session_id`` stays ``None`` and the
    next turn falls back to a fresh ``codex exec`` rather than
    resuming against a bogus id. A warning is logged.

    Not thread-safe — only one turn in flight at a time.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        http_url: str,
        log_dir: Path,
        config: CodexAgentConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._http_url = http_url
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or CodexAgentConfig()
        self._session_id = session_id or str(uuid.uuid4())
        self._codex_session_id: str | None = None
        self._turn_count = 0

    @property
    def session_id(self) -> str:
        """Harness-side stable label. Distinct from the Codex-internal
        session id (which is scraped from turn 0's stream and used
        only to build ``exec resume``). Tools downstream key off
        this."""
        return self._session_id

    @property
    def codex_session_id(self) -> str | None:
        """The Codex-internal session id (UUID7) scraped from
        ``session_meta``. ``None`` until turn 0 emits it."""
        return self._codex_session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def run_turn(self, prompt: str) -> TurnResult:
        """Run one turn.

        Turn 0 (or any turn where we haven't scraped a codex session
        id yet): spawn ``codex exec ...``. Otherwise: spawn
        ``codex exec resume <codex_session_id> ...``.

        Uses ``--json`` so we capture the full event stream to
        ``logs/agent_turns/turn_NNN.stream.jsonl`` — even if the turn
        times out, the partial trace up to the kill is preserved.
        """
        turn_index = self._turn_count
        self._turn_count += 1

        cmd = self._build_command(prompt)
        env = self._build_env()
        stream_path = self._log_dir / f"turn_{turn_index:03d}.stream.jsonl"
        json_path = self._log_dir / f"turn_{turn_index:03d}.json"
        text_path = self._log_dir / f"turn_{turn_index:03d}.txt"
        prompt_path = self._log_dir / f"turn_{turn_index:03d}.prompt.txt"
        prompt_path.write_text(prompt)

        # Line-buffered stdout so the reader thread sees events as
        # they're emitted, not in chunks at process exit.
        proc = subprocess.Popen(
            cmd,
            cwd=self._workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        events: list[dict[str, Any]] = []
        stderr_buf: list[str] = []
        scraped_session_id: str | None = None
        started = time.monotonic()

        assert proc.stdout is not None
        assert proc.stderr is not None
        stream_file = stream_path.open("w", encoding="utf-8")

        def _stdout_reader() -> None:
            nonlocal scraped_session_id
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    stream_file.write(line)
                    stream_file.flush()
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    events.append(obj)
                    # Scrape session id from the first session_meta
                    # event. Codex emits it as event 0 in --json mode.
                    if scraped_session_id is None and obj.get("type") == "session_meta":
                        payload = obj.get("payload")
                        if isinstance(payload, dict):
                            raw = payload.get("id")
                            if isinstance(raw, str) and raw:
                                scraped_session_id = raw
            except Exception:  # pragma: no cover (defensive)
                pass

        def _stderr_reader() -> None:
            assert proc.stderr is not None
            try:
                for line in proc.stderr:
                    stderr_buf.append(line)
            except Exception:  # pragma: no cover
                pass

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        timed_out = False
        try:
            proc.wait(timeout=self._config.timeout_seconds)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            returncode = 124
            timed_out = True

        duration = time.monotonic() - started

        out_thread.join(timeout=5)
        err_thread.join(timeout=5)
        stream_file.close()
        stderr = "".join(stderr_buf)

        # Persist the scraped id (if any). If turn 0 didn't emit
        # session_meta we leave _codex_session_id as None so the next
        # turn does a fresh exec rather than resuming against
        # nothing.
        if scraped_session_id is not None and self._codex_session_id is None:
            self._codex_session_id = scraped_session_id
            log.info(
                "codex session id scraped: %s (harness label: %s)",
                scraped_session_id, self._session_id,
            )
        elif self._codex_session_id is None and turn_index == 0:
            log.warning(
                "turn 0 produced no session_meta event; turn 1 will "
                "start a fresh codex exec rather than resume",
            )

        # Best-effort text extraction: concatenate any assistant
        # message text we find. Codex 0.133 puts assistant content
        # under several event shapes; we look for ``msg.message`` or
        # ``payload.message.content`` strings. If nothing's found,
        # ``text`` stays empty — agent.jsonl will just record
        # text_len=0 and the harness will rely on remaining_budget
        # changes to detect progress.
        text_response = _extract_assistant_text(events)

        # Mirror the last event into turn_NNN.json for backwards
        # compat with the Claude file shape. Full stream lives in
        # the .stream.jsonl alongside.
        raw_json: dict[str, Any] | None = events[-1] if events else None
        json_path.write_text(
            json.dumps(raw_json, indent=2) if raw_json is not None else ""
        )
        text_path.write_text(_format_transcript(
            turn_index=turn_index,
            session_id=self._session_id,
            codex_session_id=self._codex_session_id,
            cmd=cmd,
            duration=duration,
            returncode=returncode,
            timed_out=timed_out,
            text_response=text_response,
            stderr=stderr,
        ))

        return TurnResult(
            turn_index=turn_index,
            session_id=self._session_id,
            exit_code=returncode,
            timed_out=timed_out,
            duration_seconds=round(duration, 2),
            text=text_response,
            raw_json=raw_json,
            stderr=stderr,
            # cost_usd + usage stay None for codex 0.133.
            cost_usd=None,
            inner_num_turns=None,
            usage=None,
        )

    def _build_command(self, prompt: str) -> list[str]:
        cfg = self._config
        cmd: list[str] = [cfg.codex_binary, "exec"]

        # Resume path: codex exec resume [OPTS] <SESSION_ID> <PROMPT>.
        if self._codex_session_id is not None:
            cmd.append("resume")

        cmd.extend([
            "--json",
            "--skip-git-repo-check",
            "-C", str(self._workspace),
        ])
        if cfg.bypass_approvals:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        # Note: --sandbox is rejected with --dangerously-bypass-...
        # by codex 0.133 (the bypass strictly subsumes any --sandbox
        # mode). Only forward --sandbox when we are NOT bypassing.
        elif cfg.sandbox_mode:
            cmd.extend(["-s", cfg.sandbox_mode])
        if cfg.model:
            cmd.extend(["-m", cfg.model])
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)

        # Positional: [SESSION_ID] (only for resume) [PROMPT].
        if self._codex_session_id is not None:
            cmd.append(self._codex_session_id)
        cmd.append(prompt)
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HLBENCH_URL"] = self._http_url
        env["HLBENCH_SESSION_ID"] = self._session_id
        return env


def find_codex_binary() -> str | None:
    """Resolve the ``codex`` binary on PATH. Returns ``None`` if missing.

    The CLI checks this at startup so we can fail loudly rather than
    on the first turn."""
    return shutil.which("codex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    """Best-effort: concatenate any string-typed assistant message
    content found in the codex --json event stream.

    Codex 0.133 emits several event shapes; we tolerate any of:

      {"type":"agent_message", "payload":{"message":"..."}}
      {"type":"assistant_message", "payload":{"content":"..."}}
      {"msg":{"type":"task_complete", "last_agent_message":"..."}}

    If none match, returns "" — analysts can still read the full
    stream from ``turn_NNN.stream.jsonl``.
    """
    bits: list[str] = []
    for ev in events:
        # task_complete event style
        msg = ev.get("msg") if isinstance(ev.get("msg"), dict) else None
        if msg is not None:
            for key in ("last_agent_message", "message"):
                v = msg.get(key)
                if isinstance(v, str) and v:
                    bits.append(v)
        # agent_message / assistant_message event style
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else None
        if payload is not None:
            for key in ("message", "content", "text"):
                v = payload.get(key)
                if isinstance(v, str) and v:
                    bits.append(v)
    # Deduplicate adjacent repeats (codex sometimes echoes the final
    # assistant message in both ``agent_message`` and ``task_complete``).
    deduped: list[str] = []
    for b in bits:
        if not deduped or deduped[-1] != b:
            deduped.append(b)
    return "\n".join(deduped)


def _format_transcript(
    *,
    turn_index: int,
    session_id: str,
    codex_session_id: str | None,
    cmd: list[str],
    duration: float,
    returncode: int,
    timed_out: bool,
    text_response: str,
    stderr: str,
) -> str:
    """Human-readable transcript file for one turn."""
    bits: list[str] = []
    bits.append(f"# turn {turn_index}  session_id={session_id}")
    if codex_session_id is not None:
        bits.append(f"# codex_session_id={codex_session_id}")
    bits.append(f"# duration={duration:.2f}s  rc={returncode}  timed_out={timed_out}")
    bits.append(f"# cmd: {_shell_quote(cmd)}")
    bits.append("")
    bits.append("## response")
    bits.append("")
    bits.append(text_response or "(empty)")
    if stderr.strip():
        bits.append("")
        bits.append("## stderr")
        bits.append("")
        bits.append(stderr)
    return "\n".join(bits) + "\n"


def _shell_quote(parts: list[str]) -> str:
    """Cheap shell-quote for log readability — not safe for actual exec."""
    out: list[str] = []
    for p in parts:
        if any(c in p for c in [" ", "'", '"', "\n", "\t"]):
            escaped = p.replace("'", "'\\''")
            out.append(f"'{escaped}'")
        else:
            out.append(p)
    return " ".join(out)


__all__ = [
    "CodexAgent",
    "CodexAgentConfig",
    "TurnResult",
    "find_codex_binary",
]
