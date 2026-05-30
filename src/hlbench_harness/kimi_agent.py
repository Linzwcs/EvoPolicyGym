"""Moonshot Kimi Code CLI agent backend.

Drives the inner agent by spawning ``kimi`` non-interactively
(Kimi Code 0.6.0+). Session continuity across turns rides on
kimi's ``-S <session-id>`` resume flag, exactly like Claude and
Codex (different scrape mechanics, same overall pattern).

Kimi Code is the next-gen agentic CLI from Moonshot AI. It has
built-in agentic tools (Read, Write, Bash, etc.), so the inner
``policy.py`` editing / curl-to-HTTP loop works the same as on
Claude Code or Codex.

Kimi differs from claude/codex in a few practical ways:

- No caller-allocated session id (like codex). Sessions get
  auto-generated and registered in
  ``~/.kimi-code/session_index.jsonl`` keyed by ``workDir``.
- The id is also stamped into stream-json output (we still scrape
  it from the event stream when possible; the index file is a
  reliable backup since hlbench gives each run a unique
  workspace_dir).
- ``-y`` (yolo) is the analog of claude's ``bypassPermissions``
  / codex's ``--dangerously-bypass-approvals-and-sandbox``.
- Default model is ``kimi-k2`` (per the v0 launcher in commit
  ``d121daf``).

The public ``session_id`` property is a harness-side UUID4 (minted
upfront so ``agent.jsonl:agent_start`` has a stable label). The
scraped kimi-internal session id (``session_<uuid>``) is held
privately and used to build the ``-S <id>`` resume command.

Per-turn artifacts (one set per turn) — same shape as the other
backends so analyst tools work unchanged::

    run_dir/logs/agent_turns/turn_NNN.stream.jsonl   full --output-format stream-json
    run_dir/logs/agent_turns/turn_NNN.json           mirror of last event
    run_dir/logs/agent_turns/turn_NNN.txt            human-readable transcript
    run_dir/logs/agent_turns/turn_NNN.prompt.txt     exact prompt sent

Token usage and dollar cost: not surfaced by kimi 0.6's
stream-json output, so ``TurnResult.cost_usd`` and
``TurnResult.usage`` stay ``None``. Per-session wire logs at
``~/.kimi-code/sessions/<wd>_<hash>/session_<uuid>/agents/main/wire.jsonl``
contain the raw request/response transcript for any out-of-band
analysis.

Operator note: kimi persists sessions to ``~/.kimi-code/sessions/``
outside the hlbench ``run_dir``. The harness doesn't redirect that
(no ``--ephemeral`` flag exists; redirecting would also break the
session-index lookup we rely on for the id scrape fallback).
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


# Default location of kimi's session index file. Overridable in tests
# by constructing KimiAgent with a custom session_index_path.
_DEFAULT_SESSION_INDEX = Path.home() / ".kimi-code" / "session_index.jsonl"


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one ``kimi -p`` invocation.

    Mirrors ``claude_agent.TurnResult`` field-for-field so the
    ``HarnessRunner`` (which reads via ``getattr``) works against all
    three backends without changes. ``cost_usd``, ``inner_num_turns``,
    and ``usage`` are always ``None`` for kimi 0.6 — its stream-json
    output doesn't carry per-turn token / cost information.
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
class KimiAgentConfig:
    """Knobs for the Kimi Code CLI agent backend.

    Defaults mirror the v0 launcher in commit ``d121daf``
    (``--model kimi-k2``) plus yolo mode for unattended runs.
    ``timeout_seconds`` matches the codex default (900s) since kimi
    Code's first-call latency is comparable."""

    model: str = "kimi-k2"
    yolo: bool = True  # auto-approve all actions (sandbox is harness-side)
    timeout_seconds: int = 900
    extra_args: tuple[str, ...] = ()
    kimi_binary: str = "kimi"


class KimiAgent:
    """Stateful wrapper around ``kimi -p`` with ``-S`` resume.

    Holds two ids:

    - ``session_id`` (public, UUID4): harness-side label. Stable for
      the whole run; embedded in ``agent.jsonl``,
      ``harness_runner.json``, and per-turn transcripts.
    - ``kimi_session_id`` (private, ``session_<UUID4>``): kimi's
      internal id, scraped from turn 0's stream-json or read from
      ``~/.kimi-code/session_index.jsonl`` after turn 0. Passed to
      ``-S`` on every subsequent turn.

    If we fail to determine the kimi-internal id after turn 0 (no
    stream-json scrape AND no index entry for our workspace), turn
    1 falls back to ``-C`` (continue most-recent session for cwd) —
    safe because each hlbench run uses a unique workspace_dir, so
    ``-C`` will resume our session and not someone else's.

    Not thread-safe — only one turn in flight at a time.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        http_url: str,
        log_dir: Path,
        config: KimiAgentConfig | None = None,
        session_id: str | None = None,
        session_index_path: Path | None = None,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._http_url = http_url
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or KimiAgentConfig()
        self._session_id = session_id or str(uuid.uuid4())
        self._kimi_session_id: str | None = None
        self._turn_count = 0
        self._session_index_path = session_index_path or _DEFAULT_SESSION_INDEX

    @property
    def session_id(self) -> str:
        """Harness-side stable label (UUID4)."""
        return self._session_id

    @property
    def kimi_session_id(self) -> str | None:
        """The kimi-internal session id (``session_<uuid>``) once
        scraped. ``None`` until turn 0 establishes it."""
        return self._kimi_session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def run_turn(self, prompt: str) -> TurnResult:
        """Run one turn.

        - Turn 0 (or any turn where we don't yet know
          ``_kimi_session_id``): spawn ``kimi -p ...`` fresh.
        - Subsequent turns: spawn ``kimi -S <id> -p ...`` to resume.
        - If id resolution failed but we already ran turn 0, use
          ``kimi -C -p ...`` (continue most-recent for cwd).

        Uses ``--output-format stream-json`` so we capture the full
        event stream — even on timeout, the partial trace is on
        disk.
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
                    # Defensively scrape ``sessionId`` from any event
                    # that carries one. Kimi's wire format may place
                    # it on a top-level event or nested under
                    # ``payload`` / ``meta``; cover all three.
                    if scraped_session_id is None:
                        candidate = _find_session_id(obj)
                        if candidate:
                            scraped_session_id = candidate
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

        # Determine the kimi-internal session id for next turn's
        # ``-S`` flag. Three-tier resolution:
        #   1. scraped from stream-json (cheap, in-process)
        #   2. session_index.jsonl filtered by workDir (filesystem)
        #   3. give up; next turn uses ``-C`` (continue) instead
        if self._kimi_session_id is None:
            if scraped_session_id is not None:
                self._kimi_session_id = scraped_session_id
                log.info(
                    "kimi session id scraped from stream: %s "
                    "(harness label: %s)",
                    scraped_session_id, self._session_id,
                )
            else:
                from_index = _lookup_session_in_index(
                    self._session_index_path, self._workspace,
                )
                if from_index is not None:
                    self._kimi_session_id = from_index
                    log.info(
                        "kimi session id resolved from session_index.jsonl: %s "
                        "(harness label: %s)",
                        from_index, self._session_id,
                    )
                elif turn_index == 0:
                    log.warning(
                        "turn 0 produced no scrapable session id and no "
                        "session_index.jsonl entry was found for workDir=%s; "
                        "turn 1 will use ``kimi -C`` (continue most-recent) "
                        "as a fallback",
                        self._workspace,
                    )

        text_response = _extract_assistant_text(events)

        raw_json: dict[str, Any] | None = events[-1] if events else None
        json_path.write_text(
            json.dumps(raw_json, indent=2) if raw_json is not None else ""
        )
        text_path.write_text(_format_transcript(
            turn_index=turn_index,
            session_id=self._session_id,
            kimi_session_id=self._kimi_session_id,
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
            cost_usd=None,
            inner_num_turns=None,
            usage=None,
        )

    def _build_command(self, prompt: str) -> list[str]:
        cfg = self._config
        cmd: list[str] = [cfg.kimi_binary]

        # Resume flag picker:
        #   - have explicit kimi session id → -S <id>
        #   - already ran a turn but no id → -C (continue cwd's last)
        #   - turn 0 → no resume flag
        if self._kimi_session_id is not None:
            cmd.extend(["-S", self._kimi_session_id])
        elif self._turn_count > 1:
            # _turn_count was incremented at the top of run_turn, so
            # ">1" means "we are on turn 2+ and never resolved an id".
            cmd.append("-C")

        cmd.extend(["--output-format", "stream-json"])
        if cfg.yolo:
            cmd.append("-y")
        if cfg.model:
            cmd.extend(["-m", cfg.model])
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)
        cmd.extend(["-p", prompt])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HLBENCH_URL"] = self._http_url
        env["HLBENCH_SESSION_ID"] = self._session_id
        return env


def find_kimi_binary() -> str | None:
    """Resolve the ``kimi`` binary on PATH. Returns ``None`` if missing.

    The CLI checks this at startup so we can fail loudly rather than
    on the first turn.

    Note: kimi-code installs to ``~/.kimi-code/bin/kimi`` and adds
    that to ``PATH`` via shell rc; subprocess inherits the parent's
    PATH so this lookup matches what an interactive shell sees."""
    return shutil.which("kimi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_session_id(obj: Any) -> str | None:
    """Recursively search a JSON-like object for a ``sessionId`` key
    whose value starts with ``session_`` (kimi's convention).

    Defensive against schema drift: kimi's stream-json might place
    the id at the top level, nested under ``payload``, ``meta``,
    or somewhere we haven't seen yet. We do a shallow scan and stop
    at the first match."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (
                k in ("sessionId", "session_id")
                and isinstance(v, str)
                and (v.startswith("session_") or _is_uuid_like(v))
            ):
                return v
            nested = _find_session_id(v)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_session_id(item)
            if nested is not None:
                return nested
    return None


def _is_uuid_like(s: str) -> bool:
    """True iff ``s`` parses as a UUID. Used as a permissive guard so
    we accept either ``session_<uuid>`` or bare ``<uuid>`` from the
    stream."""
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


def _lookup_session_in_index(
    index_path: Path, work_dir: Path,
) -> str | None:
    """Scan ``~/.kimi-code/session_index.jsonl`` for the most recent
    entry whose ``workDir`` matches ``work_dir``, return its
    ``sessionId`` (full string, including ``session_`` prefix).

    Returns ``None`` if the index file is missing, malformed, or
    contains no matching entry. Used as a fallback when stream-json
    scraping fails.
    """
    if not index_path.is_file():
        return None
    work_dir_resolved = str(Path(work_dir).resolve())
    matches: list[str] = []
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            wd = obj.get("workDir")
            sid = obj.get("sessionId")
            if isinstance(wd, str) and isinstance(sid, str) and wd == work_dir_resolved:
                matches.append(sid)
    except OSError:
        return None
    return matches[-1] if matches else None


def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    """Best-effort concatenation of any assistant message text in
    kimi's stream-json event stream.

    Schema is not fully documented as of 0.6; cover common shapes:
      {"type":"agent_message", "message": "..."}
      {"type":"assistant", "content": "..."}
      {"type":"output_text", "text": "..."}
      {"type":"final", "message": "..."}

    If nothing matches, return "" — the full stream is still on
    disk at turn_NNN.stream.jsonl for analysts.
    """
    bits: list[str] = []
    for ev in events:
        for key in ("message", "text", "content"):
            v = ev.get(key)
            if isinstance(v, str) and v:
                bits.append(v)
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else None
        if payload is not None:
            for key in ("message", "text", "content"):
                v = payload.get(key)
                if isinstance(v, str) and v:
                    bits.append(v)
    deduped: list[str] = []
    for b in bits:
        if not deduped or deduped[-1] != b:
            deduped.append(b)
    return "\n".join(deduped)


def _format_transcript(
    *,
    turn_index: int,
    session_id: str,
    kimi_session_id: str | None,
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
    if kimi_session_id is not None:
        bits.append(f"# kimi_session_id={kimi_session_id}")
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
    "KimiAgent",
    "KimiAgentConfig",
    "TurnResult",
    "find_kimi_binary",
]
