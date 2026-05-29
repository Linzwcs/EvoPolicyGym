"""Claude Code CLI agent backend.

Drives the inner agent by spawning ``claude --print`` non-interactively,
preserving conversation context across calls via a pre-allocated session
UUID (``--session-id`` on turn 0, ``--resume <uuid>`` on subsequent turns).

The harness owns this UUID, so we never have to scrape it from the
agent's first reply — that means we can also re-attach to a session
that survived an aborted run, if we ever wanted resumption (not used
in v0.1).

Per-turn artifacts (one set per turn):

    run_dir/logs/agent_turns/turn_NNN.json   — claude's full --output-format=json
    run_dir/logs/agent_turns/turn_NNN.txt    — human-readable transcript

The harness intentionally captures stdout/stderr but does NOT echo them
to its own stdout. Tail those files if you want to watch live.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tools we allow the inner agent. Read/Edit/Write covers policy.py edits;
# Bash covers curl-to-HTTP. Glob/Grep let it search its own feedback dir.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
)


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one ``claude --print`` invocation.

    ``raw_json`` is the full ``--output-format=json`` response (when
    parseable); ``text`` is the agent's final message. ``exit_code``
    is the subprocess return code; ``timed_out`` is set if subprocess
    timed out (caller decides whether to abort the loop or retry).

    ``cost_usd``, ``inner_num_turns``, and ``usage`` are extracted
    from claude's JSON envelope when available — the real
    ``--output-format=json`` body carries them, but stub binaries used
    in tests may omit them, hence the ``None`` defaults."""

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
class ClaudeAgentConfig:
    """Knobs for the Claude CLI agent backend.

    Defaults err on the side of "let the agent work without nagging":
    bypassPermissions skips per-tool prompts (we are deliberately
    sandboxing via the harness's own workspace boundaries), and the
    timeout is generous because each turn may include several
    Bash + Edit + Read tool calls."""

    model: str = "sonnet"  # opus / sonnet / haiku, or full model id
    permission_mode: str = "bypassPermissions"
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    timeout_seconds: int = 600  # per-turn cap
    extra_args: tuple[str, ...] = ()  # passthrough for --max-budget-usd etc.
    claude_binary: str = "claude"  # overridable for testing


class ClaudeAgent:
    """Stateful wrapper around ``claude --print``.

    Holds a pre-allocated ``session_id`` (UUID4) and the path to the
    per-turn log directory. Each ``run_turn(prompt)`` call:

      1. spawns ``claude`` with ``--session-id`` (turn 0) or
         ``--resume`` (turn 1+),
      2. captures stdout (JSON) + stderr,
      3. writes the full transcript to ``logs/agent_turns/turn_NNN.json``
         and ``turn_NNN.txt``,
      4. returns a ``TurnResult`` for the caller to act on.

    Not thread-safe — only one turn in flight at a time.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        http_url: str,
        log_dir: Path,
        config: ClaudeAgentConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._http_url = http_url
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or ClaudeAgentConfig()
        self._session_id = session_id or str(uuid.uuid4())
        self._turn_count = 0
        self._first_turn_done = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def run_turn(self, prompt: str) -> TurnResult:
        """Run one turn. ``prompt`` becomes the user message for this turn.

        On turn 0 the session is created with ``--session-id``; on turn
        1+ we ``--resume`` that same UUID so the conversation history is
        preserved. The agent is free to use any of its allowed tools to
        read files, edit ``policy.py``, and curl the HTTP endpoints.

        Uses ``--output-format=stream-json`` so we capture every
        assistant message, tool call, and tool result *as they happen*
        — even if the turn times out, the full thought-process trace
        up to the timeout is preserved at
        ``logs/agent_turns/turn_NNN.stream.jsonl``.
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

        # Spawn claude with line-buffered stdout so the streaming
        # reader sees events as they're emitted, not in chunks at
        # process exit.
        proc = subprocess.Popen(
            cmd,
            cwd=self._workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Holders mutated by the reader thread. We grab the latest
        # ``type:"result"`` event for cost/text extraction; the stream
        # file gets every line regardless.
        last_result: dict[str, Any] | None = None
        stderr_buf: list[str] = []
        started = time.monotonic()

        assert proc.stdout is not None
        assert proc.stderr is not None
        stream_file = stream_path.open("w", encoding="utf-8")

        def _stdout_reader() -> None:
            nonlocal last_result
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    # Persist every line as it arrives — this is the
                    # "thought process" capture: thinking blocks, tool
                    # calls, tool results, assistant messages, etc.
                    stream_file.write(line)
                    stream_file.flush()
                    # Try to parse; on the result event, capture it.
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("type") == "result":
                        last_result = obj
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
            # 5s grace for the kill to land; if claude is wedged we
            # just give up and continue with returncode=124.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            returncode = 124
            timed_out = True

        duration = time.monotonic() - started

        # Drain the reader threads. Stdout/stderr are closed by the
        # kill, so the threads will hit EOF and exit.
        out_thread.join(timeout=5)
        err_thread.join(timeout=5)
        stream_file.close()
        stderr = "".join(stderr_buf)

        # Extract cost/usage/text from the captured result event (if any).
        raw_json: dict[str, Any] | None = last_result
        text_response = ""
        cost_usd: float | None = None
        inner_num_turns: int | None = None
        usage: dict[str, int] | None = None
        if last_result is not None:
            result = last_result.get("result")
            if isinstance(result, str):
                text_response = result
            raw_cost = last_result.get("total_cost_usd")
            if isinstance(raw_cost, (int, float)):
                cost_usd = float(raw_cost)
            raw_turns = last_result.get("num_turns")
            if isinstance(raw_turns, int):
                inner_num_turns = raw_turns
            raw_usage = last_result.get("usage")
            if isinstance(raw_usage, dict):
                # Keep only int-valued fields; nested dicts
                # (server_tool_use, cache_creation) are dropped to
                # match the dict[str, int] type contract.
                usage = {
                    k: int(v) for k, v in raw_usage.items()
                    if isinstance(v, int)
                }

        # Mirror the result event (or empty) to turn_NNN.json for
        # backwards-compat — old consumers expect this single-file
        # summary. Full stream is in turn_NNN.stream.jsonl.
        json_path.write_text(
            json.dumps(raw_json, indent=2) if raw_json is not None else ""
        )
        text_path.write_text(_format_transcript(
            turn_index=turn_index,
            session_id=self._session_id,
            cmd=cmd,
            duration=duration,
            returncode=returncode,
            timed_out=timed_out,
            text_response=text_response,
            stderr=stderr,
        ))

        self._first_turn_done = True
        return TurnResult(
            turn_index=turn_index,
            session_id=self._session_id,
            exit_code=returncode,
            timed_out=timed_out,
            duration_seconds=round(duration, 2),
            text=text_response,
            raw_json=raw_json,
            stderr=stderr,
            cost_usd=cost_usd,
            inner_num_turns=inner_num_turns,
            usage=usage,
        )

    def _build_command(self, prompt: str) -> list[str]:
        cfg = self._config
        # stream-json is mandatory for thought-process capture;
        # --verbose is required by claude CLI when combining
        # --print with stream-json output.
        cmd: list[str] = [
            cfg.claude_binary, "--print",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if not self._first_turn_done:
            cmd.extend(["--session-id", self._session_id])
        else:
            cmd.extend(["--resume", self._session_id])
        cmd.extend(["--permission-mode", cfg.permission_mode])
        if cfg.allowed_tools:
            cmd.extend(["--allowedTools", " ".join(cfg.allowed_tools)])
        if cfg.model:
            cmd.extend(["--model", cfg.model])
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)
        cmd.append(prompt)
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HLBENCH_URL"] = self._http_url
        # Tell the agent its own session id (so it can correlate logs
        # if it wants to). Not required for correctness.
        env["HLBENCH_SESSION_ID"] = self._session_id
        return env


def find_claude_binary() -> str | None:
    """Resolve the ``claude`` binary on PATH. Returns ``None`` if missing.

    The CLI checks this at startup so we can fail loudly rather than
    on the first turn."""
    return shutil.which("claude")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_transcript(
    *,
    turn_index: int,
    session_id: str,
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


# Public re-export so consumers can also grab a fresh UUID without
# importing uuid themselves.
def new_session_id() -> str:
    """Return a fresh UUID4 string suitable for ``--session-id``."""
    return str(uuid.uuid4())


# Used by tests to construct ClaudeAgent without claude on PATH.
_TurnResult_for_export = TurnResult  # noqa: E305
__all__ = [
    "ClaudeAgent",
    "ClaudeAgentConfig",
    "DEFAULT_ALLOWED_TOOLS",
    "TurnResult",
    "find_claude_binary",
    "new_session_id",
]

# field is exposed only because dataclass-with-default forwards may want it
_ = field
