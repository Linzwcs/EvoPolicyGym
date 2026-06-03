"""Claude Code CLI harness adapter."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .session import Launch, Reply

Json = Any
DEFAULT_TOOLS = ("Bash", "Read", "Edit", "Write", "Glob", "Grep")


@dataclass(frozen=True, slots=True)
class Claude:
    """Drive Claude Code turns with print mode and resumed sessions.

    Claude Code is non-interactive at the process level: each `step` spawns one
    command. The logical session is preserved by reading the Claude session id
    from the first stream and passing it to `--resume` on later turns. If a
    stream does not expose an id, later turns fall back to `--continue` inside
    the run directory.
    """

    binary: str = "claude"
    model: str = ""
    permission: str = "bypassPermissions"
    tools: Sequence[str] = DEFAULT_TOOLS
    args: Sequence[str] = field(default_factory=tuple)
    name: str = "claude"
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.binary:
            raise ValueError("binary must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "tools", tuple(str(tool) for tool in self.tools))
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))

    def start(self, launch: Launch) -> ClaudeSession:
        return ClaudeSession(
            launch=launch,
            binary=self.binary,
            model=self.model,
            permission=self.permission,
            tools=tuple(self.tools),
            args=tuple(self.args),
            name=_safe(self.name),
            timeout=self.timeout,
        )


@dataclass(slots=True)
class ClaudeSession:
    """Logical Claude Code session backed by per-turn subprocesses."""

    launch: Launch
    binary: str
    model: str = ""
    permission: str = "bypassPermissions"
    tools: tuple[str, ...] = DEFAULT_TOOLS
    args: tuple[str, ...] = ()
    name: str = "claude"
    timeout: float | None = None
    turn: int = 0
    session: str | None = None
    label: str = field(default_factory=lambda: str(uuid.uuid4()))
    closed: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}:{self.session or self.label}"

    def step(self, message: str) -> Reply:
        if self.closed:
            raise RuntimeError("claude session is closed")

        turn = self.turn
        self.launch.workspace.mkdir(parents=True, exist_ok=True)
        root = self.launch.logs / f"{self.name}_turns"
        root.mkdir(parents=True, exist_ok=True)
        prompt_path = root / f"turn_{turn:03d}.prompt.txt"
        stream_path = root / f"turn_{turn:03d}.stream.jsonl"
        stderr_path = root / f"turn_{turn:03d}.stderr.txt"
        text_path = root / f"turn_{turn:03d}.txt"
        command_path = root / f"turn_{turn:03d}.command.json"

        cmd = self._command(message)
        prompt_path.write_text(message, encoding="utf-8")
        command_path.write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")

        started = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.launch.workspace),
            env=self._env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
            code = 124

        duration = time.monotonic() - started
        stream_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        events, raw_text = _events(stdout)
        if self.session is None:
            self.session = _session(events)
        result = _result(events)
        text = _text(result, events) or raw_text.strip()
        text_path.write_text(
            _transcript(
                turn=turn,
                key=self.key,
                command=cmd,
                code=code,
                timed_out=timed_out,
                duration=duration,
                text=text,
                stderr=stderr,
            ),
            encoding="utf-8",
        )

        self.turn += 1
        return Reply(
            turn=turn,
            text=text,
            stop=timed_out or code != 0,
            data={
                "exit_code": code,
                "timed_out": timed_out,
                "duration_seconds": round(duration, 3),
                "claude_session": self.session,
                "cost_usd": _number(result, "total_cost_usd"),
                "inner_turns": _integer(result, "num_turns"),
            },
        )

    def close(self) -> None:
        self.closed = True

    def _command(self, message: str) -> list[str]:
        cmd = [self.binary, "--print", "--output-format", "stream-json", "--verbose"]
        if self.turn > 0:
            if self.session is not None:
                cmd.extend(["--resume", self.session])
            else:
                cmd.append("--continue")
        if self.permission:
            cmd.extend(["--permission-mode", self.permission])
        if self.tools:
            cmd.extend(["--allowedTools", ",".join(self.tools)])
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend(self.args)
        cmd.append(message)
        return cmd

    def _env(self) -> dict[str, str]:
        values = dict(os.environ)
        values.update(self.launch.environ())
        values["EVOPOLICYGYM_AGENT"] = self.name
        values["EVOPOLICYGYM_SESSION"] = self.session or self.label
        return values


def _events(stdout: str) -> tuple[list[dict[str, Json]], str]:
    events: list[dict[str, Json]] = []
    raw: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            raw.append(line)
            continue
        if isinstance(item, dict):
            events.append(item)
        else:
            raw.append(line)
    return events, "\n".join(raw)


def _result(events: list[dict[str, Json]]) -> dict[str, Json]:
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return {}


def _session(events: list[dict[str, Json]]) -> str | None:
    for event in events:
        found = _id(event)
        if found is not None:
            return found
        for key in ("message", "payload", "result", "session"):
            value = event.get(key)
            if isinstance(value, Mapping):
                found = _id(value)
                if found is not None:
                    return found
    return None


def _id(value: Mapping[str, Json]) -> str | None:
    for key in ("session_id", "sessionId", "id"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def _text(result: Mapping[str, Json], events: list[dict[str, Json]]) -> str:
    raw = result.get("result")
    if isinstance(raw, str):
        return raw
    bits: list[str] = []
    for event in events:
        _collect(bits, event)
        for key in ("message", "payload"):
            value = event.get(key)
            if isinstance(value, Mapping):
                _collect(bits, value)
    return "\n".join(bits)


def _collect(bits: list[str], value: Mapping[str, Json]) -> None:
    for key in ("text", "content", "message"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            bits.append(raw)


def _number(value: Mapping[str, Json], key: str) -> float | None:
    raw = value.get(key)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _integer(value: Mapping[str, Json], key: str) -> int | None:
    raw = value.get(key)
    if isinstance(raw, int):
        return raw
    return None


def _transcript(
    *,
    turn: int,
    key: str,
    command: list[str],
    code: int,
    timed_out: bool,
    duration: float,
    text: str,
    stderr: str,
) -> str:
    return "\n".join(
        (
            f"# turn={turn} key={key}",
            f"# exit_code={code} timed_out={timed_out} duration_seconds={duration:.3f}",
            f"# command={json.dumps(command)}",
            "",
            "## response",
            text or "(empty)",
            "",
            "## stderr",
            stderr.strip() or "(empty)",
            "",
        )
    )


def _safe(value: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return name or "claude"
