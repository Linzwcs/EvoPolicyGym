"""Process-backed agent harness adapter."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from .session import Launch, Reply

Json = Any


class Codec(Protocol):
    """Prompt/reply framing used by a command session."""

    def request(self, turn: int, message: str) -> str: ...

    def reply(self, line: str, *, turn: int) -> Reply: ...


@dataclass(frozen=True, slots=True)
class Jsonl:
    """Line-delimited JSON framing for custom harness scripts.

    Requests are written to stdin as one JSON object per line:
    `{ "type": "prompt", "turn": N, "message": "..." }`.
    Replies are read from stdout as one JSON object per line with optional
    `text`, `stop`, and `data` fields.
    """

    def request(self, turn: int, message: str) -> str:
        return json.dumps(
            {"type": "prompt", "turn": turn, "message": message},
            sort_keys=True,
        ) + "\n"

    def reply(self, line: str, *, turn: int) -> Reply:
        try:
            body = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("agent reply must be JSON") from exc
        if not isinstance(body, dict):
            raise ValueError("agent reply must be a JSON object")

        reply_turn = body.get("turn", turn)
        if not isinstance(reply_turn, int):
            raise ValueError("agent reply turn must be an integer")

        text = body.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = json.dumps(text, sort_keys=True)

        data = body.get("data", {})
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError("agent reply data must be an object")

        return Reply(
            turn=reply_turn,
            text=text,
            stop=bool(body.get("stop", False)),
            data=dict(data),
        )


@dataclass(frozen=True, slots=True)
class Command:
    """Start a persistent stdio command as an agent harness.

    This adapter is intentionally generic. It is suitable for custom harness
    scripts and for thin wrappers around CLI agents that can preserve context
    behind a stable stdin/stdout protocol. By default, the process starts in
    the workspace root so relative paths such as `system/policy.py` and
    `feedback/submit_000` match `AGENTS.md`.
    """

    argv: Sequence[str]
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    codec: Codec = field(default_factory=Jsonl)
    name: str = "agent"

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must not be empty")
        object.__setattr__(self, "argv", tuple(str(part) for part in self.argv))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))

    def start(self, launch: Launch) -> Process:
        logs = launch.logs
        logs.mkdir(parents=True, exist_ok=True)
        workdir = self.cwd or launch.workspace
        workdir.mkdir(parents=True, exist_ok=True)

        name = _safe(self.name)
        stream = (logs / f"{name}.jsonl").open("a", encoding="utf-8")
        stderr = (logs / f"{name}.stderr.txt").open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                tuple(self.argv),
                cwd=str(workdir),
                env=self._env(launch),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
            )
        except Exception:
            stream.close()
            stderr.close()
            raise

        return Process(
            proc=proc,
            codec=self.codec,
            stream=stream,
            stderr=stderr,
            name=name,
        )

    def _env(self, launch: Launch) -> dict[str, str]:
        values = dict(os.environ)
        values.update(self.env)
        values.update(launch.environ())
        return values


@dataclass(slots=True)
class Process:
    """Persistent subprocess-backed agent session."""

    proc: subprocess.Popen[str]
    codec: Codec
    stream: TextIO
    stderr: TextIO
    name: str = "agent"
    turn: int = 0
    closed: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}:{self.proc.pid}"

    def step(self, message: str) -> Reply:
        if self.closed:
            raise RuntimeError("agent session is closed")
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("agent command was not started with pipes")

        turn = self.turn
        frame = self.codec.request(turn, message)
        self._record("prompt", {"turn": turn, "message": message})
        try:
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("agent command exited before receiving prompt") from exc

        line = self.proc.stdout.readline()
        if line == "":
            code = self.proc.poll()
            raise RuntimeError(f"agent command exited before reply: {code}")

        self._record("stdout", {"turn": turn, "line": line.rstrip("\n")})
        reply = self.codec.reply(line, turn=turn)
        self._record(
            "reply",
            {
                "turn": reply.turn,
                "text": reply.text,
                "stop": reply.stop,
                "data": dict(reply.data),
            },
        )
        self.turn += 1
        return reply

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=1.0)
        finally:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            self.stream.close()
            self.stderr.close()

    def _record(self, kind: str, body: Mapping[str, Json]) -> None:
        self.stream.write(json.dumps({"kind": kind, **body}, sort_keys=True) + "\n")
        self.stream.flush()


def _safe(value: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return name or "agent"
