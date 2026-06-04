"""Moonshot Kimi Code CLI harness adapter."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .session import Launch, Reply

Json = Any


@dataclass(frozen=True, slots=True)
class Kimi:
    """Drive Kimi Code turns with print mode and resumed sessions.

    Kimi Code is non-interactive at the process level: each `step` spawns one
    command in the workspace root. The logical session is preserved by reading
    Kimi's session id from the first stream and passing it to `-S` on later
    turns. If no id is exposed, later turns fall back to `-C` from the same
    workspace.
    """

    binary: str = "kimi"
    model: str = ""
    args: Sequence[str] = field(default_factory=tuple)
    name: str = "kimi"
    timeout: float | None = None
    index: str | Path = ""

    def __post_init__(self) -> None:
        if not self.binary:
            raise ValueError("binary must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))
        if self.index:
            object.__setattr__(self, "index", Path(self.index))

    def start(self, launch: Launch) -> KimiSession:
        index = self.index if isinstance(self.index, Path) else None
        return KimiSession(
            launch=launch,
            binary=self.binary,
            model=self.model,
            args=tuple(self.args),
            name=_safe(self.name),
            timeout=self.timeout,
            index=index,
        )


@dataclass(slots=True)
class KimiSession:
    """Logical Kimi Code session backed by per-turn subprocesses."""

    launch: Launch
    binary: str
    model: str = ""
    args: tuple[str, ...] = ()
    name: str = "kimi"
    timeout: float | None = None
    index: Path | None = None
    turn: int = 0
    session: str | None = None
    label: str = field(default_factory=lambda: str(uuid.uuid4()))
    closed: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}:{self.session or self.label}"

    def step(self, message: str) -> Reply:
        if self.closed:
            raise RuntimeError("kimi session is closed")

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
            self.session = _session(events) or _lookup(self.index, self.launch.workspace)
        text = _text(events) or raw_text.strip()
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
                "kimi_session": self.session,
            },
        )

    def close(self) -> None:
        self.closed = True

    def _command(self, message: str) -> list[str]:
        cmd = [self.binary]
        if self.session is not None:
            cmd.extend(["-S", self.session])
        elif self.turn > 0:
            cmd.append("-C")
        cmd.extend(["--output-format", "stream-json"])
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.extend(self.args)
        cmd.extend(["-p", message])
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


def _session(events: list[dict[str, Json]]) -> str | None:
    for event in events:
        found = _id(event)
        if found is not None:
            return found
    return None


def _id(value: Json) -> str | None:
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if key in {"sessionId", "sessionID", "session_id"} and isinstance(raw, str):
                if raw:
                    return raw
            if key == "id" and isinstance(raw, str) and raw.startswith("session_"):
                return raw
            nested = _id(raw)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _id(item)
            if nested is not None:
                return nested
    return None


def _lookup(index: Path | None, workspace: Path) -> str | None:
    source = index or Path.home() / ".kimi-code" / "session_index.jsonl"
    if not source.is_file():
        return None
    workdir = str(workspace.resolve())
    found: list[str] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, Mapping):
            continue
        session = item.get("sessionId") or item.get("session_id")
        raw_workdir = item.get("workDir") or item.get("workdir")
        if isinstance(session, str) and isinstance(raw_workdir, str):
            if str(Path(raw_workdir).resolve()) == workdir:
                found.append(session)
    return found[-1] if found else None


def _text(events: list[dict[str, Json]]) -> str:
    bits: list[str] = []
    for event in events:
        _collect(bits, event)
        for key in ("payload", "message", "data", "result"):
            value = event.get(key)
            if isinstance(value, Mapping):
                _collect(bits, value)
    deduped: list[str] = []
    for bit in bits:
        if not deduped or deduped[-1] != bit:
            deduped.append(bit)
    return "\n".join(deduped)


def _collect(bits: list[str], value: Mapping[str, Json]) -> None:
    for key in ("result", "message", "text", "content", "last_agent_message"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            bits.append(raw)


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
    return name or "kimi"
