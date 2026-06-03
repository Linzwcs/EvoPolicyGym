"""OpenAI Codex CLI harness adapter."""

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


@dataclass(frozen=True, slots=True)
class Codex:
    """Drive Codex CLI turns with `codex exec` and `exec resume`.

    Codex is non-interactive at the process level: each `step` spawns one
    command. The logical session is preserved by scraping the Codex session id
    from the first JSON event stream and passing it to `codex exec resume` on
    later turns.
    """

    binary: str = "codex"
    model: str = ""
    sandbox: str = "workspace-write"
    approval: str = "never"
    bypass: bool = False
    args: Sequence[str] = field(default_factory=tuple)
    name: str = "codex"
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.binary:
            raise ValueError("binary must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))

    def start(self, launch: Launch) -> CodexSession:
        return CodexSession(
            launch=launch,
            binary=self.binary,
            model=self.model,
            sandbox=self.sandbox,
            approval=self.approval,
            bypass=self.bypass,
            args=tuple(self.args),
            name=_safe(self.name),
            timeout=self.timeout,
        )


@dataclass(slots=True)
class CodexSession:
    """Logical Codex session backed by per-turn subprocesses."""

    launch: Launch
    binary: str
    model: str = ""
    sandbox: str = "workspace-write"
    approval: str = "never"
    bypass: bool = False
    args: tuple[str, ...] = ()
    name: str = "codex"
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
            raise RuntimeError("codex session is closed")

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
                "codex_session": self.session,
            },
        )

    def close(self) -> None:
        self.closed = True

    def _command(self, message: str) -> list[str]:
        cmd = [self.binary, "exec"]
        cmd.append("--json")
        cmd.extend(["--cd", str(self.launch.workspace)])
        if self.bypass:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.sandbox:
            cmd.extend(["--sandbox", self.sandbox])
        if self.approval and not self.bypass:
            cmd.extend(["--config", f"approval_policy={json.dumps(self.approval)}"])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.session is not None:
            cmd.append("resume")
        cmd.extend(self.args)
        if self.session is not None:
            cmd.append(self.session)
        cmd.append(message)
        return cmd

    def _env(self) -> dict[str, str]:
        values = dict(os.environ)
        values.update(self.launch.environ())
        values["EVOPOLICYGYM_AGENT"] = self.name
        values["EVOPOLICYGYM_SESSION"] = self.label
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
        for key in ("payload", "msg", "thread", "session"):
            value = event.get(key)
            if isinstance(value, Mapping):
                found = _id(value)
                if found is not None:
                    return found
    return None


def _id(value: Mapping[str, Json]) -> str | None:
    for key in ("id", "session_id", "thread_id", "conversation_id"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def _text(events: list[dict[str, Json]]) -> str:
    bits: list[str] = []
    for event in events:
        _collect(bits, event)
        for key in ("payload", "msg", "message"):
            value = event.get(key)
            if isinstance(value, Mapping):
                _collect(bits, value)
    deduped: list[str] = []
    for bit in bits:
        if not deduped or deduped[-1] != bit:
            deduped.append(bit)
    return "\n".join(deduped)


def _collect(bits: list[str], value: Mapping[str, Json]) -> None:
    for key in ("text", "content", "message", "last_agent_message"):
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
    return name or "codex"
