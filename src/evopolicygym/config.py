"""Typed run configuration for EvoPolicyGym CLI runs."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .layout import root as run_root

Json = Any


@dataclass(frozen=True, slots=True)
class Agent:
    kind: str = "command"
    argv: tuple[str, ...] = ()
    name: str = "agent"
    limit: int | None = None
    retries: int = 0
    retry_backoff: float = 1.0
    binary: str = ""
    model: str = ""
    sandbox: str = "workspace-write"
    approval: str = "never"
    bypass: bool = False
    permission: str = "bypassPermissions"
    tools: tuple[str, ...] = ()
    args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"command", "codex", "claude", "kimi"}:
            raise ValueError(f"unsupported agent kind: {self.kind}")
        object.__setattr__(self, "argv", tuple(str(item) for item in self.argv))
        object.__setattr__(self, "tools", tuple(str(item) for item in self.tools))
        object.__setattr__(self, "args", tuple(str(item) for item in self.args))
        if self.limit is not None and self.limit <= 0:
            raise ValueError("agent.limit must be positive")
        if self.retries < 0:
            raise ValueError("agent.retries must be non-negative")
        if self.retry_backoff < 0:
            raise ValueError("agent.retry_backoff must be non-negative")
        if not self.name:
            raise ValueError("agent.name must not be empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Json] | None = None) -> Agent:
        values = dict(data or {})
        return cls(
            kind=str(values.get("kind", "command")),
            argv=_strings(values.get("argv", ()), "agent.argv"),
            name=str(values.get("name", "agent")),
            limit=_optional_int(values.get("limit", None), "agent.limit"),
            retries=_int(values.get("retries", 0), "agent.retries"),
            retry_backoff=_float(values.get("retry_backoff", 1.0), "agent.retry_backoff"),
            binary=str(values.get("binary", "")),
            model=str(values.get("model", "")),
            sandbox=str(values.get("sandbox", "workspace-write")),
            approval=str(values.get("approval", "never")),
            bypass=_bool(values.get("bypass", False), "agent.bypass"),
            permission=str(values.get("permission", "bypassPermissions")),
            tools=_strings(values.get("tools", ()), "agent.tools"),
            args=_strings(values.get("args", ()), "agent.args"),
        )


@dataclass(frozen=True, slots=True)
class Server:
    bind: str = "127.0.0.1"
    port: int = 0

    def __post_init__(self) -> None:
        if self.port < 0:
            raise ValueError("server.port must be non-negative")
        if not self.bind:
            raise ValueError("server.bind must not be empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Json] | None = None) -> Server:
        values = dict(data or {})
        return cls(
            bind=str(values.get("bind", "127.0.0.1")),
            port=_int(values.get("port", 0), "server.port"),
        )


@dataclass(frozen=True, slots=True)
class Spec:
    env: str
    budget: int
    bulk: bool = False
    root: Path | None = None
    runs: Path | None = None
    data: Path | None = None
    key: str = ""
    model: str = "agent"
    exp: str = "default"
    minimum: int = 1
    maximum: int | None = None
    valid_size: int | None = None
    final_size: int | None = None
    agent: Agent = Agent()
    server: Server = Server()

    def __post_init__(self) -> None:
        if self.runs is not None:
            object.__setattr__(self, "runs", Path(self.runs))
        root = Path(self.root) if self.root is not None else None
        if root is None:
            if self.runs is None:
                raise ValueError("run.root or run.runs is required")
            root = run_root(self.runs, model=self.model, env=self.env, exp=self.exp)
        object.__setattr__(self, "root", root)
        if self.data is not None:
            object.__setattr__(self, "data", Path(self.data))
        if not self.env:
            raise ValueError("run.env must not be empty")
        if self.budget < 0:
            raise ValueError("run.budget must be non-negative")
        if self.agent.limit is None:
            object.__setattr__(
                self,
                "agent",
                replace(self.agent, limit=max(1, self.budget)),
            )
        if self.minimum <= 0:
            raise ValueError("run.minimum must be positive")
        if self.maximum is not None and self.maximum <= 0:
            raise ValueError("run.maximum must be positive")
        if self.valid_size is not None and self.valid_size < 0:
            raise ValueError("run.valid_size must be non-negative")
        if self.final_size is not None and self.final_size < 0:
            raise ValueError("run.final_size must be non-negative")

    @property
    def run_key(self) -> str:
        return self.key or self.root.name or "run"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Json]) -> Spec:
        run = _section(data, "run")
        env = _get(data, run, "env", "toy")
        root = _get(data, run, "root", None)
        runs = _get(data, run, "runs", None)
        budget = _get(data, run, "budget", None)
        if root is not None and runs is not None:
            raise ValueError("run.root and run.runs are mutually exclusive")
        if budget is None:
            raise ValueError("run.budget is required")
        model = str(_get(data, run, "model", "agent"))
        exp = str(_get(data, run, "exp_id", _get(data, run, "exp", "default")))

        return cls(
            env=str(env),
            bulk=_bool(_get(data, run, "bulk", False), "run.bulk"),
            root=_optional_path(root, "run.root"),
            runs=_optional_path(runs, "run.runs"),
            data=_optional_path(_get(data, run, "data", None), "run.data"),
            key=str(_get(data, run, "key", "")),
            model=model,
            exp=exp,
            budget=_int(budget, "run.budget"),
            minimum=_int(_get(data, run, "minimum", 1), "run.minimum"),
            maximum=_optional_int(_get(data, run, "maximum", None), "run.maximum"),
            valid_size=_optional_int(_get(data, run, "valid_size", None), "run.valid_size"),
            final_size=_optional_int(_get(data, run, "final_size", None), "run.final_size"),
            agent=Agent.from_mapping(_section(data, "agent")),
            server=Server.from_mapping(_section(data, "server")),
        )


def load(path: str | Path) -> Spec:
    """Load a run spec from a JSON or TOML file."""

    source = Path(path)
    if source.suffix == ".toml":
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("config root must be an object")
    return Spec.from_mapping(data)


def overlay(spec: Spec, **values: Json) -> Spec:
    """Return a copy of `spec` with non-None values applied."""

    updates = {key: value for key, value in values.items() if value is not None}
    if not updates:
        return spec
    return replace(spec, **updates)


def _section(data: Mapping[str, Json], name: str) -> Mapping[str, Json]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _get(data: Mapping[str, Json], section: Mapping[str, Json], name: str, default: Json) -> Json:
    if name in section:
        return section[name]
    return data.get(name, default)


def _int(value: Json, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_int(value: Json, name: str) -> int | None:
    if value is None:
        return None
    return _int(value, name)


def _float(value: Json, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_path(value: Json, name: str) -> Path | None:
    if value is None:
        return None
    raw = str(value)
    if not raw:
        raise ValueError(f"{name} must not be empty")
    return Path(raw)


def _bool(value: Json, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _strings(value: Json, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a string array")
    return tuple(str(item) for item in value)
