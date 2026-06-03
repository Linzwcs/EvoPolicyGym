"""Serial suite planning and reporting."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Agent, Spec
from .layout import root as run_root
from .layout import slug

Json = Any


@dataclass(frozen=True, slots=True)
class Job:
    """One expanded benchmark run inside a suite."""

    index: int
    repeat: int
    env: str
    agent: str
    spec: Spec


@dataclass(frozen=True, slots=True)
class Result:
    """Serializable summary for one suite job."""

    job: Job
    done: bool
    reason: str
    root: Path
    run: str
    session: str
    submits: int
    check_ok: bool | None = None
    check_issues: tuple[str, ...] = ()
    error: str = ""

    @classmethod
    def from_summary(cls, job: Job, data: Mapping[str, Json]) -> Result:
        return cls(
            job=job,
            done=bool(data.get("done", False)),
            reason=str(data.get("reason", "")),
            root=Path(str(data.get("root", job.spec.root))),
            run=str(data.get("run", job.spec.run_key)),
            session=str(data.get("session", "")),
            submits=_int(data.get("submits", 0), "submits"),
        )

    @classmethod
    def from_error(cls, job: Job, exc: BaseException) -> Result:
        return cls(
            job=job,
            done=False,
            reason="framework_error",
            root=job.spec.root,
            run=job.spec.run_key,
            session="",
            submits=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    @property
    def category(self) -> str:
        if self.error:
            return "framework_error"
        if not self.done:
            return self.reason or "failed"
        if self.check_ok is False:
            return "artifact_check_failed"
        return "completed"

    @property
    def passed(self) -> bool:
        return self.done and self.check_ok is not False and not self.error

    def checked(self, ok: bool, issues: Sequence[str] = ()) -> Result:
        return replace(self, check_ok=ok, check_issues=tuple(issues))

    def record(self) -> dict[str, Json]:
        return {
            "index": self.job.index,
            "repeat": self.job.repeat,
            "env": self.job.env,
            "agent": self.job.agent,
            "root": str(self.root),
            "run": self.run,
            "done": self.done,
            "reason": self.reason,
            "session": self.session,
            "submits": self.submits,
            "category": self.category,
            "check": {
                "ok": self.check_ok,
                "issues": list(self.check_issues),
            },
            "error": self.error or None,
        }


@dataclass(frozen=True, slots=True)
class Suite:
    """A serial list of benchmark jobs expanded from a suite config."""

    root: Path
    jobs: tuple[Job, ...]
    repeats: int = 1
    name: str = "suite"
    concurrency: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if self.repeats <= 0:
            raise ValueError("suite.repeats must be positive")
        if self.concurrency <= 0:
            raise ValueError("suite.jobs must be positive")
        if not self.name:
            raise ValueError("suite.name must not be empty")

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        source = Path(path)
        if source.suffix == ".toml":
            data = tomllib.loads(source.read_text(encoding="utf-8"))
        else:
            data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("suite config root must be an object")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Json]) -> Suite:
        section = _section(data, "suite")
        root = _path(_get(data, section, "root", None), "suite.root")
        repeats = _positive(_get(data, section, "repeats", 1), "suite.repeats")
        concurrency = _positive(_get(data, section, "jobs", 1), "suite.jobs")
        name = str(_get(data, section, "name", "suite"))
        runs = _items(data.get("run", ()), "run")
        agents = _items(data.get("agent", ()), "agent")
        server = _section(data, "server")

        if not runs:
            raise ValueError("suite requires at least one run")
        if not agents:
            raise ValueError("suite requires at least one agent")

        jobs: list[Job] = []
        index = 0
        for repeat in range(repeats):
            for run in runs:
                for agent_data in agents:
                    agent = Agent.from_mapping(agent_data)
                    env = str(run.get("env", "toy"))
                    label = _label(agent)
                    model = str(run.get("model") or label)
                    base = str(run.get("key") or f"{env}_{model}")
                    leaf = f"{index:03d}_{slug(base)}_r{repeat:02d}"
                    run_data = dict(run)
                    run_data["root"] = str(run_root(root, model=model, env=env, exp=leaf))
                    run_data["key"] = leaf
                    run_data.setdefault("model", model)
                    run_data["exp"] = leaf
                    spec = Spec.from_mapping(
                        {
                            "run": run_data,
                            "agent": agent_data,
                            "server": server,
                        }
                    )
                    jobs.append(
                        Job(
                            index=index,
                            repeat=repeat,
                            env=spec.env,
                            agent=label,
                            spec=spec,
                        )
                    )
                    index += 1

        return cls(
            root=root,
            jobs=tuple(jobs),
            repeats=repeats,
            name=name,
            concurrency=concurrency,
        )

    def report(self, results: Sequence[Result]) -> dict[str, Json]:
        records = [result.record() for result in results]
        passed = sum(1 for result in results if result.passed)
        failed = len(results) - passed
        return {
            "name": self.name,
            "root": str(self.root),
            "repeats": self.repeats,
            "jobs_configured": self.concurrency,
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "done": failed == 0 and len(results) == len(self.jobs),
            "by_reason": _counts(result.reason for result in results),
            "by_category": _counts(result.category for result in results),
            "jobs": records,
        }

    def write(self, results: Sequence[Result], path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.root / "suite.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.report(results), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def load(path: str | Path) -> Suite:
    """Load and expand a suite config."""

    return Suite.load(path)


def _section(data: Mapping[str, Json], name: str) -> Mapping[str, Json]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _get(
    data: Mapping[str, Json],
    section: Mapping[str, Json],
    name: str,
    default: Json,
) -> Json:
    if name in section:
        return section[name]
    return data.get(name, default)


def _items(value: Json, name: str) -> tuple[Mapping[str, Json], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an object or object array")
    items: list[Mapping[str, Json]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be an object or object array")
        items.append(item)
    return tuple(items)


def _path(value: Json, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required")
    raw = str(value)
    if not raw:
        raise ValueError(f"{name} must not be empty")
    return Path(raw)


def _int(value: Json, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive(value: Json, name: str) -> int:
    number = _int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _label(agent: Agent) -> str:
    if agent.name and agent.name != "agent":
        return agent.name
    if agent.model:
        return f"{agent.kind}_{agent.model}"
    return agent.kind
