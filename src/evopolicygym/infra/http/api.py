"""Framework-neutral agent HTTP API facade."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from ...core import (
    Caps,
    Log,
    Pool,
    Run,
    Runtime,
    Snap,
    Store,
    Task,
    Verdict,
)
from ...core import (
    Submit as SubmitRecord,
)
from ...judge import JudgeClose, JudgeSubmit, Limits
from ...protocol import PROTOCOL

Clock = Callable[[], datetime]
Json = Any


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _NullLog:
    def emit(self, event: str, **data: Json) -> None:
        return None


@dataclass(frozen=True, slots=True)
class InfoResponse:
    protocol_version: str
    state: dict[str, Json]
    env_meta: dict[str, Json]
    episode_budget: int
    min_episodes_per_submit: int
    max_episodes_per_submit: int
    resource_limits: dict[str, Json] = field(default_factory=dict)
    allowed_imports: tuple[str, ...] = ()
    denied_imports: tuple[str, ...] = ()

    def body(self) -> dict[str, Json]:
        return {
            "protocol_version": self.protocol_version,
            "state": self.state,
            "env_meta": self.env_meta,
            "episode_budget": self.episode_budget,
            "min_episodes_per_submit": self.min_episodes_per_submit,
            "max_episodes_per_submit": self.max_episodes_per_submit,
            "resource_limits": self.resource_limits,
            "allowed_imports": list(self.allowed_imports),
            "denied_imports": list(self.denied_imports),
        }


@dataclass(frozen=True, slots=True)
class TaskResponse:
    text: str
    media: str = "text/markdown"


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    env_instances: Sequence[int] | str


@dataclass(frozen=True, slots=True)
class SubmitResponse:
    code: int
    submit_id: int
    status: str
    summary: dict[str, Json]

    def body(self) -> dict[str, Json]:
        return {
            "submit_id": self.submit_id,
            "status": self.status,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    code: int
    status: str
    message: str

    def body(self) -> dict[str, Json]:
        return {"status": self.status, "error": {"message": self.message}}


@dataclass(slots=True)
class Service:
    """Agent-facing API without binding to a concrete web framework."""

    run: Run
    task: Task
    train: Pool
    store: Store
    runtime: Runtime
    limits: Limits
    valid: Pool | None = None
    final: Pool | None = None
    task_text: str = ""
    resource_limits: dict[str, Json] = field(default_factory=dict)
    caps: Caps = field(default_factory=Caps)
    allowed_imports: tuple[str, ...] = ()
    denied_imports: tuple[str, ...] = ()
    protocol: str = PROTOCOL
    log: Log = field(default_factory=lambda: _NullLog())
    clock: Clock = _now
    submits: int = 0
    _snaps: list[Snap] = field(default_factory=list, init=False, repr=False)
    _submit_lock: Any = field(default_factory=Lock, init=False, repr=False)

    def info(self) -> InfoResponse:
        return InfoResponse(
            protocol_version=self.protocol,
            state={
                "remaining_budget": self.run.budget.left,
                "n_submits": self.submits,
                "is_finalized": not self.run.alive(),
            },
            env_meta=_env_meta(self.task, self.caps),
            episode_budget=self.run.budget.limit,
            min_episodes_per_submit=self.limits.minimum,
            max_episodes_per_submit=self.limits.maximum,
            resource_limits=dict(self.resource_limits),
            allowed_imports=tuple(self.allowed_imports),
            denied_imports=tuple(self.denied_imports),
        )

    def task_doc(self) -> TaskResponse:
        return TaskResponse(_task_text(self.task, self.task_text, self.limits))

    def submit(self, request: SubmitRequest) -> SubmitResponse | ErrorResponse:
        with self._submit_lock:
            return self._submit(request)

    def _submit(self, request: SubmitRequest) -> SubmitResponse | ErrorResponse:
        self._sync_submit_index()
        try:
            cases = parse_cases(request.env_instances)
        except ValueError as exc:
            submit_index = self.submits
            self.log.emit(
                "submit.reject",
                submit_index=submit_index,
                status=Verdict.budget_invalid.value,
                reason=str(exc),
                remaining_budget=self.run.budget.left,
            )
            self.submits = submit_index + 1
            return ErrorResponse(400, Verdict.budget_invalid.value, str(exc))

        submit = SubmitRecord(index=self.submits, cases=cases)
        self.log.emit(
            "submit.start",
            submit_index=submit.index,
            cases=list(cases),
            remaining_budget=self.run.budget.left,
        )
        outcome = JudgeSubmit(self.store, self.runtime, clock=self.clock)(
            self.run,
            submit,
            self.task,
            self.train,
            self.limits,
        )
        self.run = outcome.run
        if outcome.snap is not None:
            self._snaps.append(outcome.snap)
        self.submits += 1

        code = 400 if outcome.feed.verdict.rejected else 200
        self.log.emit(
            "submit.finish",
            submit_index=submit.index,
            status=outcome.feed.verdict.value,
            code=code,
            cost=outcome.feed.cost,
            remaining_budget=self.run.budget.left,
            snap=outcome.snap.index if outcome.snap is not None else None,
            steps=[
                {
                    "phase": step.phase.value,
                    "verdict": step.verdict.value if step.verdict is not None else None,
                }
                for step in outcome.steps
            ],
            wall_time_seconds=round(outcome.report.wall, 6),
        )
        self._close_if_done()
        return SubmitResponse(
            code=code,
            submit_id=submit.index,
            status=outcome.feed.verdict.value,
            summary=outcome.summary,
        )

    def _sync_submit_index(self) -> None:
        next_submit_index = getattr(self.store, "next_submit_index", None)
        if not callable(next_submit_index):
            return
        current = self.submits
        try:
            available = int(next_submit_index(current))
        except Exception as exc:  # noqa: BLE001 - index sync must not hide submits.
            self.log.emit(
                "submit.index_sync_error",
                submit_index=current,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return
        if available <= current:
            return
        self.log.emit(
            "submit.index_skip",
            from_submit_index=current,
            to_submit_index=available,
            reason="submit artifacts already exist",
        )
        self.submits = available

    def _close_if_done(self) -> None:
        if not self.run.exhausted or not self.run.alive():
            return
        if self.valid is None or self.final is None:
            return

        self.log.emit(
            "run.auto_close.start",
            n_snaps=len(self._snaps),
            budget_used=self.run.budget.used,
        )
        outcome = JudgeClose(self.store, self.runtime)(
            self.run,
            tuple(self._snaps),
            self.task,
            self.valid,
            self.final,
        )
        self.run = outcome.run
        self.log.emit(
            "run.auto_close.finish",
            status=outcome.status.value,
            best_submit_index=outcome.pick.best,
            n_validation=len(outcome.vals),
            final_score=outcome.final.score.primary if outcome.final is not None else None,
        )


def parse_cases(value: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(value, str):
        return _parse_spec(value)
    cases = tuple(value)
    if any(not isinstance(item, int) or item < 0 for item in cases):
        raise ValueError("env_instances must be non-negative integers")
    return cases


def _parse_spec(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise ValueError("env_instances spec is empty")
    cases: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError("env_instances spec contains an empty token")
        if "-" in token:
            lo, hi = _range(token)
            cases.extend(range(lo, hi + 1))
        else:
            cases.append(_number(token))
    return tuple(cases)


def _range(token: str) -> tuple[int, int]:
    parts = token.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid range token: {token}")
    lo = _number(parts[0])
    hi = _number(parts[1])
    if hi < lo:
        raise ValueError(f"invalid range token: {token}")
    return lo, hi


def _number(token: str) -> int:
    if not token.isdigit():
        raise ValueError(f"invalid env instance token: {token}")
    return int(token)


def _env_meta(task: Task, caps: Caps | None = None) -> dict[str, Json]:
    meta: dict[str, Json] = {
        "env": task.name,
        "env_version": task.version,
        "n_env_instances": task.cases,
        "max_episode_steps": task.steps,
        "obs_space": task.obs,
        "action_space": task.act,
        "obs_storage": task.storage,
    }
    if task.rewards:
        meta["reward_components"] = task.rewards
    if caps is not None and caps.any:
        meta["artifacts"] = caps.body()
    return meta


def _task_text(task: Task, text: str = "", limits: Limits | None = None) -> str:
    brief = text.strip() or f"# {task.name}\n\nVersion: {task.version}"
    return "\n\n".join((brief, _contract(task, limits))).rstrip() + "\n"


def _contract(task: Task, limits: Limits | None) -> str:
    bounds = "See `GET /info` for the current submit bounds."
    if limits is not None:
        bounds = (
            f"Submit between {limits.minimum} and {limits.maximum} visible train "
            "case IDs per accepted request."
        )
    rewards = "None declared."
    if task.rewards:
        rewards = _json(task.rewards)
    return f"""## Policy Contract

Create `system/policy.py` with this top-level class:

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs): ...
```

`__init__` is called once per submit. `reset(episode_index)` is called before
each episode in that submit. `act(obs)` receives one observation and must return
one action compatible with the action space below.

## Policy Input And Output

- Environment: `{task.name}` version `{task.version}`
- Visible train cases: integer IDs `0` through `{task.cases - 1}`
- Max episode steps: `{task.steps}`
- Observation storage: `{task.storage}`
- Reward components: {rewards}
- Input: `act(obs)` receives one observation matching `obs_space` / `Task.obs`.
- Output: `act(obs)` must return one action matching `action_space` / `Task.act`.

Input observation space (`obs_space` / `Task.obs`):

```json
{_json(task.obs)}
```

Output action space (`action_space` / `Task.act`):

```json
{_json(task.act)}
```

## Submit Notes

{bounds} Submit by posting JSON to `/submit`, for example
`{{"env_instances": [0, 1, 2, 3]}}` or `{{"env_instances": "0-3"}}`.
Read returned summaries under the feedback directory after each submit.
"""


def _json(value: Json) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
