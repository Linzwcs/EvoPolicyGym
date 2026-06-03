"""Core EvoPolicyGym vocabulary.

These objects describe the protocol, not a concrete HTTP server or file
layout. They intentionally avoid importing JSON, Path, subprocess, or
Gym-specific types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from math import inf
from typing import Any


class RunState(StrEnum):
    open = "open"
    closing = "closing"
    closed = "closed"
    failed = "failed"


class OutcomeStatus(StrEnum):
    completed = "completed"
    no_ok_submit = "no_ok_submit"
    error = "error"


class Phase(StrEnum):
    """Submit lifecycle phases from the protocol."""

    request = "request"
    snapshot = "snapshot"
    validate = "validate"
    compile = "compile"
    init = "init"
    execute = "execute"
    commit = "commit"


class Verdict(StrEnum):
    ok = "ok"
    budget_invalid = "budget_invalid"
    invalid_case = "invalid_env_instance"
    missing_policy = "missing_policy"
    denied_import = "denied_import"
    import_error = "import_error"
    init_timeout = "init_timeout"
    init_error = "init_error"
    oom = "oom"
    rollout = "rollout_timeout"

    @property
    def success(self) -> bool:
        return self == Verdict.ok

    @property
    def rejected(self) -> bool:
        """True for Phase 1 request failures that do not spend budget."""

        return self in {Verdict.budget_invalid, Verdict.invalid_case}

    @property
    def charged(self) -> bool:
        """True once a submit reaches Phase 2 or later."""

        return not self.rejected

    @property
    def partial(self) -> bool:
        """True when submit-level failure may still preserve episodes/."""

        return self in {Verdict.oom, Verdict.rollout}


class PoolKind(StrEnum):
    train = "train"
    valid = "valid"
    final = "final"

    @property
    def visible(self) -> bool:
        return self == PoolKind.train

    @property
    def hidden(self) -> bool:
        return self != PoolKind.train

    @property
    def charged(self) -> bool:
        """Only train submits consume the agent's episode budget."""

        return self == PoolKind.train


@dataclass(frozen=True, slots=True)
class Budget:
    """Episode budget for a Run."""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("budget limit must be non-negative")
        if not 0 <= self.used <= self.limit:
            raise ValueError("budget used must be within [0, limit]")

    @property
    def left(self) -> int:
        return self.limit - self.used

    def can(self, count: int) -> bool:
        return 0 <= count <= self.left

    def accepts(self, count: int, *, minimum: int, maximum: int) -> bool:
        """Phase 1 count check for a submit request."""

        cap = min(maximum, self.left)
        return minimum <= count <= cap

    def spend(self, count: int) -> Budget:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > self.left:
            raise ValueError("budget exceeded")
        return Budget(limit=self.limit, used=self.used + count)

    def settle(self, count: int, verdict: Verdict) -> Budget:
        """Apply protocol charging rules for a submit verdict."""

        if verdict.charged:
            return self.spend(count)
        return self


@dataclass(frozen=True, slots=True)
class Task:
    """Agent-visible task contract."""

    name: str
    version: str
    obs: dict[str, Any]
    act: dict[str, Any]
    steps: int
    cases: int
    storage: str = "inline"
    rewards: dict[str, str] | None = None

    def contains(self, case: int) -> bool:
        return 0 <= case < self.cases


@dataclass(frozen=True, slots=True)
class Secret:
    """Judge-only task data."""

    train: str
    valid: str
    final: str
    expert: float
    random: float
    valid_size: int = 64
    final_size: int = 256

    def __post_init__(self) -> None:
        if self.valid_size < 0:
            raise ValueError("valid_size must be non-negative")
        if self.final_size < 0:
            raise ValueError("final_size must be non-negative")


@dataclass(frozen=True, slots=True)
class Case:
    """One concrete environment instance inside a Pool."""

    id: int
    ref: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("case id must be non-negative")
        if not isinstance(self.data, Mapping):
            raise TypeError("case data must be a mapping")
        object.__setattr__(self, "data", dict(self.data))


@dataclass(frozen=True, slots=True)
class Pool:
    kind: PoolKind
    size: int
    ref: str
    cases: tuple[Case, ...] = ()

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("pool size must be non-negative")
        object.__setattr__(self, "cases", tuple(self.cases))
        if self.cases and len(self.cases) != self.size:
            raise ValueError("pool size must match concrete cases")
        for index, item in enumerate(self.cases):
            if not isinstance(item, Case):
                raise TypeError("pool cases must contain Case objects")
            if item.id != index:
                raise ValueError("concrete case ids must match pool indexes")

    @property
    def visible(self) -> bool:
        return self.kind.visible

    @property
    def hidden(self) -> bool:
        return self.kind.hidden

    def contains(self, case: int) -> bool:
        return 0 <= case < self.size

    def case(self, case: int) -> Case:
        """Map an agent-facing integer id to a pool-scoped Case."""

        if not self.contains(case):
            raise ValueError("case id outside pool")
        if self.cases:
            return self.cases[case]
        return Case(id=case, ref=f"{self.ref}/{case:06d}")

    def trim(self, size: int) -> Pool:
        """Return the first `size` cases, preserving concrete case data."""

        if size < 0 or size > self.size:
            raise ValueError("pool trim size outside pool")
        if self.cases:
            return replace(self, size=size, cases=self.cases[:size])
        return replace(self, size=size)


Value = Callable[[Pool, tuple[float, ...]], float | None]


@dataclass(frozen=True, slots=True)
class Caps:
    """Environment-declared optional artifact capabilities."""

    observations: bool = False
    video: bool = False

    @property
    def any(self) -> bool:
        return self.observations or self.video

    def body(self) -> dict[str, bool]:
        return {"observations": self.observations, "video": self.video}


@dataclass(frozen=True, slots=True)
class Env:
    """Registered environment contract."""

    task: Task
    secret: Secret
    make: Callable[[], Any]
    value: Value | None = None
    caps: Caps = field(default_factory=Caps)
    text: str = ""

    def pool(self, kind: PoolKind) -> Pool:
        if kind == PoolKind.train:
            return Pool(kind=kind, size=self.task.cases, ref=self.secret.train)
        if kind == PoolKind.valid:
            return Pool(kind=kind, size=self.secret.valid_size, ref=self.secret.valid)
        if kind == PoolKind.final:
            return Pool(kind=kind, size=self.secret.final_size, ref=self.secret.final)
        raise ValueError(f"unknown pool kind: {kind}")


@dataclass(frozen=True, slots=True)
class Score:
    mean: float | None
    std: float | None
    value: float | None = None
    returns: tuple[float, ...] = ()

    @property
    def present(self) -> bool:
        return self.mean is not None

    @property
    def primary(self) -> float | None:
        return self.value if self.value is not None else self.mean

    def rank(self, index: int) -> float:
        if self.primary is None:
            return -inf
        return float(self.primary) + index * 1e-12


@dataclass(frozen=True, slots=True)
class Work:
    """Agent-owned project directory, addressed abstractly."""

    ref: str


@dataclass(frozen=True, slots=True)
class Submit:
    """One agent-requested submission against train cases."""

    index: int
    cases: tuple[int, ...]
    verdict: Verdict | None = None
    snap: int | None = None
    feed: str | None = None

    @property
    def cost(self) -> int:
        return len(self.cases)

    def reject(
        self,
        task: Task,
        budget: Budget,
        *,
        minimum: int,
        maximum: int,
    ) -> Verdict | None:
        """Return the Phase 1 rejection verdict, or None if accepted."""

        if any(not task.contains(case) for case in self.cases):
            return Verdict.invalid_case
        if not budget.accepts(self.cost, minimum=minimum, maximum=maximum):
            return Verdict.budget_invalid
        return None

    @property
    def charged(self) -> bool:
        return self.verdict.charged if self.verdict is not None else False

    def with_snap(self, snap: int) -> Submit:
        return replace(self, snap=snap)

    def with_feed(self, feed: str) -> Submit:
        return replace(self, feed=feed)

    def with_verdict(self, verdict: Verdict) -> Submit:
        return replace(self, verdict=verdict)


@dataclass(frozen=True, slots=True)
class Snap:
    """Immutable Work snapshot created at submit phase 2."""

    index: int
    submit: int
    ref: str
    verdict: Verdict | None = None
    cost: int | None = None

    @property
    def ok(self) -> bool:
        return self.verdict == Verdict.ok

    @property
    def candidate(self) -> bool:
        """Only ok snapshots participate in validation-time selection."""

        return self.ok

    def with_verdict(self, verdict: Verdict) -> Snap:
        return replace(self, verdict=verdict)


@dataclass(frozen=True, slots=True)
class Eval:
    kind: PoolKind
    snap: int
    pool: str
    score: Score


@dataclass(frozen=True, slots=True)
class Exec:
    """Runtime execution result before protocol feedback formatting."""

    verdict: Verdict
    score: Score
    errors: tuple[str, ...] = ()
    traces: tuple[Trace, ...] = ()

    @property
    def ok(self) -> bool:
        return self.verdict == Verdict.ok


@dataclass(frozen=True, slots=True)
class Feed:
    """Agent-visible feedback for one Submit."""

    submit: int
    verdict: Verdict
    cost: int
    score: Score
    errors: tuple[str, ...] = ()
    lengths: tuple[int, ...] = ()

    @classmethod
    def from_exec(cls, submit: Submit, result: Exec) -> Feed:
        return cls(
            submit=submit.index,
            verdict=result.verdict,
            cost=submit.cost,
            score=result.score,
            errors=result.errors,
            lengths=tuple(len(trace.steps) for trace in result.traces),
        )


@dataclass(frozen=True, slots=True)
class Report:
    """One persisted submit feedback report."""

    feed: Feed
    started: datetime
    completed: datetime
    wall: float
    first: int | None = None
    traces: tuple[Trace, ...] = ()

    def __post_init__(self) -> None:
        if self.wall < 0:
            raise ValueError("wall must be non-negative")


@dataclass(frozen=True, slots=True)
class Turn:
    """One environment transition."""

    obs: Any
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class Trace:
    episode: int
    reward: float
    steps: tuple[dict[str, Any], ...]
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class Pick:
    """Validation-time best-snapshot choice."""

    best: int | None
    vals: dict[int, Score]

    def __post_init__(self) -> None:
        if self.best is not None and self.best not in self.vals:
            raise ValueError("best must be a key in vals")

    @classmethod
    def from_vals(cls, vals: dict[int, Score]) -> Pick:
        """Pick highest val score, tie-breaking by latest submit index."""

        if not vals:
            return cls(best=None, vals={})
        best = max(vals.items(), key=lambda item: item[1].rank(item[0]))[0]
        return cls(best=best, vals=dict(vals))

    @property
    def scores(self) -> dict[int, float]:
        """Scalar val_score view for run.json."""

        return {
            index: score.primary
            for index, score in self.vals.items()
            if score.primary is not None
        }

    @property
    def empty(self) -> bool:
        return self.best is None


@dataclass(frozen=True, slots=True)
class Run:
    key: str
    model: str
    env: str
    exp: str
    protocol: str
    budget: Budget
    state: RunState = RunState.open
    pick: Pick | None = None
    outcome: OutcomeStatus | None = None

    def alive(self) -> bool:
        return self.state == RunState.open

    @property
    def exhausted(self) -> bool:
        return self.budget.left == 0

    def charge(self, count: int, verdict: Verdict) -> Run:
        return replace(self, budget=self.budget.settle(count, verdict))

    def closing(self) -> Run:
        return replace(self, state=RunState.closing)

    def done(self, pick: Pick | None) -> Run:
        return Run(
            key=self.key,
            model=self.model,
            env=self.env,
            exp=self.exp,
            protocol=self.protocol,
            budget=self.budget,
            state=RunState.closed,
            pick=pick,
            outcome=(
                OutcomeStatus.no_ok_submit
                if pick is None or pick.empty
                else OutcomeStatus.completed
            ),
        )

    def fail(self) -> Run:
        return replace(self, state=RunState.failed, outcome=OutcomeStatus.error)
