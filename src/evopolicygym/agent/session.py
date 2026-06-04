"""Agent harness launch and persistent-session protocol."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Protocol

from ..protocol import PROTOCOL

Json = Any
Done = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class Launch:
    """Agent-facing startup context for one benchmark run."""

    root: Path
    endpoint: str
    protocol: str = PROTOCOL
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @classmethod
    def from_host(
        cls,
        host: object,
        endpoint: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> Launch:
        """Build launch context from a local Host-like object."""

        store = getattr(host, "store")
        run = getattr(host, "run")
        return cls(
            root=store.root,
            endpoint=endpoint,
            protocol=run.protocol,
            env=dict(env or {}),
        )

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def system(self) -> Path:
        return self.workspace / "system"

    @property
    def feedback(self) -> Path:
        return self.workspace / "feedback"

    @property
    def agents(self) -> Path:
        return self.workspace / "AGENTS.md"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def environ(self) -> dict[str, str]:
        values = {
            "EVOPOLICYGYM_API": self.endpoint,
            "EVOPOLICYGYM_WORKSPACE": str(self.workspace),
            "EVOPOLICYGYM_SYSTEM": str(self.system),
            "EVOPOLICYGYM_FEEDBACK": str(self.feedback),
            "EVOPOLICYGYM_AGENTS": str(self.agents),
            "EVOPOLICYGYM_PROTOCOL": self.protocol,
            "EVOPOLICYGYM_INFO_URL": f"{self.endpoint}/info",
            "EVOPOLICYGYM_TASK_URL": f"{self.endpoint}/task",
            "EVOPOLICYGYM_SUBMIT_URL": f"{self.endpoint}/submit",
        }
        values.update(self.env)
        return values

    def prompt(self) -> str:
        """Initial instruction sent to the harness session."""

        return dedent(
            f"""
            You are running one EvoPolicyGym benchmark session.

            Keep this harness session alive for the full run. Do not restart
            yourself between submissions. Your current working directory is the
            benchmark workspace.

            First read `AGENTS.md`. Follow its rules for files, submissions,
            feedback, and budget.

            Use relative workspace paths:
            - edit policy code under `system/`
            - read feedback artifacts under `feedback/`
            - do not write outside `system/`

            Improve both policy behavior and code structure. Keep correctness
            and return primary; use clear helper modules when structure helps
            future iterations.

            Do not run local environment rollouts, copied environment
            dynamics/reward functions, Gymnasium/MuJoCo/Box2D/highway
            simulators, or other simulator data generation outside `/submit`.
            All observations, actions, rewards, episode lengths, returns, and
            candidate-policy scores used for optimization must come from
            accepted `/submit` calls and existing `feedback/` artifacts.

            Use the server API:
            - GET {self.endpoint}/info
            - GET {self.endpoint}/task
            - POST {self.endpoint}/submit with env_instances

            Continue improving and submitting until /info reports
            state.is_finalized = true. Do not call /finalize.
            """
        ).strip()


@dataclass(frozen=True, slots=True)
class Reply:
    """One completed harness turn inside a persistent session.

    Returning a Reply means the current turn finished; it does not mean the
    agent session is over. Set `stop=True` only when the harness cannot or
    should not receive another prompt in the same context.
    """

    turn: int
    text: str = ""
    stop: bool = False
    data: Mapping[str, Json] = field(default_factory=dict)


class Session(Protocol):
    """Long-lived interaction channel to one agent harness process/session."""

    @property
    def key(self) -> str: ...

    def step(self, message: str) -> Reply: ...

    def close(self) -> None: ...


class Harness(Protocol):
    """Adapter that starts one concrete agent harness session."""

    def start(self, launch: Launch) -> Session: ...


@dataclass(frozen=True, slots=True)
class Transcript:
    launch: Launch
    session: str
    replies: tuple[Reply, ...]
    reason: str

    @property
    def done(self) -> bool:
        return self.reason == "done"


@dataclass(frozen=True, slots=True)
class Loop:
    """Drive one persistent agent session until the run is done or bounded."""

    harness: Harness
    limit: int = 128
    retries: int = 0
    backoff: float = 1.0
    opening: str = ""
    continuing: str = (
        "Continue optimizing policy behavior and code structure in the same "
        "EvoPolicyGym context."
    )
    recovery: str = ""

    def run(self, launch: Launch, done: Done) -> Transcript:
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.backoff < 0:
            raise ValueError("backoff must be non-negative")
        if done():
            return Transcript(launch=launch, session="", replies=(), reason="done")

        session = self.harness.start(launch)
        replies: list[Reply] = []
        reason = "turn_limit"
        try:
            message = self.opening or launch.prompt()
            for _ in range(self.limit):
                if done():
                    reason = "done"
                    break
                outcome = self._step(session, message, launch, done)
                if outcome.reply is None:
                    reason = "done" if done() else "retry_exhausted"
                    break
                reply = outcome.reply
                replies.append(reply)
                if reply.stop:
                    reason = "retry_exhausted" if outcome.exhausted else "session_stop"
                    break
                if done():
                    reason = "done"
                    break
                message = self.continuing
        finally:
            session.close()

        return Transcript(
            launch=launch,
            session=session.key,
            replies=tuple(replies),
            reason=reason,
        )

    def _step(self, session: Session, message: str, launch: Launch, done: Done) -> _Outcome:
        prompt = message
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                reply = session.step(prompt)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.retries:
                    _emit(
                        launch,
                        "agent.retry.exhausted",
                        attempt=attempt,
                        error=last_error,
                    )
                    return _Outcome(
                        Reply(
                            turn=_turn(session),
                            text=last_error,
                            stop=True,
                            data={"error": last_error, "retry_exhausted": True},
                        ),
                        exhausted=True,
                    )
                _emit(launch, "agent.retry", attempt=attempt + 1, error=last_error)
                self._wait(attempt)
                if done():
                    return _Outcome(None, exhausted=False)
                prompt = self._recovery()
                continue

            if not _retryable(reply):
                return _Outcome(reply, exhausted=False)
            if attempt >= self.retries:
                _emit(
                    launch,
                    "agent.retry.exhausted",
                    attempt=attempt,
                    reply=dict(reply.data),
                )
                return _Outcome(reply, exhausted=True)

            _emit(
                launch,
                "agent.retry",
                attempt=attempt + 1,
                reply=dict(reply.data),
            )
            self._wait(attempt)
            if done():
                return _Outcome(None, exhausted=False)
            prompt = self._recovery()

        return _Outcome(
            Reply(
                turn=_turn(session),
                text=last_error or "retry exhausted",
                stop=True,
                data={"error": last_error or "retry exhausted", "retry_exhausted": True},
            ),
            exhausted=True,
        )

    def _wait(self, attempt: int) -> None:
        if self.backoff <= 0:
            return
        time.sleep(self.backoff * (2**attempt))

    def _recovery(self) -> str:
        if self.recovery:
            return self.recovery
        return dedent(
            """
            Continue the same EvoPolicyGym run after a harness or service failure.
            Do not assume the previous action failed or succeeded. Read `AGENTS.md`
            if needed, call GET /info, inspect `feedback/`, and continue only from
            the current server state. If state.is_finalized is false and budget
            remains, submit the next useful batch. Do not call /finalize.
            Keep improving policy behavior and code structure without changing
            the benchmark protocol.
            """
        ).strip()


@dataclass(frozen=True, slots=True)
class _Outcome:
    reply: Reply | None
    exhausted: bool = False


def _retryable(reply: Reply) -> bool:
    data = reply.data
    if data.get("retryable") is True:
        return True
    if data.get("timed_out") is True:
        return True
    code = data.get("exit_code")
    return isinstance(code, int) and code != 0


def _turn(session: Session) -> int:
    value = getattr(session, "turn", 0)
    return value if isinstance(value, int) else 0


def _emit(launch: Launch, event: str, **data: Json) -> None:
    path = launch.logs / "harness.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event": event,
        **data,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
