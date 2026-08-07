"""Keep Session state transitions on the Host coordinator thread."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

from .._agent import AgentOutcome, AgentRunner, TerminalSignal


class SessionRequestDispatcher(Protocol):
    def dispatch_next(self, *, timeout_seconds: float) -> bool:
        ...


class SessionDispatchingAgentRunner:
    """Run the Agent worker while the caller dispatches Session requests."""

    def __init__(
        self,
        *,
        agent_runner: AgentRunner,
        dispatcher: SessionRequestDispatcher,
    ) -> None:
        self._agent_runner = agent_runner
        self._dispatcher = dispatcher

    def run(
        self,
        terminal: TerminalSignal,
        *,
        timeout_seconds: float,
    ) -> AgentOutcome:
        completion = _AgentCompletion()

        def run_agent() -> None:
            try:
                completion.outcome = self._agent_runner.run(
                    terminal,
                    timeout_seconds=timeout_seconds,
                )
            except BaseException as error:
                completion.error = error
            finally:
                completion.done.set()

        worker = threading.Thread(
            target=run_agent,
            name="evopolicygym-agent-runner",
        )
        worker.start()
        try:
            while not completion.done.is_set():
                self._dispatcher.dispatch_next(timeout_seconds=0.02)
        finally:
            worker.join()

        if completion.error is not None:
            raise completion.error
        assert completion.outcome is not None
        return completion.outcome


@dataclass(slots=True)
class _AgentCompletion:
    done: threading.Event = field(default_factory=threading.Event)
    outcome: AgentOutcome | None = None
    error: BaseException | None = None


__all__: list[str] = []
