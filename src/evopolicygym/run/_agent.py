"""Run-owned Agent lifecycle values and narrow execution contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """Detached outcome of one Coding Agent execution."""

    returncode: int | None
    timed_out: bool = False
    stopped_after_terminal: bool = False
    start_failed: bool = False
    start_error_type: str | None = None
    start_errno: int | None = None

    def __post_init__(self) -> None:
        if self.returncode is not None and type(self.returncode) is not int:
            raise TypeError("returncode must be an integer or None")
        if (
            type(self.timed_out) is not bool
            or type(self.stopped_after_terminal) is not bool
            or type(self.start_failed) is not bool
        ):
            raise TypeError("Agent outcome flags must be exact bool values")
        if sum(
            (
                self.timed_out,
                self.stopped_after_terminal,
                self.start_failed,
            )
        ) > 1:
            raise ValueError(
                "Agent outcome classifications are mutually exclusive"
            )
        if self.start_failed and self.returncode is not None:
            raise ValueError("a start failure cannot have a return code")
        if not self.start_failed and (
            self.start_error_type is not None or self.start_errno is not None
        ):
            raise ValueError("start error details require a start failure")
        if self.start_error_type is not None and (
            type(self.start_error_type) is not str or not self.start_error_type
        ):
            raise ValueError("start_error_type must be non-empty text or None")
        if self.start_errno is not None and type(self.start_errno) is not int:
            raise TypeError("start_errno must be an integer or None")


class TerminalSignal(Protocol):
    def wait(self, timeout: float | None = None) -> bool:
        ...


class AgentRunner(Protocol):
    def run(
        self,
        terminal: TerminalSignal,
        *,
        timeout_seconds: float,
    ) -> AgentOutcome:
        ...


__all__: list[str] = []
