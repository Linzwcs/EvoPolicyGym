"""Public execution settings and their private implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class ProcessExecution:
    """Acknowledgement of unisolated local subprocess execution.

    This setting is not a sandbox. Policy and Agent code can exercise the
    authority of the current operating-system user.
    """

    def __init__(self) -> None:
        raise TypeError("use ProcessExecution.unsafe()")

    @classmethod
    def unsafe(cls) -> ProcessExecution:
        return object.__new__(cls)


__all__ = ["ProcessExecution"]
