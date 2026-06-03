"""Default episode roller.

The Roller converts a small environment adapter into protocol Traces. It
does not import Gym; Gym-specific wrappers can implement the World
protocol separately.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

from ...core import Pool, Task, Trace, Turn, World

Make = Callable[[], World]


@dataclass(frozen=True, slots=True)
class Roller:
    """Run policy episodes against a World factory."""

    make: Make

    def run(
        self,
        policy: object,
        task: Task,
        pool: Pool,
        cases: tuple[int, ...],
    ) -> tuple[Trace, ...]:
        traces: list[Trace] = []
        for local, case in enumerate(cases):
            traces.append(self._episode(policy, task, pool, case, local))
        return tuple(traces)

    def _episode(
        self,
        policy: object,
        task: Task,
        pool: Pool,
        case: int,
        local: int,
    ) -> Trace:
        world = self.make()
        steps: list[dict[str, Any]] = []
        stdout = io.StringIO()
        stderr = io.StringIO()
        total = 0.0

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                policy.reset(local)
            obs = world.reset(pool.case(case))
        except Exception as exc:
            return Trace(
                episode=case,
                reward=0.0,
                steps=(),
                error=_error("reset_error", exc),
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
            )

        error: str | None = None
        for index in range(task.steps):
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    action = policy.act(obs)
            except Exception as exc:
                action = world.sample()
                error = _error("act_error", exc)

            turn = world.step(action)
            total += float(turn.reward)
            steps.append(_step(index, obs, action, turn))
            obs = turn.obs

            if turn.done or error is not None:
                break

        return Trace(
            episode=case,
            reward=total,
            steps=tuple(steps),
            error=error,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )


def _step(index: int, obs: Any, action: Any, turn: Turn) -> dict[str, Any]:
    return {
        "t": index,
        "obs": obs,
        "action": action,
        "reward": float(turn.reward),
        "terminated": turn.terminated,
        "truncated": turn.truncated,
        "info": dict(turn.info),
    }


def _error(kind: str, exc: Exception) -> str:
    return f"{kind}: {type(exc).__name__}: {exc}"
