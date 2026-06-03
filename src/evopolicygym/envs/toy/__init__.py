"""Toy built-in environment for smoke tests and examples."""

from __future__ import annotations

from dataclasses import dataclass

from ...core import Case, Env, Pool, PoolKind, Secret, Task, Turn
from ..docs import task as _task


@dataclass(slots=True)
class Toy:
    """One-step additive world.

    The observation starts at the case id. The policy action is added to the
    state, the new state is returned as reward, and the episode terminates.
    """

    state: int = 0

    def reset(self, case: Case) -> int:
        raw = case.data.get("start", case.data.get("value", case.id))
        self.state = int(raw)
        return self.state

    def step(self, action: int) -> Turn:
        self.state += int(action)
        return Turn(obs=self.state, reward=float(self.state), terminated=True)

    def sample(self) -> int:
        return 0


def toy() -> Env:
    """Return the built-in toy environment registration."""

    return Env(
        task=Task(
            name="toy",
            version="0.1",
            obs={"type": "Discrete", "n": 16},
            act={"type": "Discrete", "n": 3},
            steps=1,
            cases=8,
        ),
        secret=Secret(
            train="toy/train",
            valid="toy/validation",
            final="toy/heldout",
            expert=10.0,
            random=0.0,
        ),
        make=Toy,
        value=_value,
        text=_task(__file__),
    )


def _value(pool: Pool, returns: tuple[float, ...]) -> float | None:
    if pool.kind != PoolKind.final or not returns:
        return None
    return sum(returns) / len(returns)
