"""Dependency-free CartPole environment registration."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from ...core import Case, Env, Pool, PoolKind, Secret, Task, Turn
from ..docs import task as _task

_MAX_STEPS = 500
_GRAVITY = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_TOTAL_MASS = _MASSCART + _MASSPOLE
_LENGTH = 0.5
_POLEMASS_LENGTH = _MASSPOLE * _LENGTH
_FORCE_MAG = 10.0
_TAU = 0.02
_X_THRESHOLD = 2.4
_THETA_THRESHOLD = 12 * 2 * math.pi / 360


@dataclass(slots=True)
class CartPole:
    """Classic cart-pole control world with deterministic case seeding."""

    max_steps: int = _MAX_STEPS
    state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    elapsed: int = 0
    done: bool = False

    def reset(self, case: Case) -> list[float]:
        rng = random.Random(_seed(case))
        self.state = _initial(rng)
        self.elapsed = 0
        self.done = False
        return _obs(self.state)

    def step(self, action: Any) -> Turn:
        if self.done:
            return Turn(obs=_obs(self.state), reward=0.0, terminated=True)

        x, x_dot, theta, theta_dot = self.state
        force = _FORCE_MAG if _action(action) == 1 else -_FORCE_MAG
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + _POLEMASS_LENGTH * theta_dot**2 * sintheta) / _TOTAL_MASS
        thetaacc = (_GRAVITY * sintheta - costheta * temp) / (
            _LENGTH * (4.0 / 3.0 - _MASSPOLE * costheta**2 / _TOTAL_MASS)
        )
        xacc = temp - _POLEMASS_LENGTH * thetaacc * costheta / _TOTAL_MASS

        x = x + _TAU * x_dot
        x_dot = x_dot + _TAU * xacc
        theta = theta + _TAU * theta_dot
        theta_dot = theta_dot + _TAU * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        self.elapsed += 1

        terminated = (
            x < -_X_THRESHOLD
            or x > _X_THRESHOLD
            or theta < -_THETA_THRESHOLD
            or theta > _THETA_THRESHOLD
        )
        truncated = self.elapsed >= self.max_steps and not terminated
        self.done = terminated or truncated

        return Turn(
            obs=_obs(self.state),
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
            info={"steps": self.elapsed},
        )

    def sample(self) -> int:
        return 0


def cartpole() -> Env:
    """Return the built-in CartPole environment registration."""

    return Env(
        task=Task(
            name="cartpole",
            version="0.1",
            obs={
                "type": "Box",
                "shape": [4],
                "dtype": "float64",
                "fields": ["x", "x_dot", "theta", "theta_dot"],
                "low": [-4.8, -10.0, -0.41887902047863906, -10.0],
                "high": [4.8, 10.0, 0.41887902047863906, 10.0],
            },
            act={"type": "Discrete", "n": 2, "labels": ["left", "right"]},
            steps=_MAX_STEPS,
            cases=64,
        ),
        secret=Secret(
            train="cartpole/train",
            valid="cartpole/validation",
            final="cartpole/heldout",
            expert=100.0,
            random=4.0,
            valid_size=32,
            final_size=64,
        ),
        make=CartPole,
        value=_value,
        text=_task(__file__),
    )


def _value(pool: Pool, returns: tuple[float, ...]) -> float | None:
    if pool.kind not in {PoolKind.valid, PoolKind.final} or not returns:
        return None
    return 100.0 * sum(returns) / (len(returns) * _MAX_STEPS)


def _seed(case: Case) -> int:
    raw = case.data.get("seed")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return int(raw)
        except ValueError:
            pass
    key = case.ref or f"case:{case.id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _action(action: Any) -> int:
    try:
        value = int(action)
    except (TypeError, ValueError):
        return 0
    return 1 if value > 0 else 0


def _initial(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
    )


def _obs(state: tuple[float, float, float, float]) -> list[float]:
    return [float(item) for item in state]
