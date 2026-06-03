"""Gymnasium World adapter."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ...core import Case, Turn
from . import space
from .minigrid_assets import patch_minigrid_wfc_assets
from .spec import GymSpec

_SEED_MOD = 2**32


@dataclass(slots=True)
class Gym:
    """A lazy Gymnasium environment wrapped in the EvoPolicyGym World protocol."""

    spec: GymSpec
    _env: Any | None = field(default=None, init=False, repr=False)
    _obs: Any = field(default=None, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)
    _elapsed: int = field(default=0, init=False, repr=False)

    def reset(self, case: Case) -> Any:
        env = self._make()
        seed = _seed(case)
        self._seed(env, seed)
        options = case.data.get("options")
        kwargs = {"seed": seed}
        if isinstance(options, Mapping):
            kwargs["options"] = dict(options)
        obs, _info = env.reset(**kwargs)
        self._obs = space.encode(obs)
        self._done = False
        self._elapsed = 0
        return self._obs

    def step(self, action: Any) -> Turn:
        env = self._make()
        if self._done:
            return Turn(obs=self._obs, reward=0.0, terminated=True, info={"already_done": True})

        raw, invalid = space.action(env.action_space, action)
        obs, reward, terminated, truncated, info = env.step(raw)
        self._obs = space.encode(obs)
        self._elapsed += 1
        self._done = bool(terminated or truncated)
        body = _info(info)
        body["steps"] = self._elapsed
        scalar_reward, raw_reward = _reward(reward)
        if raw_reward is not None:
            body["gym_reward"] = raw_reward
            body["reward_scalarization"] = "sum"
        if invalid:
            body["action_invalid"] = True
        return Turn(
            obs=self._obs,
            reward=scalar_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=body,
        )

    def sample(self) -> Any:
        return space.sample(self._make().action_space)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _make(self) -> Any:
        if self._env is None:
            gym = _gymnasium()
            patch_minigrid_wfc_assets(self.spec.id)
            self._env = gym.make(self.spec.id, **dict(self.spec.kwargs))
        return self._env

    def _seed(self, env: Any, seed: int) -> None:
        with suppress(Exception):
            env.action_space.seed(seed + 1)
        with suppress(Exception):
            env.observation_space.seed(seed + 2)


def _gymnasium() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gymnasium support requires `uv sync --extra env-gym`") from exc


def _seed(case: Case) -> int:
    raw = case.data.get("seed")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return _seed32(raw)
    if isinstance(raw, str) and raw:
        try:
            return _seed32(int(raw))
        except ValueError:
            pass
    key = case.ref or f"case:{case.id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return _seed32(int.from_bytes(digest[:8], "big"))


def _seed32(value: int) -> int:
    return int(value) % _SEED_MOD


def _info(value: Any) -> dict[str, Any]:
    encoded = space.encode(value)
    if isinstance(encoded, dict):
        return dict(encoded)
    return {"gym_info": encoded}


def _reward(value: Any) -> tuple[float, Any | None]:
    encoded = space.encode(value)
    if isinstance(encoded, int | float) and not isinstance(encoded, bool):
        return float(encoded), None
    values = tuple(_numbers(encoded))
    if not values:
        raise TypeError(f"reward is not numeric: {type(value).__name__}")
    return sum(values), encoded


def _numbers(value: Any):
    if isinstance(value, int | float) and not isinstance(value, bool):
        yield float(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _numbers(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _numbers(item)
