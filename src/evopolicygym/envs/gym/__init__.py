"""Generic Gymnasium environment registrations."""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import Any

from ...core import Caps, Env, Pool, PoolKind, Secret, Task
from .dynamic import BulkSpec
from .dynamic import discover as _bulk
from .minigrid_assets import patch_minigrid_wfc_assets
from .space import schema
from .spec import (
    ACROBOT,
    BLACKJACK,
    BY_NAME,
    CARTPOLE,
    CLIFF,
    CONTINUOUSCAR,
    DEFAULTS,
    FROZENLAKE,
    MOUNTAINCAR,
    PENDULUM,
    TAXI,
    GymSpec,
)
from .task import render as _task
from .world import Gym


def available() -> bool:
    """Return whether Gymnasium can be imported in this environment."""

    return importlib.util.find_spec("gymnasium") is not None


def gym(spec: GymSpec) -> Env:
    """Build an EvoPolicyGym registration for one Gymnasium registry id."""

    obs, act = _spaces(spec)
    storage = _storage(obs)
    return Env(
        task=Task(
            name=spec.name,
            version=spec.version,
            obs=obs,
            act=act,
            steps=spec.steps,
            cases=spec.cases,
            storage=storage,
            rewards={"return": "sum of Gymnasium rewards over the episode"},
        ),
        secret=Secret(
            train=f"{spec.name}/train",
            valid=f"{spec.name}/validation",
            final=f"{spec.name}/heldout",
            expert=spec.expert,
            random=spec.random,
            valid_size=spec.valid_size,
            final_size=spec.final_size,
        ),
        make=lambda: Gym(spec),
        value=_value,
        caps=Caps(observations=storage == "external"),
        text=_task(spec),
    )


def by_name(name: str) -> GymSpec:
    """Return a registered Gymnasium spec by EvoPolicyGym environment name."""

    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown Gymnasium environment: {name}") from exc


def acrobot() -> Env:
    """Return the Acrobot-v1 Gymnasium registration."""

    return gym(ACROBOT)


def cartpole() -> Env:
    """Return the CartPole-v1 Gymnasium registration."""

    return gym(CARTPOLE)


def mountaincar() -> Env:
    """Return the MountainCar-v0 Gymnasium registration."""

    return gym(MOUNTAINCAR)


def continuouscar() -> Env:
    """Return the MountainCarContinuous-v0 Gymnasium registration."""

    return gym(CONTINUOUSCAR)


def pendulum() -> Env:
    """Return the Pendulum-v1 Gymnasium registration."""

    return gym(PENDULUM)


def taxi() -> Env:
    """Return the Taxi-v4 Gymnasium registration."""

    return gym(TAXI)


def blackjack() -> Env:
    """Return the Blackjack-v1 Gymnasium registration."""

    return gym(BLACKJACK)


def cliff() -> Env:
    """Return the CliffWalking-v1 Gymnasium registration."""

    return gym(CLIFF)


def frozenlake() -> Env:
    """Return the FrozenLake-v1 Gymnasium registration."""

    return gym(FROZENLAKE)


def envs(*, bulk: bool = False, filters: tuple[str, ...] = ()) -> tuple[Env, ...]:
    """Return default Gymnasium registrations when the optional extra exists."""

    if not available():
        return ()
    selected = set(filters)
    rows = []
    specs = list(DEFAULTS)
    if selected:
        specs = [spec for spec in specs if spec.name in selected or spec.id in selected]
    if bulk:
        specs.extend(item.spec for item in _bulk(selected))
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        try:
            rows.append(gym(spec))
        except Exception as exc:  # noqa: BLE001 - optional native deps may be absent.
            warnings.warn(
                f"skipping Gymnasium env {spec.name} ({spec.id}): {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(rows)


def _spaces(spec: GymSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    gymnasium = _gymnasium()
    patch_minigrid_wfc_assets(spec.id)
    env = gymnasium.make(spec.id, **dict(spec.kwargs))
    try:
        return schema(env.observation_space), schema(env.action_space)
    finally:
        env.close()


def _storage(obs: dict[str, Any]) -> str:
    return "external" if _has_external(obs) else "inline"


def _has_external(schema: dict[str, Any]) -> bool:
    if schema.get("storage") == "external" or schema.get("type") == "Image":
        return True
    spaces = schema.get("spaces")
    if isinstance(spaces, list):
        return any(isinstance(item, dict) and _has_external(item) for item in spaces)
    if isinstance(spaces, dict):
        return any(isinstance(item, dict) and _has_external(item) for item in spaces.values())
    return False


def _gymnasium() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gymnasium support requires `uv sync --extra env-gym`") from exc


def _value(pool: Pool, returns: tuple[float, ...]) -> float | None:
    if pool.kind == PoolKind.train or not returns:
        return None
    return sum(returns) / len(returns)


__all__ = [
    "ACROBOT",
    "BLACKJACK",
    "BulkSpec",
    "BY_NAME",
    "CARTPOLE",
    "CLIFF",
    "CONTINUOUSCAR",
    "DEFAULTS",
    "FROZENLAKE",
    "MOUNTAINCAR",
    "PENDULUM",
    "TAXI",
    "Gym",
    "GymSpec",
    "acrobot",
    "available",
    "blackjack",
    "by_name",
    "cartpole",
    "cliff",
    "continuouscar",
    "envs",
    "frozenlake",
    "gym",
    "mountaincar",
    "pendulum",
    "taxi",
]
