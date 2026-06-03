"""Structured environment coverage manifest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .gym.spec import BOX2D, MUJOCO, P1


class Level(IntEnum):
    """Environment integration maturity level."""

    catalogued = 0
    smoke = 1
    tasked = 2
    scored = 3
    calibrated = 4

    @property
    def code(self) -> str:
        return f"L{int(self)}"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @classmethod
    def parse(cls, value: str | int | Level) -> Level:
        if isinstance(value, Level):
            return value
        if isinstance(value, int):
            return cls(value)
        raw = value.strip().lower()
        if raw.startswith("l") and raw[1:].isdigit():
            return cls(int(raw[1:]))
        for level, label in _LABELS.items():
            if raw in {level.name, label.lower()}:
                return level
        raise ValueError(f"unknown environment support level: {value}")


_LABELS = {
    Level.catalogued: "Catalogued",
    Level.smoke: "Smoke",
    Level.tasked: "Tasked",
    Level.scored: "Scored",
    Level.calibrated: "Calibrated",
}


@dataclass(frozen=True, slots=True)
class Entry:
    """One planned or integrated environment."""

    name: str
    level: Level
    family: str
    adapter: str
    upstream: str = ""
    dependency: str = "base"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("manifest entry name must be non-empty")
        if not self.family:
            raise ValueError("manifest entry family must be non-empty")
        if not self.adapter:
            raise ValueError("manifest entry adapter must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        body = {
            "name": self.name,
            "level": self.level.code,
            "level_label": self.level.label,
            "family": self.family,
            "adapter": self.adapter,
            "dependency": self.dependency,
        }
        if self.upstream:
            body["upstream"] = self.upstream
        if self.notes:
            body["notes"] = self.notes
        return body


def entries() -> tuple[Entry, ...]:
    """Return the static EvoPolicyGym environment coverage manifest."""

    return (*_base(), *_gym())


def by_name() -> dict[str, Entry]:
    """Return manifest entries keyed by EvoPolicyGym environment name."""

    return {entry.name: entry for entry in entries()}


def from_discovery(report) -> tuple[Entry, ...]:
    """Convert an installed discovery report into L0 manifest entries."""

    known_upstreams = {entry.upstream for entry in entries() if entry.upstream}
    rows: list[Entry] = []
    for family in report.families:
        for env_id in family.ids:
            if env_id in known_upstreams:
                continue
            rows.append(
                Entry(
                    name=env_id,
                    level=Level.catalogued,
                    family=family.name,
                    adapter="gymnasium",
                    upstream=env_id,
                    dependency=_dependency(family.name, env_id),
                    notes="discovered upstream registry id; not yet integrated",
                )
            )
    return tuple(rows)


def from_bulk(specs) -> tuple[Entry, ...]:
    """Convert dynamic Gymnasium bulk specs into L1 manifest entries."""

    known = {entry.name for entry in entries()}
    rows: list[Entry] = []
    for item in specs:
        if item.name in known:
            continue
        family = item.family or "Unclassified Gymnasium"
        rows.append(
            Entry(
                name=item.name,
                level=Level.smoke,
                family=family,
                adapter="gymnasium",
                upstream=item.upstream,
                dependency=_dependency(family, item.upstream),
                notes="dynamic Gymnasium registry id; bulk smoke level",
            )
        )
    return tuple(rows)


def _base() -> tuple[Entry, ...]:
    return (
        Entry(
            name="toy",
            level=Level.scored,
            family="EvoPolicyGym",
            adapter="builtin",
            notes="minimal smoke environment",
        ),
        Entry(
            name="cartpole",
            level=Level.scored,
            family="EvoPolicyGym",
            adapter="builtin",
            upstream="CartPole-v1",
            notes="dependency-free classic-control implementation",
        ),
    )


def _gym() -> tuple[Entry, ...]:
    rows: list[Entry] = []
    for spec in P1:
        rows.append(
            Entry(
                name=spec.name,
                level=Level.tasked,
                family=_gym_family(spec.id),
                adapter="gymnasium",
                upstream=spec.id,
                dependency="env-gym",
                notes="generic Gymnasium adapter with seed-backed data support",
            )
        )
    for spec in (*BOX2D, *MUJOCO):
        level = Level.tasked if spec.name == "gym/racing" else Level.smoke
        notes = (
            "generic Gymnasium adapter with external image observation artifacts"
            if spec.name == "gym/racing"
            else "generic Gymnasium adapter; native dependency smoke level"
        )
        rows.append(
            Entry(
                name=spec.name,
                level=level,
                family=_gym_family(spec.id),
                adapter="gymnasium",
                upstream=spec.id,
                dependency="env-gym",
                notes=notes,
            )
        )
    return tuple(rows)


def _gym_family(env_id: str) -> str:
    if env_id in {"Blackjack-v1", "CliffWalking-v1", "FrozenLake-v1", "Taxi-v4"}:
        return "Official Gymnasium: Toy Text"
    if env_id in {"BipedalWalker-v3", "CarRacing-v3", "LunarLander-v3"}:
        return "Official Gymnasium: Box2D"
    if env_id.endswith(("-v4", "-v5")):
        return "Official Gymnasium: MuJoCo"
    return "Official Gymnasium: Classic Control"


def _dependency(family: str, upstream: str = "") -> str:
    if upstream == "mo-supermario-v0":
        return "env-mario"
    if family == "Official Gymnasium: JAX" or upstream.startswith(("phys2d/", "tabular/")):
        return "env-jax"
    if family.startswith("Official Gymnasium"):
        return "env-gym"
    if family in {"MiniGrid", "HighwayEnv", "Gymnasium-Robotics", "MO-Gymnasium"}:
        return "env-compatible"
    if family == "MiniWorld":
        return "env-visual"
    if family == "MetaWorld":
        return "env-heavy"
    if family == "BrowserGym MiniWoB++":
        return "env-web"
    return "optional"
