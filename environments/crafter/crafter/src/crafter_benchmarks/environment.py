"""One fresh strict Crafter 1.8.3 Environment per Episode."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, MutableSet, Sequence
from typing import Protocol, SupportsFloat, cast

import crafter
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import CrafterConfig
from .constants import ACHIEVEMENTS, ACTIONS

_AREA = (64, 64)
_VIEW = (9, 9)
_OBSERVATION_SHAPE = (64, 64, 3)
_ACTION_IDS = frozenset(range(len(ACTIONS)))


class _CrafterEnv(Protocol):
    @property
    def action_names(self) -> Sequence[str]: ...

    def reset(self) -> object: ...

    def step(self, action: int) -> object: ...


class _CrafterWorld(Protocol):
    _chunks: object
    _objects: object


class _InsertionOrderedObjectSet(MutableSet[object]):
    """Set-compatible collection with deterministic insertion-order iteration."""

    def __init__(self, values: Sequence[object] = ()) -> None:
        self._members = dict.fromkeys(values)

    def __contains__(self, value: object) -> bool:
        return value in self._members

    def __iter__(self) -> Iterator[object]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def add(self, value: object) -> None:
        self._members[value] = None

    def discard(self, value: object) -> None:
        self._members.pop(value, None)


class CrafterEnvironment:
    """Strict seeded adapter around the canonical RGB Crafter profile."""

    def __init__(self, episode: EpisodeSpec, *, config: CrafterConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not CrafterConfig:
            raise TypeError("config must be CrafterConfig")
        if episode.scenario is not None:
            raise ValueError("Crafter configuration belongs in CrafterConfig")

        environment = cast(
            _CrafterEnv,
            crafter.Env(
                area=_AREA,
                view=_VIEW,
                size=_OBSERVATION_SHAPE[:2],
                reward=True,
                length=config.max_episode_steps,
                seed=episode.environment_seed,
            ),
        )
        if tuple(environment.action_names) != ACTIONS:
            raise RuntimeError("Crafter action meanings changed incompatibly")

        self._environment: _CrafterEnv | None = environment
        self._max_episode_steps = config.max_episode_steps
        self._achievements = {name: 0 for name in ACHIEVEMENTS}
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        environment = self._get_environment()
        observation = environment.reset()
        _stabilize_chunk_iteration(environment)
        self._started = True
        return _observation(observation)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTION_IDS:
            raise InvalidAction()

        result = self._get_environment().step(action)
        if type(result) is not tuple or len(result) != 4:
            raise RuntimeError("Crafter returned an invalid step result")
        observation, reward, done, information = result
        if type(done) is not bool:
            raise RuntimeError("Crafter returned an invalid done flag")
        if type(information) is not dict:
            raise RuntimeError("Crafter returned invalid information")

        discount = _discount(information)
        achievements = _achievements(information)
        unlocked = [
            name
            for name in ACHIEVEMENTS
            if self._achievements[name] == 0 and achievements[name] > 0
        ]
        self._achievements = achievements
        self._steps += 1

        terminated = bool(done and discount == 0.0)
        truncated = bool(done and not terminated)
        if self._steps >= self._max_episode_steps and not terminated:
            truncated = True
        self._done = terminated or truncated
        public_unlocked: list[PolicyValue] = list(unlocked)

        return Step(
            observation=_observation(observation),
            reward=_number(reward, "reward"),
            terminated=terminated,
            truncated=truncated,
            metrics={"achievements_unlocked": public_unlocked},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment = None
        self._closed = True

    def _get_environment(self) -> _CrafterEnv:
        if self._environment is None:
            raise RuntimeError("Environment is closed")
        return self._environment


def _observation(value: object) -> TensorValue:
    if (
        type(value) is not numpy.ndarray
        or value.dtype != numpy.dtype("uint8")
        or value.shape != _OBSERVATION_SHAPE
    ):
        raise RuntimeError("Crafter returned an invalid RGB observation")
    return TensorValue(
        dtype="uint8",
        shape=_OBSERVATION_SHAPE,
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


def _stabilize_chunk_iteration(environment: _CrafterEnv) -> None:
    """Remove address-dependent set iteration from pinned Crafter 1.8.3."""
    world_value = getattr(environment, "_world", None)
    if world_value is None:
        return
    world = cast(_CrafterWorld, world_value)
    chunks_value = world._chunks
    objects_value = world._objects
    if not isinstance(chunks_value, defaultdict) or type(objects_value) is not list:
        raise RuntimeError("Crafter 1.8.3 world internals changed incompatibly")

    objects = cast(list[object], objects_value)
    object_order = {
        id(member): index
        for index, member in enumerate(objects)
        if member is not None
    }
    chunks = cast(
        defaultdict[tuple[object, object, object, object], set[object]],
        chunks_value,
    )
    stable_chunks: defaultdict[
        tuple[object, object, object, object], _InsertionOrderedObjectSet
    ] = defaultdict(_InsertionOrderedObjectSet)
    for chunk, members in chunks.items():
        if (
            type(chunk) is not tuple
            or len(chunk) != 4
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, numpy.integer))
                for bound in chunk
            )
            or type(members) is not set
            or any(id(member) not in object_order for member in members)
        ):
            raise RuntimeError("Crafter 1.8.3 chunk internals changed incompatibly")
        stable_chunks[chunk] = _InsertionOrderedObjectSet(
            sorted(members, key=lambda member: object_order[id(member)])
        )
    world._chunks = stable_chunks


def _discount(information: Mapping[str, object]) -> float:
    if "discount" not in information:
        raise RuntimeError("Crafter information omitted discount")
    discount = _number(information["discount"], "discount")
    if discount not in {0.0, 1.0}:
        raise RuntimeError("Crafter returned an invalid discount")
    return discount


def _achievements(information: Mapping[str, object]) -> dict[str, int]:
    value = information.get("achievements")
    if type(value) is not dict or set(value) != set(ACHIEVEMENTS):
        raise RuntimeError("Crafter returned invalid achievements")
    achievements: dict[str, int] = {}
    for name in ACHIEVEMENTS:
        count = value[name]
        if type(count) is not int or count < 0:
            raise RuntimeError("Crafter returned an invalid achievement count")
        achievements[name] = count
    return achievements


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Crafter returned a non-numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Crafter returned a non-finite {name}")
    return number


__all__ = ["CrafterEnvironment"]
