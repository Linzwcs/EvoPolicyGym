"""Independent deterministic implementation of the AHC058 turn rules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

MACHINE_IDS = 10
LEVELS = 4
TURNS = 500
INITIAL_APPLES = 1
GENERATOR_ID = "evopolicygym-independent-v1"

_GENERATOR_DOMAIN = b"evopolicygym-apple-incremental/generator/v1\0"


class InvalidUpgrade(Exception):
    """A requested machine upgrade violates the public rules."""


@dataclass(frozen=True, slots=True)
class AppleCase:
    """One public AHC058 capacity and initial-cost configuration."""

    capacities: tuple[int, ...]
    costs: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.capacities) != MACHINE_IDS
            or any(
                type(value) is not int or not 1 <= value <= 100
                for value in self.capacities
            )
            or tuple(sorted(self.capacities)) != self.capacities
        ):
            raise ValueError("capacities must be ten sorted integers in [1, 100]")
        if (
            len(self.costs) != LEVELS
            or any(len(row) != MACHINE_IDS for row in self.costs)
            or any(
                type(value) is not int
                or not 1 <= value <= 1_250_000_000_000
                for row in self.costs
                for value in row
            )
            or self.costs[0][0] != 1
        ):
            raise ValueError("initial costs are invalid")


class SeedStream:
    """Small version-stable random stream for generated public cases."""

    __slots__ = ("_counter", "_seed")

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self._seed = seed
        self._counter = 0

    def unit(self) -> float:
        return self._next_u64() / 2**64

    def _next_u64(self) -> int:
        digest = hashlib.sha256()
        digest.update(_GENERATOR_DOMAIN)
        digest.update(self._seed.to_bytes(8, "big"))
        digest.update(self._counter.to_bytes(8, "big"))
        self._counter += 1
        return int.from_bytes(digest.digest()[:8], "big")


def generate_case(seed: int) -> AppleCase:
    """Generate one case from the published AHC058 distribution."""

    random = SeedStream(seed)
    capacities = [1]
    capacities.extend(_nearest_integer(10 ** (2 * random.unit())) for _ in range(9))
    capacities.sort()
    costs: list[tuple[int, ...]] = []
    for level in range(LEVELS):
        row: list[int] = []
        for machine_id, capacity in enumerate(capacities):
            if level == 0 and machine_id == 0:
                row.append(1)
            else:
                row.append(
                    _nearest_integer(
                        capacity
                        * 500**level
                        * 10 ** (2 * random.unit())
                    )
                )
        costs.append(tuple(row))
    return AppleCase(tuple(capacities), tuple(costs))


def final_score(apples: int) -> int:
    """Return the published per-case log2 score."""

    if type(apples) is not int or apples <= 0:
        raise ValueError("apples must be a positive integer")
    return math.floor(100_000 * math.log2(float(apples)) + 0.5)


class AppleSimulation:
    """Stateful simulation for one fresh 500-turn Episode."""

    __slots__ = (
        "_apples",
        "_case",
        "_counts",
        "_powers",
        "_turn",
        "_upgrades",
    )

    def __init__(self, case: AppleCase) -> None:
        if type(case) is not AppleCase:
            raise TypeError("case must be AppleCase")
        self._case = case
        self._apples = INITIAL_APPLES
        self._counts = [
            [1 for _ in range(MACHINE_IDS)]
            for _ in range(LEVELS)
        ]
        self._powers = [
            [0 for _ in range(MACHINE_IDS)]
            for _ in range(LEVELS)
        ]
        self._turn = 0
        self._upgrades = 0

    @property
    def case(self) -> AppleCase:
        return self._case

    @property
    def apples(self) -> int:
        return self._apples

    @property
    def counts(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._counts)

    @property
    def powers(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._powers)

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def upgrades(self) -> int:
        return self._upgrades

    @property
    def done(self) -> bool:
        return self._turn == TURNS

    def step(self, upgrade: tuple[int, int] | None) -> None:
        if self.done:
            raise RuntimeError("simulation is already complete")
        if upgrade is not None:
            level, machine_id = upgrade
            if not 0 <= level < LEVELS or not 0 <= machine_id < MACHINE_IDS:
                raise InvalidUpgrade()
            cost = self._case.costs[level][machine_id] * (
                self._powers[level][machine_id] + 1
            )
            if cost > self._apples:
                raise InvalidUpgrade()
            self._apples -= cost
            self._powers[level][machine_id] += 1
            self._upgrades += 1

        for level in range(LEVELS):
            for machine_id in range(MACHINE_IDS):
                produced = (
                    self._counts[level][machine_id]
                    * self._powers[level][machine_id]
                )
                if level == 0:
                    self._apples += (
                        self._case.capacities[machine_id] * produced
                    )
                else:
                    self._counts[level - 1][machine_id] += produced
        self._turn += 1


def _nearest_integer(value: float) -> int:
    return math.floor(value + 0.5)


__all__ = [
    "GENERATOR_ID",
    "INITIAL_APPLES",
    "LEVELS",
    "MACHINE_IDS",
    "TURNS",
    "AppleCase",
    "AppleSimulation",
    "InvalidUpgrade",
    "final_score",
    "generate_case",
]
