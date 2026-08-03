"""Independent deterministic implementation of the AHC057 rules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

POINTS = 300
TURNS = 1_000
TARGET_COMPONENTS = 10
TARGET_SIZE = 30
SPACE_SIZE = 100_000
MIN_VELOCITY = -100
MAX_VELOCITY = 100
GENERATOR_ID = "evopolicygym-independent-v1"

_GENERATOR_DOMAIN = b"evopolicygym-molecules/generator/v1\0"

type Position = tuple[float, float]
type Velocity = tuple[float, float]
type Bond = tuple[int, int]


class InvalidBond(Exception):
    """A complete turn's bond set violates the public rules."""


@dataclass(frozen=True, slots=True)
class MoleculesCase:
    """One public initial point and velocity configuration."""

    positions: tuple[tuple[int, int], ...]
    velocities: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.positions) != POINTS or any(
            type(x) is not int
            or type(y) is not int
            or not 0 <= x < SPACE_SIZE
            or not 0 <= y < SPACE_SIZE
            for x, y in self.positions
        ):
            raise ValueError("positions are invalid")
        if len(self.velocities) != POINTS or any(
            type(vx) is not int
            or type(vy) is not int
            or not MIN_VELOCITY <= vx <= MAX_VELOCITY
            or not MIN_VELOCITY <= vy <= MAX_VELOCITY
            for vx, vy in self.velocities
        ):
            raise ValueError("velocities are invalid")


class SeedStream:
    """Small version-stable random stream for generated public cases."""

    __slots__ = ("_counter", "_seed")

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self._seed = seed
        self._counter = 0

    def below(self, upper: int) -> int:
        if type(upper) is not int or upper <= 0:
            raise ValueError("upper must be a positive integer")
        limit = 2**64 - (2**64 % upper)
        while True:
            value = self._next_u64()
            if value < limit:
                return value % upper

    def _next_u64(self) -> int:
        digest = hashlib.sha256()
        digest.update(_GENERATOR_DOMAIN)
        digest.update(self._seed.to_bytes(8, "big"))
        digest.update(self._counter.to_bytes(8, "big"))
        self._counter += 1
        return int.from_bytes(digest.digest()[:8], "big")


def generate_case(seed: int) -> MoleculesCase:
    """Generate one case from the published AHC057 distribution."""

    random = SeedStream(seed)
    positions: list[tuple[int, int]] = []
    velocities: list[tuple[int, int]] = []
    for _ in range(POINTS):
        positions.append(
            (
                random.below(SPACE_SIZE),
                random.below(SPACE_SIZE),
            )
        )
        velocities.append(
            (
                MIN_VELOCITY + random.below(MAX_VELOCITY - MIN_VELOCITY + 1),
                MIN_VELOCITY + random.below(MAX_VELOCITY - MIN_VELOCITY + 1),
            )
        )
    return MoleculesCase(tuple(positions), tuple(velocities))


def final_score(total_cost: int) -> int:
    """Return the published per-case logarithmic score."""

    if type(total_cost) is not int or total_cost < 0:
        raise ValueError("total_cost must be a non-negative integer")
    ratio = SPACE_SIZE * (POINTS - TARGET_COMPONENTS) / (total_cost + 1)
    return math.floor(1_000_000 * math.log2(ratio) + 0.5)


class MoleculesSimulation:
    """Stateful bonding and toroidal movement simulation."""

    __slots__ = (
        "_component_velocities",
        "_parent",
        "_positions",
        "_size",
        "_total_bonds",
        "_total_cost",
        "_turn",
    )

    def __init__(self, case: MoleculesCase) -> None:
        if type(case) is not MoleculesCase:
            raise TypeError("case must be MoleculesCase")
        self._positions = [(float(x), float(y)) for x, y in case.positions]
        self._parent = list(range(POINTS))
        self._size = [1] * POINTS
        self._component_velocities = [(float(vx), float(vy)) for vx, vy in case.velocities]
        self._turn = 0
        self._total_cost = 0
        self._total_bonds = 0

    @property
    def positions(self) -> tuple[Position, ...]:
        return tuple(self._positions)

    @property
    def velocities(self) -> tuple[Velocity, ...]:
        return tuple(self._component_velocities[self._find(index)] for index in range(POINTS))

    @property
    def component_labels(self) -> tuple[int, ...]:
        minimums: dict[int, int] = {}
        roots = [self._find(index) for index in range(POINTS)]
        for index, root in enumerate(roots):
            minimums[root] = min(index, minimums.get(root, index))
        return tuple(minimums[root] for root in roots)

    @property
    def component_count(self) -> int:
        return sum(self._find(index) == index for index in range(POINTS))

    @property
    def component_sizes(self) -> tuple[int, ...]:
        return tuple(
            sorted(self._size[index] for index in range(POINTS) if self._find(index) == index)
        )

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def total_cost(self) -> int:
        return self._total_cost

    @property
    def total_bonds(self) -> int:
        return self._total_bonds

    @property
    def component_size_histogram(self) -> tuple[int, ...]:
        sizes = self.component_sizes
        return tuple(sizes.count(size) for size in range(1, TARGET_SIZE + 1))

    def bond_costs(self, bonds: tuple[Bond, ...]) -> tuple[int, ...]:
        """Return public toroidal costs at the current positions."""

        return tuple(
            _bond_cost(self._positions[first], self._positions[second]) for first, second in bonds
        )

    @property
    def done(self) -> bool:
        return self._turn == TURNS

    def step(self, bonds: tuple[Bond, ...]) -> int:
        """Validate one bond set atomically, then move every point."""

        if self.done:
            raise RuntimeError("simulation is already complete")
        parent = list(self._parent)
        sizes = list(self._size)
        velocities = list(self._component_velocities)
        action_cost = 0

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for first, second in sorted(bonds):
            root_a = find(first)
            root_b = find(second)
            if root_a == root_b:
                raise InvalidBond()
            action_cost += _bond_cost(
                self._positions[first],
                self._positions[second],
            )
            size_a = sizes[root_a]
            size_b = sizes[root_b]
            velocity_a = velocities[root_a]
            velocity_b = velocities[root_b]
            combined_size = size_a + size_b
            if combined_size > TARGET_SIZE:
                raise InvalidBond()
            combined_velocity = (
                (size_a * velocity_a[0] + size_b * velocity_b[0]) / combined_size,
                (size_a * velocity_a[1] + size_b * velocity_b[1]) / combined_size,
            )
            parent[root_b] = root_a
            sizes[root_a] = combined_size
            sizes[root_b] = 0
            velocities[root_a] = combined_velocity

        roots = {find(index) for index in range(POINTS)}
        if len(roots) < TARGET_COMPONENTS:
            raise InvalidBond()
        if self._turn == TURNS - 1 and (
            len(roots) != TARGET_COMPONENTS
            or sorted(sizes[root] for root in roots) != [TARGET_SIZE] * TARGET_COMPONENTS
        ):
            raise InvalidBond()

        self._parent = parent
        self._size = sizes
        self._component_velocities = velocities
        if self._turn < TURNS - 1:
            for index, (x, y) in enumerate(self._positions):
                velocity = velocities[find(index)]
                self._positions[index] = (
                    (x + velocity[0]) % SPACE_SIZE,
                    (y + velocity[1]) % SPACE_SIZE,
                )
        self._total_cost += action_cost
        self._total_bonds += len(bonds)
        self._turn += 1
        return action_cost

    def _find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index


def _bond_cost(first: Position, second: Position) -> int:
    delta_x = abs(first[0] - second[0])
    delta_y = abs(first[1] - second[1])
    toroidal_x = min(SPACE_SIZE - delta_x, delta_x)
    toroidal_y = min(SPACE_SIZE - delta_y, delta_y)
    distance = math.sqrt(toroidal_x * toroidal_x + toroidal_y * toroidal_y)
    return math.floor(distance + 0.5)


__all__ = [
    "GENERATOR_ID",
    "POINTS",
    "SPACE_SIZE",
    "TARGET_COMPONENTS",
    "TARGET_SIZE",
    "TURNS",
    "InvalidBond",
    "MoleculesCase",
    "MoleculesSimulation",
    "final_score",
    "generate_case",
]
