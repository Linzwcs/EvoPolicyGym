"""Independent deterministic implementation of the AHC054 turn rules."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

type Position = tuple[int, int]

MIN_SIZE = 20
MAX_SIZE = 40
GENERATOR_ID = "evopolicygym-independent-v1"

_GENERATOR_DOMAIN = b"evopolicygym-treants-forest/generator/v1\0"
_DIRECTIONS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class InvalidPlacement(Exception):
    """A complete placement set violates the Environment rules."""


@dataclass(frozen=True, slots=True)
class ForestCase:
    """One generated forest and its private adventurer target order."""

    size: int
    flower: Position
    initial_trees: frozenset[Position]
    target_order: tuple[Position, ...]

    def __post_init__(self) -> None:
        if type(self.size) is not int or not MIN_SIZE <= self.size <= MAX_SIZE:
            raise ValueError("size is outside the official range")
        entrance = (0, self.size // 2)
        if not _in_bounds(self.flower, self.size):
            raise ValueError("flower is outside the forest")
        if _manhattan(entrance, self.flower) < 5:
            raise ValueError("flower is too close to the entrance")
        if entrance in self.initial_trees or self.flower in self.initial_trees:
            raise ValueError("entrance and flower must be empty")
        if any(not _in_bounds(cell, self.size) for cell in self.initial_trees):
            raise ValueError("initial tree is outside the forest")
        expected_targets = {
            (row, column)
            for row in range(self.size)
            for column in range(self.size)
            if (row, column) != entrance
        }
        if (
            len(self.target_order) != len(expected_targets)
            or set(self.target_order) != expected_targets
        ):
            raise ValueError("target_order must permute every non-entrance cell")
        if not _all_empty_cells_connected(
            self.size,
            self.initial_trees,
            entrance,
        ):
            raise ValueError("all initial empty cells must be connected")

    @property
    def entrance(self) -> Position:
        return (0, self.size // 2)


class SeedStream:
    """Small version-stable random stream for split-derived Episode seeds."""

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

    def shuffle(self, values: list[Position]) -> None:
        for index in range(len(values) - 1, 0, -1):
            selected = self.below(index + 1)
            values[index], values[selected] = values[selected], values[index]

    def _next_u64(self) -> int:
        digest = hashlib.sha256()
        digest.update(_GENERATOR_DOMAIN)
        digest.update(self._seed.to_bytes(8, "big"))
        digest.update(self._counter.to_bytes(8, "big"))
        self._counter += 1
        return int.from_bytes(digest.digest()[:8], "big")


def generate_case(seed: int) -> ForestCase:
    """Generate one independent case using the public AHC054 distribution."""

    random = SeedStream(seed)
    size = MIN_SIZE + random.below(MAX_SIZE - MIN_SIZE + 1)
    entrance = (0, size // 2)
    candidates = [
        (row, column) for row in range(size) for column in range(size) if (row, column) != entrance
    ]
    random.shuffle(candidates)
    target_tree_count = max(1, random.below(size * size // 6 + 1))
    trees: set[Position] = set()
    for candidate in candidates:
        trees.add(candidate)
        if not _all_empty_cells_connected(size, trees, entrance):
            trees.remove(candidate)
        if len(trees) >= target_tree_count:
            break

    flower_candidates = [
        (row, column)
        for row in range(size)
        for column in range(size)
        if (row, column) not in trees and _manhattan(entrance, (row, column)) >= 5
    ]
    flower = flower_candidates[random.below(len(flower_candidates))]
    target_order = list(candidates)
    random.shuffle(target_order)
    return ForestCase(
        size=size,
        flower=flower,
        initial_trees=frozenset(trees),
        target_order=tuple(target_order),
    )


class ForestSimulation:
    """Stateful adventurer simulation for one fresh Episode."""

    __slots__ = (
        "_case",
        "_done",
        "_newly_revealed",
        "_placed_treants",
        "_position",
        "_revealed",
        "_target",
        "_target_index",
        "_trees",
        "_turn",
    )

    def __init__(self, case: ForestCase) -> None:
        if type(case) is not ForestCase:
            raise TypeError("case must be ForestCase")
        self._case = case
        self._trees = set(case.initial_trees)
        self._placed_treants: set[Position] = set()
        self._position = case.entrance
        self._revealed = {case.entrance}
        self._newly_revealed: tuple[Position, ...] = (case.entrance,)
        self._target: Position | None = None
        self._target_index = 0
        self._turn = 0
        self._done = False

    @property
    def case(self) -> ForestCase:
        return self._case

    @property
    def position(self) -> Position:
        return self._position

    @property
    def newly_revealed(self) -> tuple[Position, ...]:
        return self._newly_revealed

    @property
    def revealed_count(self) -> int:
        return len(self._revealed)

    @property
    def flower_revealed(self) -> bool:
        return self._case.flower in self._revealed

    @property
    def legal_candidate_count(self) -> int:
        """Return unseen non-tree, non-flower cells before connectivity checks."""

        return sum(
            (row, column) not in self._revealed
            and (row, column) not in self._trees
            and (row, column) != self._case.flower
            for row in range(self._case.size)
            for column in range(self._case.size)
        )

    @property
    def flower_path_length(self) -> int:
        """Return the actual current shortest path to the public flower."""

        reachable = _reachable_empty(
            self._case.size,
            self._trees,
            self._position,
        )
        if self._case.flower not in reachable:
            raise RuntimeError("flower became unreachable")
        distances = _actual_distances(
            self._case.size,
            self._trees,
            self._position,
        )
        return distances[self._case.flower]

    @property
    def placed_count(self) -> int:
        return len(self._placed_treants)

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def done(self) -> bool:
        return self._done

    def step(self, placements: tuple[Position, ...]) -> None:
        if self._done:
            raise RuntimeError("simulation is already complete")
        prospective = self._validate_placements(placements)
        self._trees = prospective
        self._placed_treants.update(placements)
        self._turn += 1
        self._reveal()
        self._select_target()
        distances = self._distances_from(self._target)
        next_position: Position | None = None
        next_distance: int | None = None
        for row_delta, column_delta in _DIRECTIONS:
            candidate = (
                self._position[0] + row_delta,
                self._position[1] + column_delta,
            )
            distance = distances.get(candidate)
            if distance is not None and (next_distance is None or distance < next_distance):
                next_position = candidate
                next_distance = distance
        if next_position is None:
            raise RuntimeError("adventurer has no valid next position")
        self._position = next_position
        self._done = self._position == self._case.flower

    def _validate_placements(
        self,
        placements: tuple[Position, ...],
    ) -> set[Position]:
        if len(set(placements)) != len(placements):
            raise InvalidPlacement()
        prospective = set(self._trees)
        for cell in placements:
            if (
                not _in_bounds(cell, self._case.size)
                or cell in self._revealed
                or cell in prospective
                or cell == self._case.flower
            ):
                raise InvalidPlacement()
            prospective.add(cell)
        if not _path_exists(
            self._case.size,
            prospective,
            self._case.entrance,
            self._case.flower,
        ) or not _path_exists(
            self._case.size,
            prospective,
            self._position,
            self._case.flower,
        ):
            raise InvalidPlacement()
        return prospective

    def _reveal(self) -> None:
        newly_revealed: list[Position] = []
        for row_delta, column_delta in _DIRECTIONS:
            row, column = self._position
            while 0 <= row < self._case.size and 0 <= column < self._case.size:
                cell = (row, column)
                if cell not in self._revealed:
                    self._revealed.add(cell)
                    newly_revealed.append(cell)
                if cell in self._trees:
                    break
                row += row_delta
                column += column_delta
        self._newly_revealed = tuple(newly_revealed)

    def _select_target(self) -> None:
        if self._case.flower in self._revealed:
            self._target = self._case.flower
        if self._target is not None:
            distances = self._distances_from(self._target)
            if self._position not in distances:
                self._target = None
        if (
            self._target is not None
            and self._target != self._case.flower
            and self._target in self._revealed
        ):
            self._target = None
        if self._target is not None:
            return

        reachable = self._distances_from(self._position)
        while self._target_index < len(self._case.target_order):
            candidate = self._case.target_order[self._target_index]
            self._target_index += 1
            if candidate not in self._revealed and candidate in reachable:
                self._target = candidate
                return
        raise RuntimeError("adventurer could not select a target")

    def _distances_from(self, source: Position | None) -> dict[Position, int]:
        if source is None:
            return {}
        distances = {source: 0}
        queue: deque[Position] = deque((source,))
        while queue:
            row, column = queue.popleft()
            next_distance = distances[(row, column)] + 1
            for row_delta, column_delta in _DIRECTIONS:
                candidate = (row + row_delta, column + column_delta)
                if (
                    candidate not in distances
                    and _in_bounds(candidate, self._case.size)
                    and self._tentatively_empty(candidate)
                ):
                    distances[candidate] = next_distance
                    queue.append(candidate)
        return distances

    def _tentatively_empty(self, cell: Position) -> bool:
        return cell not in self._revealed or cell not in self._trees


def _all_empty_cells_connected(
    size: int,
    trees: set[Position] | frozenset[Position],
    entrance: Position,
) -> bool:
    reachable = _reachable_empty(size, trees, entrance)
    return len(reachable) == size * size - len(trees)


def _path_exists(
    size: int,
    trees: set[Position],
    start: Position,
    destination: Position,
) -> bool:
    return destination in _reachable_empty(size, trees, start)


def _reachable_empty(
    size: int,
    trees: set[Position] | frozenset[Position],
    start: Position,
) -> set[Position]:
    if start in trees:
        return set()
    reached = {start}
    queue: deque[Position] = deque((start,))
    while queue:
        row, column = queue.popleft()
        for row_delta, column_delta in _DIRECTIONS:
            candidate = (row + row_delta, column + column_delta)
            if candidate not in reached and candidate not in trees and _in_bounds(candidate, size):
                reached.add(candidate)
                queue.append(candidate)
    return reached


def _actual_distances(
    size: int,
    trees: set[Position] | frozenset[Position],
    start: Position,
) -> dict[Position, int]:
    if start in trees:
        return {}
    distances = {start: 0}
    queue: deque[Position] = deque((start,))
    while queue:
        row, column = queue.popleft()
        next_distance = distances[(row, column)] + 1
        for row_delta, column_delta in _DIRECTIONS:
            candidate = (row + row_delta, column + column_delta)
            if (
                candidate not in distances
                and candidate not in trees
                and _in_bounds(candidate, size)
            ):
                distances[candidate] = next_distance
                queue.append(candidate)
    return distances


def _in_bounds(cell: Position, size: int) -> bool:
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


__all__ = [
    "GENERATOR_ID",
    "MAX_SIZE",
    "MIN_SIZE",
    "ForestCase",
    "ForestSimulation",
    "InvalidPlacement",
    "Position",
    "generate_case",
]
