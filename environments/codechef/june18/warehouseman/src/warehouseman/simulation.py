"""Independent deterministic implementation of the WAREHOUS rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

MIN_ROWS = 6
MAX_ROWS = 20
MIN_COLUMNS = 6
MAX_COLUMNS = 20
MAX_INSTRUCTION_CHARACTERS = 500_000
GENERATOR_ID = "evopolicygym-independent-v1"

_GENERATOR_DOMAIN = b"evopolicygym-warehouseman/generator/v1\0"
_DIRECTION_DELTAS = {
    "N": (-1, 0),
    "W": (0, -1),
    "S": (1, 0),
    "E": (0, 1),
}


class InvalidInstruction(Exception):
    """The submitted instruction string violates the public rules."""


@dataclass(frozen=True, slots=True)
class WarehouseCase:
    """One public warehouse layout and shipment arrival order."""

    rows: int
    columns: int
    arrivals: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.rows) is not int or not MIN_ROWS <= self.rows <= MAX_ROWS:
            raise ValueError("rows is outside the official range")
        if (
            type(self.columns) is not int
            or not MIN_COLUMNS <= self.columns <= MAX_COLUMNS
        ):
            raise ValueError("columns is outside the official range")
        shipment_count = self.rows * self.columns - 1
        if (
            len(self.arrivals) != shipment_count
            or any(type(item) is not int for item in self.arrivals)
            or set(self.arrivals) != set(range(1, shipment_count + 1))
        ):
            raise ValueError("arrivals must permute every shipment")
        if self.arrivals[-1] == 1:
            raise ValueError("shipment 1 cannot arrive last")


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

    def shuffle(self, values: list[int]) -> None:
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


def generate_case(seed: int) -> WarehouseCase:
    """Generate one case from the published WAREHOUS distribution."""

    random = SeedStream(seed)
    rows = MIN_ROWS + random.below(MAX_ROWS - MIN_ROWS + 1)
    columns = MIN_COLUMNS + random.below(MAX_COLUMNS - MIN_COLUMNS + 1)
    arrivals = list(range(1, rows * columns))
    while True:
        random.shuffle(arrivals)
        if arrivals[-1] != 1:
            break
    return WarehouseCase(rows, columns, tuple(arrivals))


def normalized_cost(
    instruction_characters: int,
    rows: int,
    columns: int,
) -> float:
    """Return the official per-case normalized instruction cost."""

    return (
        (instruction_characters + 2) / (rows + columns - 1)
        - 2 * rows * columns
        + 20
    )


class WarehouseSimulation:
    """Execute one complete WAREHOUS instruction submission."""

    __slots__ = (
        "_board",
        "_carrying",
        "_case",
        "_drops",
        "_loads",
        "_moves",
        "_picks",
        "_position",
        "_shipments_dropped",
        "_shipments_picked",
        "_unloads",
    )

    def __init__(self, case: WarehouseCase) -> None:
        if type(case) is not WarehouseCase:
            raise TypeError("case must be WarehouseCase")
        self._case = case
        self._board: list[int | None] = [None] * (case.rows * case.columns)
        self._position = 0
        self._carrying: int | None = None
        self._shipments_picked = 0
        self._shipments_dropped = 0
        self._moves = 0
        self._picks = 0
        self._drops = 0
        self._loads = 0
        self._unloads = 0

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def picks(self) -> int:
        return self._picks

    @property
    def drops(self) -> int:
        return self._drops

    @property
    def loads(self) -> int:
        return self._loads

    @property
    def unloads(self) -> int:
        return self._unloads

    def execute(self, instructions: str) -> None:
        """Execute and require one complete valid solution."""

        if type(instructions) is not str:
            raise TypeError("instructions must be text")
        if len(instructions) > MAX_INSTRUCTION_CHARACTERS:
            raise InvalidInstruction()
        try:
            instructions.encode("ascii")
        except UnicodeEncodeError:
            raise InvalidInstruction() from None

        index = 0
        while index < len(instructions):
            opcode = instructions[index]
            if opcode in _DIRECTION_DELTAS:
                self._move(opcode)
                index += 1
            elif opcode == "P":
                self._pick()
                index += 1
            elif opcode == "D":
                self._drop()
                index += 1
            elif opcode in {"L", "U"}:
                if index + 1 >= len(instructions):
                    raise InvalidInstruction()
                direction = instructions[index + 1]
                if direction not in _DIRECTION_DELTAS:
                    raise InvalidInstruction()
                if opcode == "L":
                    self._load(direction)
                else:
                    self._unload(direction)
                index += 2
            else:
                raise InvalidInstruction()

        shipment_count = len(self._case.arrivals)
        if (
            self._shipments_picked != shipment_count
            or self._shipments_dropped != shipment_count
            or self._position != 0
            or self._carrying is not None
            or any(item is not None for item in self._board)
        ):
            raise InvalidInstruction()

    def _move(self, direction: str) -> None:
        destination = self._adjacent(direction)
        if self._board[destination] is not None:
            raise InvalidInstruction()
        self._position = destination
        self._moves += 1

    def _pick(self) -> None:
        if (
            self._position != 0
            or self._carrying is not None
            or self._shipments_picked >= len(self._case.arrivals)
        ):
            raise InvalidInstruction()
        self._carrying = self._case.arrivals[self._shipments_picked]
        self._shipments_picked += 1
        self._picks += 1

    def _drop(self) -> None:
        expected = self._shipments_dropped + 1
        if (
            self._position != 0
            or self._carrying != expected
            or self._shipments_picked != len(self._case.arrivals)
        ):
            raise InvalidInstruction()
        self._carrying = None
        self._shipments_dropped += 1
        self._drops += 1

    def _load(self, direction: str) -> None:
        if self._carrying is not None:
            raise InvalidInstruction()
        source = self._adjacent(direction)
        shipment = self._board[source]
        if shipment is None:
            raise InvalidInstruction()
        self._board[source] = None
        self._carrying = shipment
        self._loads += 1

    def _unload(self, direction: str) -> None:
        if self._carrying is None:
            raise InvalidInstruction()
        destination = self._adjacent(direction)
        if self._board[destination] is not None:
            raise InvalidInstruction()
        self._board[destination] = self._carrying
        self._carrying = None
        self._unloads += 1

    def _adjacent(self, direction: str) -> int:
        row, column = divmod(self._position, self._case.columns)
        row_delta, column_delta = _DIRECTION_DELTAS[direction]
        next_row = row + row_delta
        next_column = column + column_delta
        if (
            not 0 <= next_row < self._case.rows
            or not 0 <= next_column < self._case.columns
        ):
            raise InvalidInstruction()
        return next_row * self._case.columns + next_column


__all__ = [
    "GENERATOR_ID",
    "MAX_COLUMNS",
    "MAX_INSTRUCTION_CHARACTERS",
    "MAX_ROWS",
    "MIN_COLUMNS",
    "MIN_ROWS",
    "InvalidInstruction",
    "WarehouseCase",
    "WarehouseSimulation",
    "generate_case",
    "normalized_cost",
]
