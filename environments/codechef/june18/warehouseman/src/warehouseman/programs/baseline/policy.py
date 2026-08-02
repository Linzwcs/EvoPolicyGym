"""Complete constructive baseline for the full WAREHOUS size range."""

from __future__ import annotations

from collections import deque
from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue

_DIRECTIONS = ("N", "W", "S", "E")
_OPPOSITE = {"N": "S", "W": "E", "S": "N", "E": "W"}
_MAX_INSTRUCTION_CHARACTERS = 500_000


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        rows, columns, arrivals = _decode_observation(observation)
        return _solve(rows, columns, arrivals)


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()


def _decode_observation(
    observation: PolicyValue,
) -> tuple[int, int, tuple[int, ...]]:
    if type(observation) is not dict or set(observation) != {
        "rows",
        "columns",
        "arrivals",
        "instruction_limit",
    }:
        raise ValueError("observation is invalid")
    rows = observation["rows"]
    columns = observation["columns"]
    raw_arrivals = observation["arrivals"]
    limit = observation["instruction_limit"]
    if (
        type(rows) is not int
        or type(columns) is not int
        or not 6 <= rows <= 20
        or not 6 <= columns <= 20
        or type(limit) is not int
        or limit != _MAX_INSTRUCTION_CHARACTERS
        or type(raw_arrivals) is not list
        or any(type(item) is not int for item in raw_arrivals)
    ):
        raise ValueError("observation is invalid")
    arrivals = tuple(cast(list[int], raw_arrivals))
    if (
        len(arrivals) != rows * columns - 1
        or set(arrivals) != set(range(1, rows * columns))
        or arrivals[-1] == 1
    ):
        raise ValueError("arrival order is invalid")
    return rows, columns, arrivals


def _solve(rows: int, columns: int, arrivals: tuple[int, ...]) -> str:
    cell_count = rows * columns
    path = _serpentine_path(rows, columns)
    board: list[int | None] = [None] * cell_count
    positions: dict[int, int] = {}
    instructions: list[str] = []

    for arrival_index, shipment in enumerate(arrivals):
        target_index = cell_count - 1 - arrival_index
        instructions.append("P")
        for index in range(target_index - 1):
            instructions.append(
                _direction(path[index], path[index + 1], columns)
            )
        instructions.append(
            "U"
            + _direction(
                path[target_index - 1],
                path[target_index],
                columns,
            )
        )
        for index in range(target_index - 1, 0, -1):
            instructions.append(
                _direction(path[index], path[index - 1], columns)
            )
        target = path[target_index]
        board[target] = shipment
        positions[shipment] = target

    adjacency = _adjacency(rows, columns)
    toward_goal = _toward_entrance_table(adjacency, cell_count)
    for shipment in range(1, cell_count):
        forklift = 0
        target = positions[shipment]
        while forklift != 0 or target not in adjacency[0]:
            next_forklift = toward_goal[forklift * cell_count + target]
            if next_forklift < 0:
                raise RuntimeError("sliding construction has no route")
            direction = _direction(forklift, next_forklift, columns)
            moving = board[next_forklift]
            if moving is None:
                instructions.append(direction)
            else:
                instructions.extend(
                    (
                        "L" + direction,
                        direction,
                        "U" + _OPPOSITE[direction],
                    )
                )
                board[forklift] = moving
                positions[moving] = forklift
                board[next_forklift] = None
                if moving == shipment:
                    target = forklift
            forklift = next_forklift

        direction = _direction(0, target, columns)
        instructions.extend(("L" + direction, "D"))
        board[target] = None
        del positions[shipment]

    solution = "".join(instructions)
    if len(solution) > _MAX_INSTRUCTION_CHARACTERS:
        raise RuntimeError("constructive solution exceeds instruction limit")
    return solution


def _serpentine_path(rows: int, columns: int) -> tuple[int, ...]:
    cells: list[int] = []
    for row in range(rows):
        column_range = (
            range(columns)
            if row % 2 == 0
            else range(columns - 1, -1, -1)
        )
        cells.extend(row * columns + column for column in column_range)
    return tuple(cells)


def _adjacency(rows: int, columns: int) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for cell in range(rows * columns):
        row, column = divmod(cell, columns)
        neighbors: list[int] = []
        if row > 0:
            neighbors.append(cell - columns)
        if column > 0:
            neighbors.append(cell - 1)
        if row + 1 < rows:
            neighbors.append(cell + columns)
        if column + 1 < columns:
            neighbors.append(cell + 1)
        result.append(tuple(neighbors))
    return tuple(result)


def _toward_entrance_table(
    adjacency: tuple[tuple[int, ...], ...],
    cell_count: int,
) -> list[int]:
    toward = [-1] * (cell_count * cell_count)
    queue: deque[tuple[int, int]] = deque()
    for target in adjacency[0]:
        toward[target] = 0
        queue.append((0, target))

    while queue:
        forklift, target = queue.popleft()
        for next_forklift in adjacency[forklift]:
            next_target = forklift if next_forklift == target else target
            code = next_forklift * cell_count + next_target
            if toward[code] < 0:
                toward[code] = forklift
                queue.append((next_forklift, next_target))
    return toward


def _direction(source: int, destination: int, columns: int) -> str:
    delta = destination - source
    if delta == -columns:
        return "N"
    if delta == -1:
        return "W"
    if delta == columns:
        return "S"
    if delta == 1:
        return "E"
    raise RuntimeError("cells are not side-adjacent")
