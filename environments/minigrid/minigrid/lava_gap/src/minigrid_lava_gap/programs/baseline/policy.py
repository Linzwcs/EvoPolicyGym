"""A public-observation safe exploration baseline for LavaGap."""

from __future__ import annotations

from collections import deque
from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

type Position = tuple[int, int]
type Cell = tuple[int, int]

_LEFT = 0
_RIGHT = 1
_FORWARD = 2
_VECTORS: tuple[Position, ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class BaselinePolicy:
    """Reveal unknown cells without entering them and plan through openings."""

    def __init__(
        self,
        *,
        object_encoding: dict[str, int],
        state_encoding: dict[str, int],
        view_size: int,
        agent_view_position: Position,
        size: int,
    ) -> None:
        self._objects = object_encoding
        self._states = state_encoding
        self._view_size = view_size
        self._agent_view_position = agent_view_position
        self._position = (0, 0)
        self._goal = (size - 3, size - 3)
        self._grid: dict[Position, Cell] = {
            self._position: (self._objects["empty"], self._states["open"]),
            self._goal: (self._objects["goal"], self._states["open"]),
        }

    def act(self, observation: PolicyValue) -> PolicyValue:
        image, direction = self._read_observation(observation)
        self._integrate(image, direction)
        action = self._navigate(self._goal, direction)
        if action is not None:
            return action
        return self._explore(direction)

    def _read_observation(
        self,
        observation: PolicyValue,
    ) -> tuple[TensorValue, int]:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        image = observation.get("image")
        direction = observation.get("direction")
        if (
            type(image) is not TensorValue
            or image.shape != (self._view_size, self._view_size, 3)
            or image.dtype != "uint8"
            or type(direction) is not int
            or not 0 <= direction <= 3
        ):
            raise ValueError("observation is invalid")
        return image, direction

    def _integrate(self, image: TensorValue, direction: int) -> None:
        agent_x, agent_y = self._agent_view_position
        forward_x, forward_y = _VECTORS[direction]
        right_x, right_y = _VECTORS[(direction + 1) % 4]
        for view_x in range(self._view_size):
            for view_y in range(self._view_size):
                offset = (view_x * self._view_size + view_y) * 3
                object_code = image.data[offset]
                state_code = image.data[offset + 2]
                if (view_x, view_y) == self._agent_view_position:
                    self._grid[self._position] = (
                        self._objects["empty"],
                        self._states["open"],
                    )
                    continue
                if object_code == self._objects["unseen"]:
                    continue
                side = view_x - agent_x
                forward = agent_y - view_y
                world = (
                    self._position[0]
                    + forward * forward_x
                    + side * right_x,
                    self._position[1]
                    + forward * forward_y
                    + side * right_y,
                )
                self._grid[world] = (object_code, state_code)

    def _navigate(
        self,
        destination: Position,
        direction: int,
    ) -> int | None:
        path = self._shortest_path(destination)
        return None if not path else self._follow(path, direction)

    def _follow(
        self,
        path: tuple[Position, ...],
        direction: int,
    ) -> int:
        next_position = path[0]
        turn = _turn_toward(
            direction,
            _direction_between(self._position, next_position),
        )
        if turn is not None:
            return turn
        self._position = next_position
        return _FORWARD

    def _shortest_path(
        self,
        destination: Position,
    ) -> tuple[Position, ...] | None:
        if destination == self._position:
            return ()
        queue: deque[Position] = deque((self._position,))
        previous: dict[Position, Position | None] = {
            self._position: None
        }
        while queue:
            current = queue.popleft()
            for _, neighbor in _neighbors(current):
                if neighbor in previous or not self._traversable(neighbor):
                    continue
                previous[neighbor] = current
                if neighbor == destination:
                    return _reconstruct(previous, destination)
                queue.append(neighbor)
        return None

    def _explore(self, direction: int) -> int:
        distances = self._reachable_distances()
        frontiers = [
            (distance, position, frontier_direction)
            for position, distance in distances.items()
            for frontier_direction, neighbor in _neighbors(position)
            if neighbor not in self._grid
        ]
        if not frontiers:
            return _RIGHT
        _, frontier, frontier_direction = min(frontiers)
        if frontier != self._position:
            action = self._navigate(frontier, direction)
            if action is not None:
                return action
        turn = _turn_toward(direction, frontier_direction)
        return _RIGHT if turn is None else turn

    def _reachable_distances(self) -> dict[Position, int]:
        queue: deque[Position] = deque((self._position,))
        distances = {self._position: 0}
        while queue:
            current = queue.popleft()
            for _, neighbor in _neighbors(current):
                if neighbor in distances or not self._traversable(neighbor):
                    continue
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
        return distances

    def _traversable(self, position: Position) -> bool:
        cell = self._grid.get(position)
        return bool(
            cell is not None
            and cell[0]
            in {
                self._objects["empty"],
                self._objects["floor"],
                self._objects["goal"],
            }
        )


def make_policy(context: PolicyContext) -> BaselinePolicy:
    values = context.environment_parameters
    objects = values.get("object_encoding")
    states = values.get("state_encoding")
    view_size = values.get("view_size")
    agent_position = values.get("agent_view_position")
    size = values.get("size")
    names = {
        "unseen",
        "empty",
        "wall",
        "floor",
        "door",
        "key",
        "ball",
        "box",
        "goal",
        "lava",
        "agent",
    }
    if (
        type(objects) is not dict
        or set(objects) != names
        or any(type(value) is not int for value in objects.values())
    ):
        raise ValueError("object_encoding is invalid")
    if (
        type(states) is not dict
        or set(states) != {"open", "closed", "locked"}
        or any(type(value) is not int for value in states.values())
    ):
        raise ValueError("state_encoding is invalid")
    if type(view_size) is not int or view_size != 7:
        raise ValueError("view_size is invalid")
    if type(agent_position) is not list or agent_position != [3, 6]:
        raise ValueError("agent_view_position is invalid")
    if type(size) is not int or size not in {5, 6, 7}:
        raise ValueError("size is invalid")
    return BaselinePolicy(
        object_encoding={
            key: cast(int, value) for key, value in objects.items()
        },
        state_encoding={
            key: cast(int, value) for key, value in states.items()
        },
        view_size=view_size,
        agent_view_position=(3, 6),
        size=size,
    )


def _neighbors(position: Position) -> tuple[tuple[int, Position], ...]:
    return tuple(
        (
            direction,
            (position[0] + delta[0], position[1] + delta[1]),
        )
        for direction, delta in enumerate(_VECTORS)
    )


def _direction_between(start: Position, end: Position) -> int:
    delta = (end[0] - start[0], end[1] - start[1])
    try:
        return _VECTORS.index(delta)
    except ValueError as error:
        raise ValueError("positions must be cardinally adjacent") from error


def _turn_toward(current: int, target: int) -> int | None:
    delta = (target - current) % 4
    if delta == 0:
        return None
    return _LEFT if delta == 3 else _RIGHT


def _reconstruct(
    previous: dict[Position, Position | None],
    destination: Position,
) -> tuple[Position, ...]:
    path: list[Position] = []
    current: Position | None = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return tuple(path[1:])
