"""A public-observation mapping and planning BabyAI baseline."""

from __future__ import annotations

from collections import deque
from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

type Position = tuple[int, int]
type Cell = tuple[int, int, int]

_LEFT = 0
_RIGHT = 1
_FORWARD = 2
_PICKUP = 3
_DROP = 4
_TOGGLE = 5
_VECTORS: tuple[Position, ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class BaselinePolicy:
    """Open doors and explore every reachable room."""

    def __init__(
        self,
        *,
        object_encoding: dict[str, int],
        color_encoding: dict[str, int],
        state_encoding: dict[str, int],
        view_size: int,
        agent_view_position: Position,
    ) -> None:
        self._objects = object_encoding
        self._colors = color_encoding
        self._states = state_encoding
        self._view_size = view_size
        self._agent_view_position = agent_view_position
        self._position = (0, 0)
        self._grid: dict[Position, Cell] = {
            self._position: (
                self._objects["empty"],
                self._colors["red"],
                self._states["open"],
            )
        }
        self._carried: Cell | None = None
        self._moved_objects: set[Position] = set()
    def act(self, observation: PolicyValue) -> PolicyValue:
        image, direction, mission = self._read_observation(observation)
        if not mission:
            raise ValueError("observation mission is invalid")
        self._integrate(image, direction)

        closed = self._nearest_door(self._states["closed"])
        if closed is not None:
            return self._open(closed, direction)
        if not self._has_frontier():
            if self._carried is not None:
                return self._drop(direction)
            portable = self._nearest_portable()
            if portable is not None:
                return self._pick_up(portable, direction)
        return self._explore(direction)

    def _read_observation(
        self,
        observation: PolicyValue,
    ) -> tuple[TensorValue, int, str]:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        if set(observation) != {"image", "direction", "mission"}:
            raise ValueError("observation fields are invalid")
        image = observation["image"]
        direction = observation["direction"]
        mission = observation["mission"]
        if (
            type(image) is not TensorValue
            or image.dtype != "uint8"
            or image.shape != (self._view_size, self._view_size, 3)
        ):
            raise ValueError("observation image is invalid")
        if type(direction) is not int or not 0 <= direction <= 3:
            raise ValueError("observation direction is invalid")
        if type(mission) is not str:
            raise ValueError("observation mission is invalid")
        return image, direction, mission

    def _integrate(self, image: TensorValue, direction: int) -> None:
        agent_x, agent_y = self._agent_view_position
        forward_x, forward_y = _VECTORS[direction]
        right_x, right_y = _VECTORS[(direction + 1) % 4]
        for view_x in range(self._view_size):
            for view_y in range(self._view_size):
                object_code, color_code, state_code = self._cell(
                    image,
                    view_x,
                    view_y,
                )
                if (view_x, view_y) == self._agent_view_position:
                    self._grid[self._position] = (
                        self._objects["empty"],
                        self._colors["red"],
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
                self._grid[world] = (
                    object_code,
                    color_code,
                    state_code,
                )

    def _cell(self, image: TensorValue, x: int, y: int) -> Cell:
        offset = (x * self._view_size + y) * 3
        return (
            image.data[offset],
            image.data[offset + 1],
            image.data[offset + 2],
        )

    def _nearest_door(self, state: int) -> Position | None:
        candidates: list[tuple[int, Position]] = []
        for position, cell in self._grid.items():
            if (
                cell[0] != self._objects["door"]
                or cell[2] != state
            ):
                continue
            approach = self._best_approach(position)
            if approach is not None:
                candidates.append((len(approach[1]), position))
        return (
            None
            if not candidates
            else min(candidates, key=lambda item: (item[0], item[1]))[1]
        )

    def _nearest_portable(self) -> Position | None:
        candidates: list[tuple[int, Position]] = []
        portable = {
            self._objects["key"],
            self._objects["ball"],
            self._objects["box"],
        }
        for position, cell in self._grid.items():
            if position in self._moved_objects or cell[0] not in portable:
                continue
            approach = self._best_approach(position)
            if approach is not None:
                candidates.append((len(approach[1]), position))
        return (
            None
            if not candidates
            else min(candidates, key=lambda item: (item[0], item[1]))[1]
        )

    def _open(
        self,
        target: Position,
        direction: int,
    ) -> int:
        approach = self._best_approach(target)
        if approach is None:
            return self._explore(direction)
        approach_position, path = approach
        if path:
            return self._follow(path, direction)
        turn = _turn_toward(
            direction,
            _direction_between(approach_position, target),
        )
        if turn is not None:
            return turn
        cell = self._grid[target]
        self._grid[target] = (
            self._objects["door"],
            cell[1],
            self._states["open"],
        )
        return _TOGGLE

    def _pick_up(self, target: Position, direction: int) -> int:
        approach = self._best_approach(target)
        if approach is None:
            return _RIGHT
        approach_position, path = approach
        if path:
            return self._follow(path, direction)
        turn = _turn_toward(
            direction,
            _direction_between(approach_position, target),
        )
        if turn is not None:
            return turn
        self._carried = self._grid[target]
        self._grid[target] = (
            self._objects["empty"],
            self._colors["red"],
            self._states["open"],
        )
        return _PICKUP

    def _drop(self, direction: int) -> int:
        if self._carried is None:
            return _RIGHT
        candidates = [
            (target_direction, position)
            for target_direction, position in _neighbors(self._position)
            if self._grid.get(position, (None, None, None))[0]
            == self._objects["empty"]
        ]
        if not candidates:
            return _RIGHT
        target_direction, position = min(candidates)
        turn = _turn_toward(direction, target_direction)
        if turn is not None:
            return turn
        self._grid[position] = self._carried
        self._moved_objects.add(position)
        self._carried = None
        return _DROP

    def _best_approach(
        self,
        target: Position,
    ) -> tuple[Position, tuple[Position, ...]] | None:
        candidates: list[tuple[int, Position, tuple[Position, ...]]] = []
        for _, neighbor in _neighbors(target):
            if self._traversable(neighbor):
                path = self._shortest_path(neighbor)
                if path is not None:
                    candidates.append((len(path), neighbor, path))
        if not candidates:
            return None
        _, position, path = min(candidates)
        return position, path

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
            (distance, position, unknown_direction)
            for position, distance in distances.items()
            for unknown_direction, neighbor in _neighbors(position)
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

    def _has_frontier(self) -> bool:
        return any(
            neighbor not in self._grid
            for position in self._reachable_distances()
            for _, neighbor in _neighbors(position)
        )

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
            and (
                cell[0]
                in {
                    self._objects["empty"],
                    self._objects["floor"],
                    self._objects["goal"],
                }
                or (
                    cell[0] == self._objects["door"]
                    and cell[2] == self._states["open"]
                )
            )
        )


def make_policy(context: PolicyContext) -> BaselinePolicy:
    values = context.environment_parameters
    objects = values.get("object_encoding")
    colors = values.get("color_encoding")
    states = values.get("state_encoding")
    view_size = values.get("view_size")
    agent_position = values.get("agent_view_position")
    object_names = {
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
        or set(objects) != object_names
        or any(type(value) is not int for value in objects.values())
    ):
        raise ValueError("object_encoding is invalid")
    if (
        type(colors) is not dict
        or set(colors)
        != {"red", "green", "blue", "purple", "yellow", "grey"}
        or any(type(value) is not int for value in colors.values())
    ):
        raise ValueError("color_encoding is invalid")
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
    return BaselinePolicy(
        object_encoding={
            key: cast(int, value) for key, value in objects.items()
        },
        color_encoding={
            key: cast(int, value) for key, value in colors.items()
        },
        state_encoding={
            key: cast(int, value) for key, value in states.items()
        },
        view_size=view_size,
        agent_view_position=(3, 6),
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
