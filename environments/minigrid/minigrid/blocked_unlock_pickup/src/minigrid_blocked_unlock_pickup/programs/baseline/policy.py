"""A public-observation planner for BlockedUnlockPickup."""

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
_DIRECTION_VECTORS: tuple[Position, ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class BaselinePolicy:
    """Move the blocker, unlock the room, and collect the mission box."""

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
        self._target: tuple[int, int] | None = None
        self._carried: tuple[int, int] | None = None

    def act(self, observation: PolicyValue) -> PolicyValue:
        image, direction, mission = self._read_observation(observation)
        target = self._read_target(mission)
        if self._target is None:
            self._target = target
        elif self._target != target:
            raise ValueError("mission changed during the Episode")
        self._integrate(image, direction)

        if self._carried is not None:
            carried_type, carried_color = self._carried
            if carried_type == self._objects["key"]:
                door = self._known_door(
                    color=carried_color,
                    state=self._states["locked"],
                )
                if door is not None:
                    return self._interact(
                        door,
                        action=_TOGGLE,
                        direction=direction,
                    )
            return self._drop_safely(direction)

        target_position = self._known_matching(target)
        if (
            target_position is not None
            and self._best_approach(target_position) is not None
        ):
            return self._interact(
                target_position,
                action=_PICKUP,
                direction=direction,
            )

        locked_door = self._known_door(state=self._states["locked"])
        if locked_door is not None:
            blocker = self._door_blocker(locked_door)
            if blocker is not None:
                return self._interact(
                    blocker,
                    action=_PICKUP,
                    direction=direction,
                )
            door_color = self._grid[locked_door][1]
            key = self._known_matching(
                (self._objects["key"], door_color)
            )
            if key is not None:
                return self._interact(
                    key,
                    action=_PICKUP,
                    direction=direction,
                )
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

    def _read_target(self, mission: str) -> tuple[int, int]:
        prefix = "pick up the "
        if not mission.startswith(prefix):
            raise ValueError("observation mission is invalid")
        parts = mission.removeprefix(prefix).split(" ")
        if (
            len(parts) != 2
            or parts[0] not in self._colors
            or parts[1] not in {"box", "key"}
        ):
            raise ValueError("observation mission is invalid")
        return self._objects[parts[1]], self._colors[parts[0]]

    def _integrate(self, image: TensorValue, direction: int) -> None:
        agent_x, agent_y = self._agent_view_position
        forward_x, forward_y = _DIRECTION_VECTORS[direction]
        right_x, right_y = _DIRECTION_VECTORS[(direction + 1) % 4]
        carried_types = {
            self._objects["key"],
            self._objects["ball"],
            self._objects["box"],
        }
        self._carried = None
        for view_x in range(self._view_size):
            for view_y in range(self._view_size):
                object_code, color_code, state_code = self._cell_at(
                    image,
                    view_x,
                    view_y,
                )
                if (view_x, view_y) == self._agent_view_position:
                    if object_code in carried_types:
                        self._carried = (object_code, color_code)
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

    def _cell_at(
        self,
        image: TensorValue,
        x: int,
        y: int,
    ) -> Cell:
        offset = (x * self._view_size + y) * 3
        return (
            image.data[offset],
            image.data[offset + 1],
            image.data[offset + 2],
        )

    def _known_matching(
        self,
        target: tuple[int, int],
    ) -> Position | None:
        return next(
            (
                position
                for position, cell in self._grid.items()
                if cell[:2] == target
            ),
            None,
        )

    def _known_door(
        self,
        *,
        color: int | None = None,
        state: int | None = None,
    ) -> Position | None:
        return next(
            (
                position
                for position, cell in self._grid.items()
                if cell[0] == self._objects["door"]
                and (color is None or cell[1] == color)
                and (state is None or cell[2] == state)
            ),
            None,
        )

    def _door_blocker(self, door: Position) -> Position | None:
        for _, neighbor in _neighbors(door):
            cell = self._grid.get(neighbor)
            if (
                cell is not None
                and cell[0] == self._objects["ball"]
                and self._best_approach(neighbor) is not None
            ):
                return neighbor
        return None

    def _interact(
        self,
        target: Position,
        *,
        action: int,
        direction: int,
    ) -> int:
        approach = self._best_approach(target)
        if approach is None:
            return self._explore(direction)
        approach_position, path = approach
        if path:
            return self._follow(path, direction)
        target_direction = _direction_between(approach_position, target)
        turn = _turn_toward(direction, target_direction)
        if turn is not None:
            return turn
        if action == _PICKUP:
            cell = self._grid[target]
            self._carried = (cell[0], cell[1])
            self._grid[target] = (
                self._objects["empty"],
                self._colors["red"],
                self._states["open"],
            )
        elif action == _TOGGLE:
            cell = self._grid[target]
            self._grid[target] = (
                self._objects["door"],
                cell[1],
                self._states["open"],
            )
        return action

    def _drop_safely(self, direction: int) -> int:
        plan = self._best_drop_plan(prefer_away_from_doors=True)
        if plan is None:
            plan = self._best_drop_plan(prefer_away_from_doors=False)
        if plan is None:
            return self._explore(direction)
        approach, drop_position, path = plan
        if path:
            return self._follow(path, direction)
        drop_direction = _direction_between(approach, drop_position)
        turn = _turn_toward(direction, drop_direction)
        if turn is not None:
            return turn
        if self._carried is None:
            raise ValueError("carried object disappeared")
        self._grid[drop_position] = (
            self._carried[0],
            self._carried[1],
            self._states["open"],
        )
        self._carried = None
        return _DROP

    def _best_drop_plan(
        self,
        *,
        prefer_away_from_doors: bool,
    ) -> tuple[Position, Position, tuple[Position, ...]] | None:
        doors = tuple(
            position
            for position, cell in self._grid.items()
            if cell[0] == self._objects["door"]
        )
        candidates: list[
            tuple[int, Position, Position, tuple[Position, ...]]
        ] = []
        for drop_position, cell in self._grid.items():
            if cell[0] not in {
                self._objects["empty"],
                self._objects["floor"],
            }:
                continue
            if prefer_away_from_doors and any(
                max(
                    abs(drop_position[0] - door[0]),
                    abs(drop_position[1] - door[1]),
                )
                <= 1
                for door in doors
            ):
                continue
            for _, approach in _neighbors(drop_position):
                if not self._traversable(approach):
                    continue
                path = self._shortest_path(approach)
                if path is not None:
                    candidates.append(
                        (
                            len(path),
                            approach,
                            drop_position,
                            path,
                        )
                    )
        if not candidates:
            return None
        _, approach, drop_position, path = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return approach, drop_position, path

    def _best_approach(
        self,
        target: Position,
    ) -> tuple[Position, tuple[Position, ...]] | None:
        candidates: list[tuple[int, Position, tuple[Position, ...]]] = []
        for _, neighbor in _neighbors(target):
            if not self._traversable(neighbor):
                continue
            path = self._shortest_path(neighbor)
            if path is not None:
                candidates.append((len(path), neighbor, path))
        if not candidates:
            return None
        _, position, path = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return position, path

    def _navigate(
        self,
        destination: Position,
        direction: int,
    ) -> int | None:
        path = self._shortest_path(destination)
        if not path:
            return None
        return self._follow(path, direction)

    def _follow(
        self,
        path: tuple[Position, ...],
        direction: int,
    ) -> int:
        next_position = path[0]
        target_direction = _direction_between(
            self._position,
            next_position,
        )
        turn = _turn_toward(direction, target_direction)
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
        frontiers: list[tuple[int, Position, int]] = []
        for position, distance in distances.items():
            for unknown_direction, neighbor in _neighbors(position):
                if neighbor not in self._grid:
                    frontiers.append(
                        (distance, position, unknown_direction)
                    )
        if not frontiers:
            return _RIGHT
        _, frontier, frontier_direction = min(
            frontiers,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if frontier != self._position:
            action = self._navigate(frontier, direction)
            if action is not None:
                return action
        turn = _turn_toward(direction, frontier_direction)
        if turn is not None:
            return turn
        return _RIGHT

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
        if cell is None:
            return False
        return bool(
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


def make_policy(context: PolicyContext) -> BaselinePolicy:
    parameters = context.environment_parameters
    raw_objects = parameters.get("object_encoding")
    raw_colors = parameters.get("color_encoding")
    raw_states = parameters.get("state_encoding")
    raw_view_size = parameters.get("view_size")
    raw_agent_position = parameters.get("agent_view_position")
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
        type(raw_objects) is not dict
        or set(raw_objects) != object_names
        or any(type(value) is not int for value in raw_objects.values())
    ):
        raise ValueError("object_encoding is invalid")
    if (
        type(raw_colors) is not dict
        or set(raw_colors)
        != {"red", "green", "blue", "purple", "yellow", "grey"}
        or any(type(value) is not int for value in raw_colors.values())
    ):
        raise ValueError("color_encoding is invalid")
    if (
        type(raw_states) is not dict
        or set(raw_states) != {"open", "closed", "locked"}
        or any(type(value) is not int for value in raw_states.values())
    ):
        raise ValueError("state_encoding is invalid")
    if type(raw_view_size) is not int or raw_view_size != 7:
        raise ValueError("view_size is invalid")
    if (
        type(raw_agent_position) is not list
        or raw_agent_position != [3, 6]
    ):
        raise ValueError("agent_view_position is invalid")
    return BaselinePolicy(
        object_encoding={
            key: cast(int, value) for key, value in raw_objects.items()
        },
        color_encoding={
            key: cast(int, value) for key, value in raw_colors.items()
        },
        state_encoding={
            key: cast(int, value) for key, value in raw_states.items()
        },
        view_size=raw_view_size,
        agent_view_position=(3, 6),
    )


def _neighbors(position: Position) -> tuple[tuple[int, Position], ...]:
    return tuple(
        (
            direction,
            (position[0] + delta[0], position[1] + delta[1]),
        )
        for direction, delta in enumerate(_DIRECTION_VECTORS)
    )


def _direction_between(start: Position, end: Position) -> int:
    delta = (end[0] - start[0], end[1] - start[1])
    try:
        return _DIRECTION_VECTORS.index(delta)
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
    reversed_path: list[Position] = []
    current: Position | None = destination
    while current is not None:
        reversed_path.append(current)
        current = previous[current]
    reversed_path.reverse()
    return tuple(reversed_path[1:])
