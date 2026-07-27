"""A public-observation replanning DynamicObstacles baseline."""

from __future__ import annotations

from collections import deque
from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

type Position = tuple[int, int]
type Cell = tuple[int, int]

_LEFT = 0
_RIGHT = 1
_FORWARD = 2
_DIRECTION_VECTORS: tuple[Position, ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class BaselinePolicy:
    """Map the room and replan around moving obstacle observations."""

    def __init__(
        self,
        *,
        object_encoding: dict[str, int],
        state_encoding: dict[str, int],
        view_size: int,
        agent_view_position: Position,
        size: int,
        random_start: bool,
    ) -> None:
        self._objects = object_encoding
        self._states = state_encoding
        self._view_size = view_size
        self._agent_view_position = agent_view_position
        self._position = (0, 0)
        self._grid: dict[Position, Cell] = {
            self._position: (self._objects["empty"], self._states["open"])
        }
        self._static_grid: dict[Position, Cell] = dict(self._grid)
        self._pending_position: Position | None = None
        self._conservative_motion = size >= 16 and not random_start
        if not random_start:
            self._initialize_known_room(size)

    def _initialize_known_room(self, size: int) -> None:
        empty = (self._objects["empty"], self._states["open"])
        wall = (self._objects["wall"], self._states["open"])
        last_interior = size - 3
        for x in range(last_interior + 1):
            for y in range(last_interior + 1):
                self._grid[(x, y)] = empty
                self._static_grid[(x, y)] = empty
        goal = (self._objects["goal"], self._states["open"])
        self._grid[(last_interior, last_interior)] = goal
        self._static_grid[(last_interior, last_interior)] = goal
        for coordinate in range(-1, size - 1):
            for position in (
                (-1, coordinate),
                (size - 2, coordinate),
                (coordinate, -1),
                (coordinate, size - 2),
            ):
                self._grid[position] = wall
                self._static_grid[position] = wall

    def act(self, observation: PolicyValue) -> PolicyValue:
        image, direction = self._read_observation(observation)
        self._reconcile_position(image, direction)
        self._integrate(image, direction)

        goal = self._known_object("goal")
        if goal is not None:
            action = self._navigate(goal, direction)
            if action is not None:
                return action
        return self._explore(direction)

    def _read_observation(
        self,
        observation: PolicyValue,
    ) -> tuple[TensorValue, int]:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        if set(observation) != {"image", "direction", "mission"}:
            raise ValueError("observation fields are invalid")
        image = observation["image"]
        direction = observation["direction"]
        if (
            type(image) is not TensorValue
            or image.dtype != "uint8"
            or image.shape != (self._view_size, self._view_size, 3)
        ):
            raise ValueError("observation image is invalid")
        if type(direction) is not int or not 0 <= direction <= 3:
            raise ValueError("observation direction is invalid")
        return image, direction

    def _integrate(self, image: TensorValue, direction: int) -> None:
        agent_x, agent_y = self._agent_view_position
        forward_x, forward_y = _DIRECTION_VECTORS[direction]
        right_x, right_y = _DIRECTION_VECTORS[(direction + 1) % 4]

        for view_x in range(self._view_size):
            for view_y in range(self._view_size):
                object_code, _, state_code = self._cell_at(
                    image,
                    view_x,
                    view_y,
                )
                if (view_x, view_y) == self._agent_view_position:
                    cell = (
                        self._objects["empty"],
                        self._states["open"],
                    )
                    self._grid[self._position] = cell
                    self._static_grid[self._position] = cell
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
                if object_code != self._objects["ball"]:
                    self._static_grid[world] = (object_code, state_code)

    def _reconcile_position(
        self,
        image: TensorValue,
        direction: int,
    ) -> None:
        pending = self._pending_position
        if pending is None:
            return
        old_score = self._alignment_score(
            image,
            direction,
            self._position,
        )
        pending_score = self._alignment_score(
            image,
            direction,
            pending,
        )
        if pending_score >= old_score:
            self._position = pending
        self._pending_position = None

    def _alignment_score(
        self,
        image: TensorValue,
        direction: int,
        candidate: Position,
    ) -> int:
        agent_x, agent_y = self._agent_view_position
        forward_x, forward_y = _DIRECTION_VECTORS[direction]
        right_x, right_y = _DIRECTION_VECTORS[(direction + 1) % 4]
        score = 0
        ignored = {
            self._objects["unseen"],
            self._objects["ball"],
            self._objects["agent"],
        }
        for view_x in range(self._view_size):
            for view_y in range(self._view_size):
                if (view_x, view_y) == self._agent_view_position:
                    continue
                object_code, _, state_code = self._cell_at(
                    image,
                    view_x,
                    view_y,
                )
                if object_code in ignored:
                    continue
                side = view_x - agent_x
                forward = agent_y - view_y
                world = (
                    candidate[0]
                    + forward * forward_x
                    + side * right_x,
                    candidate[1]
                    + forward * forward_y
                    + side * right_y,
                )
                known = self._static_grid.get(world)
                if known is None:
                    continue
                score += 2 if known == (object_code, state_code) else -3
        return score

    def _cell_at(
        self,
        image: TensorValue,
        x: int,
        y: int,
    ) -> tuple[int, int, int]:
        offset = (x * self._view_size + y) * 3
        return (
            image.data[offset],
            image.data[offset + 1],
            image.data[offset + 2],
        )

    def _known_object(self, name: str) -> Position | None:
        code = self._objects[name]
        return next(
            (
                position
                for position, cell in self._grid.items()
                if cell[0] == code
            ),
            None,
        )

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
        if (
            self._conservative_motion
            and not self._safe_after_obstacle_motion(next_position)
        ):
            return _RIGHT
        if self._conservative_motion:
            self._position = next_position
            return _FORWARD
        self._pending_position = next_position
        return _FORWARD

    def _safe_after_obstacle_motion(self, destination: Position) -> bool:
        obstacle = self._objects["ball"]
        return all(
            cell[0] != obstacle
            or max(
                abs(position[0] - destination[0]),
                abs(position[1] - destination[1]),
            )
            > 1
            for position, cell in self._grid.items()
        )

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
        return cell[0] in {
            self._objects["empty"],
            self._objects["floor"],
            self._objects["goal"],
        }


def make_policy(context: PolicyContext) -> BaselinePolicy:
    parameters = context.environment_parameters
    raw_objects = parameters.get("object_encoding")
    raw_states = parameters.get("state_encoding")
    raw_view_size = parameters.get("view_size")
    raw_agent_position = parameters.get("agent_view_position")
    raw_size = parameters.get("size")
    raw_random_start = parameters.get("random_start")
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
    if type(raw_size) is not int or raw_size < 5:
        raise ValueError("size is invalid")
    if type(raw_random_start) is not bool:
        raise ValueError("random_start is invalid")

    return BaselinePolicy(
        object_encoding={
            key: cast(int, value) for key, value in raw_objects.items()
        },
        state_encoding={
            key: cast(int, value) for key, value in raw_states.items()
        },
        view_size=raw_view_size,
        agent_view_position=(3, 6),
        size=raw_size,
        random_start=raw_random_start,
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
