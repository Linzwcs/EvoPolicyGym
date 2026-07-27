"""A finite-state baseline that remembers the cue object."""

from __future__ import annotations

from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

_LEFT = 0
_RIGHT = 1
_FORWARD = 2
_WEST = 2
_EAST = 0
_NORTH = 3
_SOUTH = 1
_WALL = 2


class BaselinePolicy:
    """Solve the canonical T-maze from egocentric observations only."""

    def __init__(
        self,
        *,
        object_encoding: dict[str, int],
        view_size: int,
    ) -> None:
        self._object_encoding = object_encoding
        self._view_size = view_size
        self._target: int | None = None
        self._phase = "find_cue"

    def act(self, observation: PolicyValue) -> PolicyValue:
        image, direction = self._read_observation(observation)

        if self._phase == "find_cue":
            visible = self._visible_targets(image)
            if visible:
                self._target = visible[0]
            if direction != _WEST:
                return _RIGHT
            if self._wall_ahead(image):
                if self._target is None:
                    raise ValueError("cue object was not observed")
                self._phase = "reach_junction"
                return _LEFT
            return _FORWARD

        if self._phase == "reach_junction":
            if direction != _EAST:
                return _LEFT
            if self._wall_ahead(image):
                self._phase = "choose_branch"
                return _LEFT
            return _FORWARD

        if self._phase == "choose_branch":
            if direction != _NORTH:
                return _LEFT
            assert self._target is not None
            if self._target in self._visible_targets(image):
                self._phase = "finish"
                return _FORWARD
            self._phase = "turn_south"
            return _RIGHT

        if self._phase == "turn_south":
            if direction != _SOUTH:
                return _RIGHT
            self._phase = "finish"
            return _FORWARD

        return _FORWARD

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

    def _wall_ahead(self, image: TensorValue) -> bool:
        center = self._view_size // 2
        return self._object_at(image, center, self._view_size - 2) == _WALL

    def _visible_targets(self, image: TensorValue) -> tuple[int, ...]:
        targets = {
            self._object_encoding["key"],
            self._object_encoding["ball"],
        }
        visible: list[int] = []
        for x in range(self._view_size):
            for y in range(self._view_size):
                object_code = self._object_at(image, x, y)
                if object_code in targets:
                    visible.append(object_code)
        return tuple(visible)

    def _object_at(self, image: TensorValue, x: int, y: int) -> int:
        offset = (x * self._view_size + y) * 3
        return image.data[offset]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    parameters = context.environment_parameters
    raw_encoding = parameters.get("object_encoding")
    raw_view_size = parameters.get("view_size")
    if type(raw_encoding) is not dict:
        raise ValueError("object_encoding is invalid")
    encoding = raw_encoding
    if (
        set(encoding) != {
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
        or any(type(value) is not int for value in encoding.values())
    ):
        raise ValueError("object_encoding is invalid")
    if type(raw_view_size) is not int or raw_view_size != 7:
        raise ValueError("view_size is invalid")
    return BaselinePolicy(
        object_encoding={
            key: cast(int, value) for key, value in encoding.items()
        },
        view_size=raw_view_size,
    )
