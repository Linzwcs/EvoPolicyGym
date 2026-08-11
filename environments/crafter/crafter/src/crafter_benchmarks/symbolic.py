"""Trusted local-symbolic projection of pinned Crafter 1.8.3 state."""

from __future__ import annotations

import math
from typing import Protocol, cast

import crafter.objects
import numpy as np
from evopolicygym.policy import PolicyValue, TensorValue

from .constants import (
    SYMBOLIC_ENTITY_NAMES,
    SYMBOLIC_FACING_NAMES,
    SYMBOLIC_INVENTORY_KEYS,
    SYMBOLIC_PLAYER_CENTER,
    SYMBOLIC_TERRAIN_NAMES,
    SYMBOLIC_VIEW_SHAPE,
)

_WORLD_SHAPE = (64, 64)
_UPSTREAM_MATERIAL_NAMES = {
    index: name
    for index, name in enumerate((None, *SYMBOLIC_TERRAIN_NAMES[1:]))
}
_UPSTREAM_MATERIAL_IDS = {
    name: index for index, name in _UPSTREAM_MATERIAL_NAMES.items()
}
_FACING_BY_VECTOR = {
    (-1, 0): "left",
    (1, 0): "right",
    (0, -1): "up",
    (0, 1): "down",
}
_ARROW_ID_BY_VECTOR = {
    (-1, 0): 5,
    (1, 0): 6,
    (0, -1): 7,
    (0, 1): 8,
}


class _SymbolicEnvironment(Protocol):
    _world: object
    _player: object


class _SymbolicWorld(Protocol):
    area: object
    daylight: object
    _mat_names: object
    _mat_ids: object
    _mat_map: np.ndarray
    _obj_map: np.ndarray
    _objects: list[object | None]


def local_symbolic_observation(environment: object) -> dict[str, PolicyValue]:
    """Extract one bounded, player-centered observation without mutation."""

    env = cast(_SymbolicEnvironment, environment)
    world = _validated_world(env._world)
    player = _validated_player(env._player)
    position = _position(player, "player")
    terrain, entities = _local_grids(world, position)
    inventory = _inventory(player)
    facing = _direction(getattr(player, "facing", None), "player facing")
    sleeping = getattr(player, "sleeping", None)
    if type(sleeping) is not bool:
        raise RuntimeError("Crafter 1.8.3 player sleeping state changed incompatibly")
    daylight_value = getattr(world, "daylight", None)
    if isinstance(daylight_value, bool) or not isinstance(
        daylight_value, (int, float, np.integer, np.floating)
    ):
        raise RuntimeError("Crafter 1.8.3 daylight changed incompatibly")
    daylight = float(daylight_value)
    if not math.isfinite(daylight) or not 0.0 <= daylight <= 1.0:
        raise RuntimeError("Crafter 1.8.3 daylight is outside [0, 1]")

    return {
        "terrain": _tensor(terrain),
        "entities": _tensor(entities),
        "inventory": inventory,
        "facing": facing,
        "sleeping": sleeping,
        "daylight": daylight,
    }


def symbolic_observation_arrays(
    value: PolicyValue,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.uint8,
    np.bool_,
    np.float64,
]:
    """Validate and decode a live symbolic observation for lossless artifacts."""

    if type(value) is not dict or set(value) != {
        "terrain",
        "entities",
        "inventory",
        "facing",
        "sleeping",
        "daylight",
    }:
        raise ValueError("Crafter symbolic Feedback observation is invalid")
    terrain = _tensor_array(value["terrain"], "terrain")
    entities = _tensor_array(value["entities"], "entities")
    if np.any(terrain >= len(SYMBOLIC_TERRAIN_NAMES)):
        raise ValueError("Crafter symbolic terrain ID is invalid")
    if np.any(entities >= len(SYMBOLIC_ENTITY_NAMES)):
        raise ValueError("Crafter symbolic entity ID is invalid")

    inventory_value = value["inventory"]
    if type(inventory_value) is not dict or set(inventory_value) != set(
        SYMBOLIC_INVENTORY_KEYS
    ):
        raise ValueError("Crafter symbolic inventory is invalid")
    inventory = np.empty(len(SYMBOLIC_INVENTORY_KEYS), dtype=np.uint8)
    for index, name in enumerate(SYMBOLIC_INVENTORY_KEYS):
        amount = inventory_value[name]
        if type(amount) is not int or not 0 <= amount <= 9:
            raise ValueError("Crafter symbolic inventory amount is invalid")
        inventory[index] = amount

    facing_value = value["facing"]
    if type(facing_value) is not str or facing_value not in SYMBOLIC_FACING_NAMES:
        raise ValueError("Crafter symbolic facing is invalid")
    sleeping_value = value["sleeping"]
    if type(sleeping_value) is not bool:
        raise ValueError("Crafter symbolic sleeping state is invalid")
    daylight_value = value["daylight"]
    if type(daylight_value) is not float or not 0.0 <= daylight_value <= 1.0:
        raise ValueError("Crafter symbolic daylight is invalid")
    return (
        terrain,
        entities,
        inventory,
        np.uint8(SYMBOLIC_FACING_NAMES.index(facing_value)),
        np.bool_(sleeping_value),
        np.float64(daylight_value),
    )


def _validated_world(value: object) -> _SymbolicWorld:
    if value is None:
        raise RuntimeError("Crafter 1.8.3 world is unavailable")
    area = getattr(value, "area", None)
    if not isinstance(area, (tuple, list, np.ndarray)) or tuple(area) != _WORLD_SHAPE:
        raise RuntimeError("Crafter 1.8.3 world area changed incompatibly")
    material_names = getattr(value, "_mat_names", None)
    material_ids = getattr(value, "_mat_ids", None)
    if material_names != _UPSTREAM_MATERIAL_NAMES:
        raise RuntimeError("Crafter 1.8.3 material names changed incompatibly")
    if material_ids != _UPSTREAM_MATERIAL_IDS:
        raise RuntimeError("Crafter 1.8.3 material IDs changed incompatibly")
    material_map = getattr(value, "_mat_map", None)
    object_map = getattr(value, "_obj_map", None)
    objects = getattr(value, "_objects", None)
    if (
        type(material_map) is not np.ndarray
        or material_map.dtype != np.dtype("uint8")
        or material_map.shape != _WORLD_SHAPE
        or type(object_map) is not np.ndarray
        or object_map.dtype != np.dtype("uint32")
        or object_map.shape != _WORLD_SHAPE
        or type(objects) is not list
        or not objects
        or objects[0] is not None
    ):
        raise RuntimeError("Crafter 1.8.3 world maps changed incompatibly")
    return cast(_SymbolicWorld, value)


def _validated_player(value: object) -> object:
    if type(value) is not crafter.objects.Player:
        raise RuntimeError("Crafter 1.8.3 player changed incompatibly")
    return value


def _local_grids(
    world: _SymbolicWorld,
    player_position: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    terrain = np.zeros(SYMBOLIC_VIEW_SHAPE, dtype=np.uint8)
    entities = np.zeros(SYMBOLIC_VIEW_SHAPE, dtype=np.uint8)
    material_map = world._mat_map
    object_map = world._obj_map
    objects = world._objects
    center_row, center_column = SYMBOLIC_PLAYER_CENTER
    player_x, player_y = player_position
    for row in range(SYMBOLIC_VIEW_SHAPE[0]):
        world_y = player_y + row - center_row
        for column in range(SYMBOLIC_VIEW_SHAPE[1]):
            world_x = player_x + column - center_column
            if not (0 <= world_x < _WORLD_SHAPE[0] and 0 <= world_y < _WORLD_SHAPE[1]):
                continue
            terrain[row, column] = material_map[world_x, world_y]
            object_index = int(object_map[world_x, world_y])
            if not 0 <= object_index < len(objects):
                raise RuntimeError("Crafter 1.8.3 object map changed incompatibly")
            if object_index:
                entity = objects[object_index]
                if entity is None or bool(getattr(entity, "removed", False)):
                    raise RuntimeError("Crafter 1.8.3 object map is inconsistent")
                if _position(entity, "entity") != (world_x, world_y):
                    raise RuntimeError("Crafter 1.8.3 entity position is inconsistent")
                entities[row, column] = _entity_id(entity)
    if entities[SYMBOLIC_PLAYER_CENTER] != 1:
        raise RuntimeError("Crafter local symbolic crop omitted the player")
    return terrain, entities


def _entity_id(value: object) -> int:
    kind = type(value)
    if kind is crafter.objects.Player:
        return 1
    if kind is crafter.objects.Cow:
        return 2
    if kind is crafter.objects.Zombie:
        return 3
    if kind is crafter.objects.Skeleton:
        return 4
    if kind is crafter.objects.Arrow:
        direction = _vector(getattr(value, "facing", None), "arrow facing")
        try:
            return _ARROW_ID_BY_VECTOR[direction]
        except KeyError as error:
            raise RuntimeError("Crafter arrow direction changed incompatibly") from error
    if kind is crafter.objects.Plant:
        ripe = getattr(value, "ripe", None)
        if type(ripe) is not bool:
            raise RuntimeError("Crafter plant ripeness changed incompatibly")
        return 10 if ripe else 9
    if kind is crafter.objects.Fence:
        return 11
    raise RuntimeError("Crafter 1.8.3 exposed an unknown local entity type")


def _inventory(player: object) -> dict[str, PolicyValue]:
    value = getattr(player, "inventory", None)
    if type(value) is not dict or tuple(value) != SYMBOLIC_INVENTORY_KEYS:
        raise RuntimeError("Crafter 1.8.3 inventory changed incompatibly")
    inventory: dict[str, PolicyValue] = {}
    for name in SYMBOLIC_INVENTORY_KEYS:
        amount = value[name]
        if type(amount) is not int or not 0 <= amount <= 9:
            raise RuntimeError("Crafter 1.8.3 inventory amount is invalid")
        inventory[name] = amount
    return inventory


def _position(value: object, name: str) -> tuple[int, int]:
    return _vector(getattr(value, "pos", None), f"{name} position")


def _direction(value: object, name: str) -> str:
    vector = _vector(value, name)
    try:
        direction = _FACING_BY_VECTOR[vector]
    except KeyError as error:
        raise RuntimeError(f"Crafter {name} changed incompatibly") from error
    if direction not in SYMBOLIC_FACING_NAMES:
        raise RuntimeError(f"Crafter {name} changed incompatibly")
    return direction


def _vector(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
        raise RuntimeError(f"Crafter {name} changed incompatibly")
    components: list[int] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, np.integer)):
            raise RuntimeError(f"Crafter {name} changed incompatibly")
        components.append(int(component))
    return components[0], components[1]


def _tensor(value: np.ndarray) -> TensorValue:
    return TensorValue(
        dtype="uint8",
        shape=SYMBOLIC_VIEW_SHAPE,
        data=np.ascontiguousarray(value).tobytes(order="C"),
    )


def _tensor_array(value: PolicyValue, name: str) -> np.ndarray:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != SYMBOLIC_VIEW_SHAPE
    ):
        raise ValueError(f"Crafter symbolic {name} tensor is invalid")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(SYMBOLIC_VIEW_SHAPE)


__all__ = ["local_symbolic_observation", "symbolic_observation_arrays"]
