"""Intentionally weak local-symbolic Crafter starting Policy."""

import random
from collections import deque

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

_ACTIONS = frozenset(range(17))
_INVENTORY_KEYS = (
    "health",
    "food",
    "drink",
    "energy",
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)
_MOVEMENT = (1, 2, 3, 4)
_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


class LocalSymbolicBaselinePolicy:
    """Validate the symbolic ABI, then mirror the weak RGB fallback exactly."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self._actions: deque[int] = deque()
        self._previous_direction: int | None = None

    def act(self, observation: PolicyValue) -> int:
        _decode(observation)

        if not self._actions:
            candidates = tuple(
                direction
                for direction in _MOVEMENT
                if self._previous_direction is None
                or direction != _OPPOSITE[self._previous_direction]
            )
            direction = self._random.choice(candidates)
            self._actions.extend([direction] * self._random.randint(2, 5))
            self._actions.append(5)
            self._previous_direction = direction
        action = self._actions.popleft()
        if type(action) is not int or action not in _ACTIONS:
            raise ValueError("Crafter baseline selected an invalid Action")
        return action


def _decode(
    observation: PolicyValue,
) -> tuple[bytes, bytes, dict[str, int], str]:
    if type(observation) is not dict or set(observation) != {
        "terrain",
        "entities",
        "inventory",
        "facing",
        "sleeping",
        "daylight",
    }:
        raise ValueError("Crafter local-symbolic observation is invalid")
    terrain = observation["terrain"]
    entities = observation["entities"]
    if (
        type(terrain) is not TensorValue
        or terrain.dtype != "uint8"
        or terrain.shape != (7, 9)
        or type(entities) is not TensorValue
        or entities.dtype != "uint8"
        or entities.shape != (7, 9)
    ):
        raise ValueError("Crafter local-symbolic spatial tensors are invalid")
    if any(value > 12 for value in terrain.data):
        raise ValueError("Crafter local-symbolic terrain ID is invalid")
    if any(value > 11 for value in entities.data):
        raise ValueError("Crafter local-symbolic entity ID is invalid")
    inventory_value = observation["inventory"]
    if type(inventory_value) is not dict or set(inventory_value) != set(_INVENTORY_KEYS):
        raise ValueError("Crafter local-symbolic inventory is invalid")
    inventory: dict[str, int] = {}
    for name in _INVENTORY_KEYS:
        amount = inventory_value[name]
        if type(amount) is not int or not 0 <= amount <= 9:
            raise ValueError("Crafter local-symbolic inventory amount is invalid")
        inventory[name] = amount
    facing = observation["facing"]
    if type(facing) is not str or facing not in {"left", "right", "up", "down"}:
        raise ValueError("Crafter local-symbolic facing is invalid")
    if type(observation["sleeping"]) is not bool:
        raise ValueError("Crafter local-symbolic sleeping state is invalid")
    daylight = observation["daylight"]
    if type(daylight) is not float or not 0.0 <= daylight <= 1.0:
        raise ValueError("Crafter local-symbolic daylight is invalid")
    return terrain.data, entities.data, inventory, facing


def make_policy(context: PolicyContext) -> LocalSymbolicBaselinePolicy:
    if context.environment_parameters.get("observation_profile") != "local-symbolic-v1":
        raise ValueError("Crafter observation profile is invalid")
    if context.environment_parameters.get("symbolic_view_rows") != 7:
        raise ValueError("Crafter symbolic rows are invalid")
    if context.environment_parameters.get("symbolic_view_columns") != 9:
        raise ValueError("Crafter symbolic columns are invalid")
    return LocalSymbolicBaselinePolicy(context.policy_seed)
