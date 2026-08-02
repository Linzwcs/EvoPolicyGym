"""A weak observation-aware exploration baseline for NLE NetHackScore."""

from __future__ import annotations

import random
from collections.abc import Mapping

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

_DIRECTIONS = (
    (1, 0, -1),
    (2, 1, 0),
    (3, 0, 1),
    (4, -1, 0),
    (5, 1, -1),
    (6, 1, 1),
    (7, -1, 1),
    (8, -1, -1),
)
_OPPOSITE = {1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
_RAW_KEY_ACTION = {
    "k": 1,
    "l": 2,
    "j": 3,
    "h": 4,
    "u": 5,
    "n": 6,
    "b": 7,
    "y": 8,
    "K": 9,
    "L": 10,
    "J": 11,
    "H": 12,
    "U": 13,
    "N": 14,
    "B": 15,
    "Y": 16,
    "<": 17,
    ">": 18,
    ".": 19,
    "e": 21,
    "s": 22,
}
_FOOD_WORDS = (
    "food ration",
    "cram ration",
    "lembas wafer",
    "apple",
    "orange",
    "pear",
    "melon",
    "banana",
    "carrot",
    "fortune cookie",
    "cream pie",
    "candy bar",
    "meatball",
    "meat ring",
    "egg",
    "corpse",
)


class BaselinePolicy:
    """Explore low-visit visible cells, search when stuck, and eat known food."""

    def __init__(self, policy_seed: int) -> None:
        self._random = random.Random(policy_seed)
        self._visits: dict[tuple[int, int, int], int] = {}
        self._previous_direction: int | None = None
        self._pending_direction: int | None = None
        self._pending_food_letter: str | None = None
        self._steps = 0

    def act(self, observation: PolicyValue) -> PolicyValue:
        public = _observation(observation)
        stats = public["stats"]
        message = _text(public, "message").lower()
        input_mode = _text(public, "input_mode")
        inventory = public["inventory"]

        if input_mode == "more":
            return 0
        if "in what direction" in message and self._pending_direction is not None:
            action = self._pending_direction
            self._pending_direction = None
            return action
        if "what do you want to eat" in message:
            food_action = _RAW_KEY_ACTION.get(self._pending_food_letter or "")
            self._pending_food_letter = None
            return 0 if food_action is None else food_action
        if input_mode == "yes_no":
            return 6  # raw key "n"
        if input_mode == "get_line":
            return 0

        hunger = _integer(stats, "hunger")
        food_letter = _food_letter(inventory)
        if hunger >= 2 and food_letter is not None:
            self._pending_food_letter = food_letter
            return 21

        x = _integer(stats, "x")
        y = _integer(stats, "y")
        dungeon_level = _integer(stats, "dungeon_level")
        chars = _chars(public)
        self._visits[(dungeon_level, x, y)] = (
            self._visits.get((dungeon_level, x, y), 0) + 1
        )
        self._steps += 1

        if self._steps % 24 == 0:
            return 22

        candidates: list[tuple[int, int]] = []
        doors: list[tuple[int, int]] = []
        for action, dx, dy in _DIRECTIONS:
            nx = x + dx
            ny = y + dy
            if not 0 <= nx < 79 or not 0 <= ny < 21:
                continue
            tile = chars[ny * 79 + nx]
            visits = self._visits.get((dungeon_level, nx, ny), 0)
            if tile == ord("+"):
                doors.append((visits, action))
            elif tile not in {0, ord(" "), ord("|"), ord("-")}:
                candidates.append((visits, action))

        if candidates:
            least = min(visits for visits, _ in candidates)
            actions = [action for visits, action in candidates if visits == least]
            if self._previous_direction is not None and len(actions) > 1:
                reverse = _OPPOSITE[self._previous_direction]
                actions = [action for action in actions if action != reverse] or actions
            action = self._random.choice(actions)
            self._previous_direction = action
            return action

        if doors:
            least = min(visits for visits, _ in doors)
            directions = [action for visits, action in doors if visits == least]
            self._pending_direction = self._random.choice(directions)
            return 20

        self._previous_direction = None
        return 22


def _observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise ValueError("observation must be the NLE object")
    required = {"screen", "stats", "message", "inventory", "input_mode"}
    if set(value) != required:
        raise ValueError("observation schema is invalid")
    if (
        type(value["screen"]) is not dict
        or type(value["stats"]) is not dict
        or type(value["message"]) is not str
        or type(value["inventory"]) is not list
        or type(value["input_mode"]) is not str
    ):
        raise ValueError("observation values are invalid")
    return value


def _chars(observation: Mapping[str, PolicyValue]) -> bytes:
    screen = observation["screen"]
    if type(screen) is not dict:
        raise ValueError("screen is invalid")
    chars = screen.get("chars")
    if (
        type(chars) is not TensorValue
        or chars.dtype != "uint8"
        or chars.shape != (21, 79)
    ):
        raise ValueError("character screen is invalid")
    return chars.data


def _integer(stats: PolicyValue, name: str) -> int:
    if type(stats) is not dict:
        raise ValueError("stats are invalid")
    value = stats.get(name)
    if type(value) is not int:
        raise ValueError(f"stat {name} is invalid")
    return value


def _text(observation: Mapping[str, PolicyValue], name: str) -> str:
    value = observation[name]
    if type(value) is not str:
        raise ValueError(f"{name} is invalid")
    return value


def _food_letter(inventory: PolicyValue) -> str | None:
    if type(inventory) is not list:
        raise ValueError("inventory is invalid")
    for value in inventory:
        if type(value) is not dict:
            raise ValueError("inventory entry is invalid")
        letter = value.get("letter")
        description = value.get("description")
        if type(letter) is not str or type(description) is not str:
            raise ValueError("inventory entry is invalid")
        lowered = description.lower()
        if letter in _RAW_KEY_ACTION and any(word in lowered for word in _FOOD_WORDS):
            return letter
    return None


def make_policy(context: PolicyContext) -> BaselinePolicy:
    parameters = context.environment_parameters
    if parameters.get("task") != "NetHackScore-v0":
        raise ValueError("task is invalid")
    if parameters.get("action_profile") != "nle-task-actions-v1":
        raise ValueError("action profile is invalid")
    max_steps = parameters.get("max_episode_steps")
    if type(max_steps) is not int or not 1 <= max_steps <= 5_000:
        raise ValueError("max_episode_steps is invalid")
    return BaselinePolicy(context.policy_seed)
