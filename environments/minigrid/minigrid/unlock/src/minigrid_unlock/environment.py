"""One fresh MiniGrid Unlock Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

_ENVIRONMENT_ID = "MiniGrid-Unlock-v0"
_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "open the door"
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECTS = (
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
)
_STATES = ("open", "closed", "locked")
_DOOR = 4
_KEY = 5
_OPEN = 0
_LOCKED = 2
_MAX_STEPS = 8 * 6**2
_ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "move_forward",
    "pick_up",
    "drop",
    "toggle",
    "done",
)


@dataclass(frozen=True, slots=True)
class _Facts:
    key_visible: bool
    carried_key_color: int | None
    locked_door_visible: bool
    open_door_visible: bool
    visible_key_colors: tuple[int, ...]
    visible_door_colors: tuple[int, ...]
    front_object: tuple[int, int, int]
    front_label: str


class UnlockEnvironment:
    """Strict seeded adapter around MiniGrid Unlock."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Unlock has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(_ENVIRONMENT_ID),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._previous_facts: _Facts | None = None
        self._key_found = False
        self._key_first_seen_step = -1
        self._key_color_found = "unknown"
        self._key_picked_up = False
        self._key_picked_up_step = -1
        self._key_dropped = False
        self._door_found = False
        self._door_first_seen_step = -1
        self._door_color_found = "unknown"
        self._door_opened = False
        self._door_opened_step = -1
        self._steps = 0
        self._pickup_attempts = 0
        self._failed_pickup_attempts = 0
        self._drop_attempts = 0
        self._failed_drop_attempts = 0
        self._toggle_attempts = 0
        self._failed_toggle_attempts = 0
        self._previous_observation_signature: tuple[bytes, int] | None = None
        self._seen_observation_signatures: set[tuple[bytes, int]] = set()
        self._novel_observation_steps = 0
        self._ineffective_actions = 0
        self._action_counts = [0] * len(_ACTION_NAMES)
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        horizon = self._environment.get_wrapper_attr("max_steps")
        if type(horizon) is not int or horizon != _MAX_STEPS:
            raise RuntimeError("MiniGrid Unlock returned an unexpected horizon")
        self._update_discovery(facts, step=0)
        self._previous_facts = facts
        signature = _observation_signature(public)
        self._previous_observation_signature = signature
        self._seen_observation_signatures.add(signature)
        self._started = True
        return public

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()

        previous_facts = self._previous_facts
        if previous_facts is None:
            raise RuntimeError("MiniGrid Unlock observation history is unavailable")
        carried_before_action = previous_facts.carried_key_color
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid Unlock returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid Unlock observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        pickup_attempt = action == 3
        key_picked_up = bool(
            pickup_attempt and carried_before_action is None and facts.carried_key_color is not None
        )
        failed_pickup = pickup_attempt and not key_picked_up
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)
        drop_attempt = action == 4
        key_dropped = bool(
            drop_attempt and carried_before_action is not None and facts.carried_key_color is None
        )
        failed_drop = drop_attempt and not key_dropped
        self._drop_attempts += int(drop_attempt)
        self._failed_drop_attempts += int(failed_drop)
        toggle_attempt = action == 5
        door_opened = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and front_before_action[2] == _LOCKED
            and carried_before_action == front_before_action[1]
        )
        failed_toggle = toggle_attempt and not door_opened
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        if key_picked_up and facts.carried_key_color != front_before_action[1]:
            raise RuntimeError("MiniGrid Unlock key pickup semantics drifted")
        if door_opened and not (facts.front_object[0] == _DOOR and facts.front_object[2] == _OPEN):
            raise RuntimeError("MiniGrid Unlock door semantics drifted")
        if key_picked_up and not self._key_picked_up:
            self._key_picked_up_step = self._steps
        self._key_picked_up = self._key_picked_up or key_picked_up
        self._key_dropped = self._key_dropped or key_dropped
        if door_opened and not self._door_opened:
            self._door_opened_step = self._steps
        self._door_opened = self._door_opened or door_opened
        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != door_opened or terminated != success:
            raise RuntimeError("MiniGrid Unlock termination semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / _MAX_STEPS if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid Unlock reward semantics drifted")
        if truncated != (self._steps == _MAX_STEPS):
            raise RuntimeError("MiniGrid Unlock horizon semantics drifted")
        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated and not truncated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = self._task_stage(facts, terminal_reason=terminal_reason)
        self._done = terminated or truncated
        self._previous_facts = facts
        carried_key_color = (
            _COLORS[facts.carried_key_color] if facts.carried_key_color is not None else "none"
        )
        carried_before = (
            _COLORS[carried_before_action] if carried_before_action is not None else "none"
        )
        matching_key_carried = bool(
            facts.carried_key_color is not None
            and self._door_color_found != "unknown"
            and _COLORS[facts.carried_key_color] == self._door_color_found
        )
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(_MAX_STEPS - self._steps, 0),
            "key_visible": facts.key_visible,
            "key_found": self._key_found,
            "key_first_seen_step": self._key_first_seen_step,
            "key_color_found": self._key_color_found,
            "key_picked_up_this_step": key_picked_up,
            "key_picked_up": self._key_picked_up,
            "key_picked_up_step": self._key_picked_up_step,
            "key_dropped_this_step": key_dropped,
            "key_dropped": self._key_dropped,
            "carried_key_color": carried_key_color,
            "carried_key_color_before_action": carried_before,
            "matching_key_carried": matching_key_carried,
            "door_found": self._door_found,
            "door_first_seen_step": self._door_first_seen_step,
            "door_color_found": self._door_color_found,
            "locked_door_visible": facts.locked_door_visible,
            "open_door_visible": facts.open_door_visible,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "door_opened_this_step": door_opened,
            "door_opened": self._door_opened,
            "door_opened_step": self._door_opened_step,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "drop_attempt": drop_attempt,
            "drop_attempt_count": self._drop_attempts,
            "failed_drop": failed_drop,
            "failed_drop_count": self._failed_drop_attempts,
            "toggle_attempt": toggle_attempt,
            "toggle_attempt_count": self._toggle_attempts,
            "failed_toggle": failed_toggle,
            "failed_toggle_count": self._failed_toggle_attempts,
            "task_stage": task_stage,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": (1.0 - 0.9 * self._steps / _MAX_STEPS),
            "cumulative_return": self._cumulative_return,
            "success": success,
            "terminal_reason": terminal_reason,
        }
        for name, count in zip(_ACTION_NAMES, self._action_counts, strict=True):
            metrics[f"{name}_count"] = count
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _update_discovery(self, facts: _Facts, *, step: int) -> None:
        if len(facts.visible_key_colors) > 1:
            raise RuntimeError("MiniGrid Unlock returned multiple key colors")
        if len(facts.visible_door_colors) > 1:
            raise RuntimeError("MiniGrid Unlock returned multiple door colors")
        if facts.key_visible and not self._key_found:
            self._key_first_seen_step = step
        if facts.visible_door_colors and not self._door_found:
            self._door_first_seen_step = step
        if facts.visible_key_colors:
            color = _COLORS[facts.visible_key_colors[0]]
            if self._key_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid Unlock changed its key color")
            self._key_color_found = color
        if facts.visible_door_colors:
            color = _COLORS[facts.visible_door_colors[0]]
            if self._door_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid Unlock changed its door color")
            self._door_color_found = color
        self._key_found = self._key_found or facts.key_visible
        self._door_found = self._door_found or bool(facts.visible_door_colors)

    def _task_stage(self, facts: _Facts, *, terminal_reason: str) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.carried_key_color is not None:
            if (
                facts.front_object[0] == _DOOR
                and facts.front_object[2] == _LOCKED
                and facts.front_object[1] == facts.carried_key_color
            ):
                return "open_door"
            return "find_door"
        if self._key_picked_up:
            return "recover_key"
        if self._key_found:
            return "acquire_key"
        return "find_key"


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Unlock returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Unlock returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid Unlock returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Unlock returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Unlock returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid Unlock returned invalid mission")
    carried_key_color = int(image[3, 6, 1]) if image[3, 6, 0] == _KEY else None
    visible_key_colors = tuple(
        sorted(
            {
                int(color_code)
                for object_code, color_code in image[:, :, :2].reshape(-1, 2)
                if int(object_code) == _KEY
            }
        )
    )
    visible_door_colors = tuple(
        sorted(
            {
                int(color_code)
                for object_code, color_code in image[:, :, :2].reshape(-1, 2)
                if int(object_code) == _DOOR
            }
        )
    )
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    return (
        {
            "image": TensorValue(
                dtype="uint8",
                shape=_IMAGE_SHAPE,
                data=image.tobytes(order="C"),
            ),
            "direction": direction,
            "mission": mission,
        },
        _Facts(
            key_visible=bool(numpy.any(image[:, :, 0] == _KEY)),
            carried_key_color=carried_key_color,
            locked_door_visible=bool(
                numpy.any((image[:, :, 0] == _DOOR) & (image[:, :, 2] == _LOCKED))
            ),
            open_door_visible=bool(
                numpy.any((image[:, :, 0] == _DOOR) & (image[:, :, 2] == _OPEN))
            ),
            visible_key_colors=visible_key_colors,
            visible_door_colors=visible_door_colors,
            front_object=front_object,
            front_label=_object_label(*front_object),
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid Unlock returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid Unlock returned non-finite reward")
    return number


def _object_label(object_code: int, color_code: int, state_code: int) -> str:
    if not 0 <= object_code < len(_OBJECTS):
        return "unknown"
    if object_code == 0:
        return "unseen"
    if object_code == 1:
        return "empty"
    if not 0 <= color_code < len(_COLORS):
        return "unknown"
    if object_code == _DOOR:
        if not 0 <= state_code < len(_STATES):
            return "unknown"
        return f"{_COLORS[color_code]}_{_STATES[state_code]}_door"
    return f"{_COLORS[color_code]}_{_OBJECTS[object_code]}"


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid Unlock public observation is invalid")
    return image.data, direction


__all__ = ["UnlockEnvironment"]
