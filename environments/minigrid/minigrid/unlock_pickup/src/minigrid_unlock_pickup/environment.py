"""One fresh MiniGrid UnlockPickup Environment per Episode."""

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

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
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
_BOX = 7
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
    target: tuple[int, int]
    carried: tuple[int, int] | None
    key_visible: bool
    target_visible: bool
    locked_door_visible: bool
    open_door_visible: bool
    visible_key_colors: tuple[int, ...]
    visible_door_colors: tuple[int, ...]
    front_object: tuple[int, int, int]
    front_label: str
    front_target: bool


class UnlockPickupEnvironment:
    """Strict seeded adapter around MiniGrid UnlockPickup."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("UnlockPickup has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make("MiniGrid-UnlockPickup-v0"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target: tuple[int, int] | None = None
        self._previous_facts: _Facts | None = None
        self._key_found = False
        self._key_first_seen_step = -1
        self._key_color_found = "unknown"
        self._key_picked_up = False
        self._key_picked_up_step = -1
        self._key_dropped = False
        self._door_opened = False
        self._door_opened_step = -1
        self._door_found = False
        self._door_first_seen_step = -1
        self._door_color_found = "unknown"
        self._target_found = False
        self._target_first_seen_step = -1
        self._target_destroyed = False
        self._target_destroyed_step = -1
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
            raise RuntimeError("MiniGrid UnlockPickup returned an unexpected horizon")
        self._target = facts.target
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
            raise RuntimeError("MiniGrid UnlockPickup observation history is unavailable")
        carried_before_action = previous_facts.carried
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        target_in_front_before_action = previous_facts.front_target
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid UnlockPickup returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.target != self._target:
            raise RuntimeError("MiniGrid UnlockPickup changed its mission")
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid UnlockPickup observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        pickup_attempt = action == 3
        pickup_succeeded = bool(
            pickup_attempt and carried_before_action is None and facts.carried is not None
        )
        key_picked_up = bool(pickup_succeeded and front_before_action[0] == _KEY)
        target_picked_up = bool(pickup_succeeded and front_before_action[:2] == facts.target)
        picked_up_label = _carried_label(facts.carried) if pickup_succeeded else "none"
        failed_pickup = pickup_attempt and not pickup_succeeded
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)

        drop_attempt = action == 4
        drop_succeeded = bool(
            drop_attempt and carried_before_action is not None and facts.carried is None
        )
        key_dropped = bool(
            drop_succeeded
            and carried_before_action is not None
            and carried_before_action[0] == _KEY
        )
        dropped_label = _carried_label(carried_before_action) if drop_succeeded else "none"
        failed_drop = drop_attempt and not drop_succeeded
        self._drop_attempts += int(drop_attempt)
        self._failed_drop_attempts += int(failed_drop)

        toggle_attempt = action == 5
        door_opened = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and front_before_action[2] == _LOCKED
            and carried_before_action == (_KEY, front_before_action[1])
        )
        target_destroyed = bool(toggle_attempt and target_in_front_before_action)
        toggle_effective = door_opened or target_destroyed
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)

        if key_picked_up and facts.carried != front_before_action[:2]:
            raise RuntimeError("MiniGrid UnlockPickup key pickup semantics drifted")
        if door_opened and not (facts.front_object[0] == _DOOR and facts.front_object[2] == _OPEN):
            raise RuntimeError("MiniGrid UnlockPickup door semantics drifted")
        if target_destroyed and facts.front_target:
            raise RuntimeError("MiniGrid UnlockPickup box toggle semantics drifted")
        if key_picked_up and not self._key_picked_up:
            self._key_picked_up_step = self._steps
        self._key_picked_up = self._key_picked_up or key_picked_up
        self._key_dropped = self._key_dropped or key_dropped
        if door_opened and not self._door_opened:
            self._door_opened_step = self._steps
        self._door_opened = self._door_opened or door_opened
        if target_destroyed and not self._target_destroyed:
            self._target_destroyed_step = self._steps
        self._target_destroyed = self._target_destroyed or target_destroyed
        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != target_picked_up or terminated != success:
            raise RuntimeError("MiniGrid UnlockPickup termination semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / _MAX_STEPS if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid UnlockPickup reward semantics drifted")
        if truncated != (self._steps == _MAX_STEPS):
            raise RuntimeError("MiniGrid UnlockPickup horizon semantics drifted")
        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated and not truncated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif self._target_destroyed and truncated:
            terminal_reason = "target_destroyed_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = self._task_stage(facts, terminal_reason=terminal_reason)
        self._done = terminated or truncated
        self._previous_facts = facts
        target_label = _task_object_label(facts.target)
        carried_label = _carried_label(facts.carried)
        carried_before_label = _carried_label(carried_before_action)
        matching_key_carried = bool(
            facts.carried is not None
            and facts.carried[0] == _KEY
            and self._door_color_found != "unknown"
            and _COLORS[facts.carried[1]] == self._door_color_found
        )
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(_MAX_STEPS - self._steps, 0),
            "target_color": _COLORS[facts.target[1]],
            "target_type": _OBJECTS[facts.target[0]],
            "target_label": target_label,
            "target_visible": facts.target_visible,
            "target_found": self._target_found,
            "target_first_seen_step": self._target_first_seen_step,
            "target_in_front": facts.front_target,
            "target_in_front_before_action": target_in_front_before_action,
            "target_destroyed_this_step": target_destroyed,
            "target_destroyed": self._target_destroyed,
            "target_destroyed_step": self._target_destroyed_step,
            "key_visible": facts.key_visible,
            "key_found": self._key_found,
            "key_first_seen_step": self._key_first_seen_step,
            "key_color_found": self._key_color_found,
            "key_picked_up_this_step": key_picked_up,
            "key_picked_up": self._key_picked_up,
            "key_picked_up_step": self._key_picked_up_step,
            "key_dropped_this_step": key_dropped,
            "key_dropped": self._key_dropped,
            "door_found": self._door_found,
            "door_first_seen_step": self._door_first_seen_step,
            "door_color_found": self._door_color_found,
            "locked_door_visible": facts.locked_door_visible,
            "open_door_visible": facts.open_door_visible,
            "door_opened_this_step": door_opened,
            "door_opened": self._door_opened,
            "door_opened_step": self._door_opened_step,
            "matching_key_carried": matching_key_carried,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "carried_object": carried_label,
            "carried_object_before_action": carried_before_label,
            "pickup_attempt": pickup_attempt,
            "pickup_succeeded": pickup_succeeded,
            "picked_up_label": picked_up_label,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "drop_attempt": drop_attempt,
            "drop_succeeded": drop_succeeded,
            "dropped_label": dropped_label,
            "drop_attempt_count": self._drop_attempts,
            "failed_drop": failed_drop,
            "failed_drop_count": self._failed_drop_attempts,
            "toggle_attempt": toggle_attempt,
            "toggle_effective": toggle_effective,
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
            raise RuntimeError("MiniGrid UnlockPickup returned multiple key colors")
        if len(facts.visible_door_colors) > 1:
            raise RuntimeError("MiniGrid UnlockPickup returned multiple door colors")
        if facts.key_visible and not self._key_found:
            self._key_first_seen_step = step
        if facts.target_visible and not self._target_found:
            self._target_first_seen_step = step
        if facts.visible_door_colors and not self._door_found:
            self._door_first_seen_step = step
        if facts.visible_key_colors:
            color = _COLORS[facts.visible_key_colors[0]]
            if self._key_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid UnlockPickup changed its key color")
            self._key_color_found = color
        if facts.visible_door_colors:
            color = _COLORS[facts.visible_door_colors[0]]
            if self._door_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid UnlockPickup changed its door color")
            self._door_color_found = color
        self._key_found = self._key_found or facts.key_visible
        self._door_found = self._door_found or bool(facts.visible_door_colors)
        self._target_found = self._target_found or facts.target_visible

    def _task_stage(self, facts: _Facts, *, terminal_reason: str) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if self._target_destroyed:
            return "target_destroyed"
        if facts.front_target and facts.carried is None:
            return "pick_up_target"
        if self._door_opened:
            if facts.carried is not None:
                return "drop_key"
            if self._target_found:
                return "approach_target"
            return "find_target"
        if facts.carried is not None and facts.carried[0] == _KEY:
            return "unlock_door"
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
        raise RuntimeError("MiniGrid UnlockPickup returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid UnlockPickup returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid UnlockPickup returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid UnlockPickup returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid UnlockPickup returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid UnlockPickup returned invalid mission")
    target = _target(mission)
    carried_code = int(image[3, 6, 0])
    carried = (carried_code, int(image[3, 6, 1])) if carried_code in {_KEY, _BOX} else None
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
            target=target,
            carried=carried,
            key_visible=bool(numpy.any(image[:, :, 0] == _KEY)),
            target_visible=bool(
                numpy.any((image[:, :, 0] == target[0]) & (image[:, :, 1] == target[1]))
            ),
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
            front_target=front_object[:2] == target,
        ),
    )


def _target(mission: str) -> tuple[int, int]:
    prefix = "pick up the "
    if not mission.startswith(prefix):
        raise RuntimeError("MiniGrid UnlockPickup returned invalid mission")
    parts = mission.removeprefix(prefix).split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] != "box":
        raise RuntimeError("MiniGrid UnlockPickup returned invalid mission")
    return _BOX, _COLORS.index(parts[0])


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid UnlockPickup returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid UnlockPickup returned non-finite reward")
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


def _task_object_label(value: tuple[int, int]) -> str:
    object_code, color_code = value
    if object_code not in {_KEY, _BOX} or not 0 <= color_code < len(_COLORS):
        return "none"
    return f"{_COLORS[color_code]}_{_OBJECTS[object_code]}"


def _carried_label(value: tuple[int, int] | None) -> str:
    return "none" if value is None else _task_object_label(value)


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid UnlockPickup public observation is invalid")
    return image.data, direction


__all__ = ["UnlockPickupEnvironment"]
