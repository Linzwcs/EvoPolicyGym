"""One fresh MiniGrid KeyCorridor Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401  # Import registers MiniGrid environments.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import KeyCorridorConfig

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
_BALL = 6
_OPEN = 0
_CLOSED = 1
_LOCKED = 2
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
class _ObservationFacts:
    target_color: int
    carried: tuple[int, int] | None
    key_visible: bool
    target_object_visible: bool
    locked_door_visible: bool
    visible_key_colors: tuple[int, ...]
    visible_locked_door_colors: tuple[int, ...]
    visible_door_count: int
    front_object: tuple[int, int, int]
    front_label: str
    front_target: bool


class KeyCorridorEnvironment:
    """The seeded strict adapter around configured MiniGrid KeyCorridor."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: KeyCorridorConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not KeyCorridorConfig:
            raise TypeError("config must be KeyCorridorConfig")
        if episode.scenario is not None:
            raise ValueError(
                "KeyCorridor configuration belongs in KeyCorridorConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._max_steps = config.max_episode_steps
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target_color: int | None = None
        self._previous_facts: _ObservationFacts | None = None
        self._found_key = False
        self._key_first_seen_step = -1
        self._key_color_found = "unknown"
        self._picked_up_key = False
        self._key_picked_up_step = -1
        self._key_dropped = False
        self._opened_target_door = False
        self._target_door_first_seen_step = -1
        self._target_door_color_found = "unknown"
        self._target_door_opened_step = -1
        self._opened_exploration_door = False
        self._exploration_door_toggle_count = 0
        self._found_target_object = False
        self._target_first_seen_step = -1
        self._steps = 0
        self._pickup_attempts = 0
        self._failed_pickup_attempts = 0
        self._target_pickup_blocked_attempts = 0
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
        if type(horizon) is not int or horizon != self._max_steps:
            raise RuntimeError("MiniGrid KeyCorridor returned an unexpected horizon")
        self._target_color = facts.target_color
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
            raise RuntimeError("MiniGrid KeyCorridor observation history is unavailable")
        carried_before_action = previous_facts.carried
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        target_in_front_before_action = previous_facts.front_target
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid KeyCorridor returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if facts.target_color != self._target_color:
            raise RuntimeError("MiniGrid KeyCorridor changed target color during an Episode")
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid KeyCorridor observation history is unavailable")
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
        target_picked_up = bool(
            pickup_succeeded and front_before_action[:2] == (_BALL, facts.target_color)
        )
        target_pickup_blocked = bool(
            pickup_attempt and target_in_front_before_action and carried_before_action is not None
        )
        failed_pickup = pickup_attempt and not pickup_succeeded
        picked_up_label = _carried_label(facts.carried) if pickup_succeeded else "none"
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)
        self._target_pickup_blocked_attempts += int(target_pickup_blocked)

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
        target_door_opened = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and front_before_action[2] == _LOCKED
            and carried_before_action == (_KEY, front_before_action[1])
        )
        door_state_changed = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and facts.front_object[0] == _DOOR
            and facts.front_object[2] != front_before_action[2]
        )
        exploration_door_toggled = door_state_changed and not target_door_opened
        exploration_door_opened = bool(
            exploration_door_toggled
            and front_before_action[2] == _CLOSED
            and facts.front_object[2] == _OPEN
        )
        toggle_effective = target_door_opened or exploration_door_toggled
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        self._exploration_door_toggle_count += int(exploration_door_toggled)

        if key_picked_up and facts.carried != front_before_action[:2]:
            raise RuntimeError("MiniGrid KeyCorridor key pickup semantics drifted")
        if target_door_opened and not (
            facts.front_object[0] == _DOOR and facts.front_object[2] == _OPEN
        ):
            raise RuntimeError("MiniGrid KeyCorridor target door semantics drifted")
        if key_picked_up and not self._picked_up_key:
            self._key_picked_up_step = self._steps
        self._picked_up_key = self._picked_up_key or key_picked_up
        self._key_dropped = self._key_dropped or key_dropped
        if target_door_opened and not self._opened_target_door:
            self._target_door_opened_step = self._steps
        self._opened_target_door = self._opened_target_door or target_door_opened
        self._opened_exploration_door = self._opened_exploration_door or exploration_door_opened
        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != target_picked_up or terminated != success:
            raise RuntimeError("MiniGrid KeyCorridor termination semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / self._max_steps if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid KeyCorridor reward semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid KeyCorridor horizon semantics drifted")
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
        target_label = f"{_COLORS[facts.target_color]}_ball"
        carried_label = _carried_label(facts.carried)
        carried_before_label = _carried_label(carried_before_action)
        carrying_key = bool(facts.carried is not None and facts.carried[0] == _KEY)
        matching_key_carried = bool(
            carrying_key
            and self._target_door_color_found != "unknown"
            and facts.carried is not None
            and _COLORS[facts.carried[1]] == self._target_door_color_found
        )
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "target_color": _COLORS[facts.target_color],
            "target_type": "ball",
            "target_label": target_label,
            "target_visible": facts.target_object_visible,
            "found_target_object": self._found_target_object,
            "target_first_seen_step": self._target_first_seen_step,
            "target_in_front": facts.front_target,
            "target_in_front_before_action": target_in_front_before_action,
            "key_visible": facts.key_visible,
            "found_key": self._found_key,
            "key_first_seen_step": self._key_first_seen_step,
            "key_color_found": self._key_color_found,
            "key_picked_up_this_step": key_picked_up,
            "picked_up_key": self._picked_up_key,
            "key_picked_up_step": self._key_picked_up_step,
            "key_dropped_this_step": key_dropped,
            "key_dropped": self._key_dropped,
            "carrying_key": carrying_key,
            "matching_key_carried": matching_key_carried,
            "target_door_visible": facts.locked_door_visible,
            "target_door_found": self._target_door_color_found != "unknown",
            "target_door_first_seen_step": self._target_door_first_seen_step,
            "target_door_color_found": self._target_door_color_found,
            "target_door_opened_this_step": target_door_opened,
            "opened_target_door": self._opened_target_door,
            "target_door_opened_step": self._target_door_opened_step,
            "visible_door_count": facts.visible_door_count,
            "exploration_door_toggled_this_step": exploration_door_toggled,
            "exploration_door_opened": self._opened_exploration_door,
            "exploration_door_toggle_count": self._exploration_door_toggle_count,
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
            "target_pickup_blocked_by_carried_object": target_pickup_blocked,
            "target_pickup_blocked_count": self._target_pickup_blocked_attempts,
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
            "success_reward_at_this_step": (1.0 - 0.9 * self._steps / self._max_steps),
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

    def _update_discovery(
        self,
        facts: _ObservationFacts,
        *,
        step: int,
    ) -> None:
        if len(facts.visible_key_colors) > 1:
            raise RuntimeError("MiniGrid KeyCorridor returned multiple key colors")
        if len(facts.visible_locked_door_colors) > 1:
            raise RuntimeError("MiniGrid KeyCorridor returned multiple locked door colors")
        if facts.key_visible and not self._found_key:
            self._key_first_seen_step = step
        if facts.target_object_visible and not self._found_target_object:
            self._target_first_seen_step = step
        if facts.visible_locked_door_colors and self._target_door_color_found == "unknown":
            self._target_door_first_seen_step = step
        if facts.visible_key_colors:
            color = _COLORS[facts.visible_key_colors[0]]
            if self._key_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid KeyCorridor changed its key color")
            self._key_color_found = color
        if facts.visible_locked_door_colors:
            color = _COLORS[facts.visible_locked_door_colors[0]]
            if self._target_door_color_found not in {"unknown", color}:
                raise RuntimeError("MiniGrid KeyCorridor changed its target door color")
            self._target_door_color_found = color
        self._found_key = self._found_key or facts.key_visible
        self._found_target_object = self._found_target_object or facts.target_object_visible

    def _task_stage(
        self,
        facts: _ObservationFacts,
        *,
        terminal_reason: str,
    ) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.front_target:
            if facts.carried is not None:
                return "free_hands_for_target"
            return "pick_up_target"
        if self._opened_target_door:
            if facts.carried is not None:
                return "drop_key"
            if self._found_target_object:
                return "approach_target"
            return "find_target"
        if facts.carried is not None and facts.carried[0] == _KEY:
            if self._target_door_color_found != "unknown":
                return "unlock_target_door"
            return "find_target_door"
        if self._picked_up_key:
            return "recover_key"
        if self._found_key:
            return "acquire_key"
        return "explore_rooms"


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid KeyCorridor returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    target_color = _target_color(mission)

    carried_code = int(image[3, 6, 0])
    carried = (carried_code, int(image[3, 6, 1])) if carried_code in {_KEY, _BALL} else None
    key_visible = bool(numpy.any(image[:, :, 0] == _KEY))
    target_object_visible = bool(
        numpy.any((image[:, :, 0] == _BALL) & (image[:, :, 1] == target_color))
    )
    visible_key_colors = tuple(
        sorted(
            {
                int(color_code)
                for object_code, color_code in image[:, :, :2].reshape(-1, 2)
                if int(object_code) == _KEY
            }
        )
    )
    visible_locked_door_colors = tuple(
        sorted(
            {
                int(color_code)
                for object_code, color_code, state_code in image.reshape(-1, 3)
                if int(object_code) == _DOOR and int(state_code) == _LOCKED
            }
        )
    )
    visible_door_count = int(numpy.count_nonzero(image[:, :, 0] == _DOOR))
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
        _ObservationFacts(
            target_color=target_color,
            carried=carried,
            key_visible=key_visible,
            target_object_visible=target_object_visible,
            locked_door_visible=bool(
                numpy.any((image[:, :, 0] == _DOOR) & (image[:, :, 2] == _LOCKED))
            ),
            visible_key_colors=visible_key_colors,
            visible_locked_door_colors=visible_locked_door_colors,
            visible_door_count=visible_door_count,
            front_object=front_object,
            front_label=_object_label(*front_object),
            front_target=front_object[:2] == (_BALL, target_color),
        ),
    )


def _target_color(mission: str) -> int:
    prefix = "pick up the "
    suffix = " ball"
    if not mission.startswith(prefix) or not mission.endswith(suffix):
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    color = mission[len(prefix) : -len(suffix)]
    if color not in _COLORS:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    return _COLORS.index(color)


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid KeyCorridor returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid KeyCorridor returned a non-finite {name}")
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


def _carried_label(value: tuple[int, int] | None) -> str:
    if value is None:
        return "none"
    object_code, color_code = value
    if object_code not in {_KEY, _BALL} or not 0 <= color_code < len(_COLORS):
        return "none"
    return f"{_COLORS[color_code]}_{_OBJECTS[object_code]}"


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid KeyCorridor public observation is invalid")
    return image.data, direction


__all__ = ["KeyCorridorEnvironment"]
