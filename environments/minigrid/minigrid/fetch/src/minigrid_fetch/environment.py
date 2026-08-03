"""One fresh MiniGrid Fetch Environment per Episode."""

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

from .config import FetchConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECT_CODES = {"key": 5, "ball": 6}
_OBJECT_TYPES_BY_CODE = {code: name for name, code in _OBJECT_CODES.items()}
_ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "move_forward",
    "pick_up",
    "drop",
    "toggle",
    "done",
)
_MISSION_PREFIXES = (
    "get a ",
    "go get a ",
    "fetch a ",
    "go fetch a ",
    "you must fetch a ",
)


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    target_color: int
    target_type: str
    target_visible: bool
    target_in_front: bool
    visible_candidates: tuple[str, ...]
    front_candidate: str
    carried_candidate: str


class FetchEnvironment:
    """The seeded strict adapter around configured MiniGrid Fetch."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: FetchConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not FetchConfig:
            raise TypeError("config must be FetchConfig")
        if episode.scenario is not None:
            raise ValueError("Fetch configuration belongs in FetchConfig, not EpisodeSpec.scenario")

        self._seed = episode.environment_seed
        self._max_steps = config.max_episode_steps
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target: tuple[int, str] | None = None
        self._target_found = False
        self._target_first_seen_step = -1
        self._target_first_in_front_step = -1
        self._candidate_signatures_found: set[str] = set()
        self._max_visible_candidate_count = 0
        self._previous_facts: _ObservationFacts | None = None
        self._steps = 0
        self._pickup_step = -1
        self._pickup_attempts = 0
        self._failed_pickup_attempts = 0
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
            raise RuntimeError("MiniGrid Fetch returned an unexpected horizon")
        self._target = (facts.target_color, facts.target_type)
        self._target_found = facts.target_visible
        self._target_first_seen_step = 0 if facts.target_visible else -1
        self._target_first_in_front_step = 0 if facts.target_in_front else -1
        self._candidate_signatures_found.update(facts.visible_candidates)
        self._max_visible_candidate_count = len(facts.visible_candidates)
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
            raise RuntimeError("MiniGrid Fetch observation history is unavailable")
        front_candidate_before_action = previous_facts.front_candidate
        target_in_front_before_action = previous_facts.target_in_front
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid Fetch returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if (facts.target_color, facts.target_type) != self._target:
            raise RuntimeError("MiniGrid Fetch changed target during an Episode")
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid Fetch observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1
        if facts.target_visible and self._target_first_seen_step < 0:
            self._target_first_seen_step = self._steps
        if facts.target_in_front and self._target_first_in_front_step < 0:
            self._target_first_in_front_step = self._steps
        self._target_found = self._target_found or facts.target_visible
        self._candidate_signatures_found.update(facts.visible_candidates)
        self._max_visible_candidate_count = max(
            self._max_visible_candidate_count,
            len(facts.visible_candidates),
        )
        success = bool(terminated and number > 0.0)
        wrong_object = bool(
            terminated and number == 0.0 and action == 3 and facts.carried_candidate != "none"
        )
        picked_up_object = success or wrong_object
        if terminated != picked_up_object:
            raise RuntimeError("MiniGrid Fetch termination semantics drifted")
        target_label = f"{_COLORS[facts.target_color]}_{facts.target_type}"
        if success and facts.carried_candidate != target_label:
            raise RuntimeError("MiniGrid Fetch success object is inconsistent")
        if wrong_object and facts.carried_candidate == target_label:
            raise RuntimeError("MiniGrid Fetch wrong-object outcome is inconsistent")
        expected_reward = 1.0 - 0.9 * self._steps / self._max_steps if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid Fetch reward semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid Fetch horizon semantics drifted")
        pickup_attempt = action == 3
        failed_pickup = pickup_attempt and not picked_up_object
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)
        if picked_up_object:
            self._pickup_step = self._steps
        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated and not truncated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif wrong_object and truncated:
            terminal_reason = "wrong_object_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif wrong_object:
            terminal_reason = "wrong_object"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = "explore_candidates"
        if terminal_reason != "none":
            task_stage = terminal_reason
        elif facts.target_in_front:
            task_stage = "pick_up_target"
        elif self._target_found:
            task_stage = "approach_target"
        self._done = terminated or truncated
        self._previous_facts = facts
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "target_color": _COLORS[facts.target_color],
            "target_type": facts.target_type,
            "target_label": target_label,
            "target_visible": facts.target_visible,
            "target_found": self._target_found,
            "target_first_seen_step": self._target_first_seen_step,
            "target_in_front": facts.target_in_front,
            "target_in_front_before_action": target_in_front_before_action,
            "target_first_in_front_step": self._target_first_in_front_step,
            "visible_candidate_count": len(facts.visible_candidates),
            "max_visible_candidate_count": self._max_visible_candidate_count,
            "unique_candidate_count_found": len(self._candidate_signatures_found),
            "front_candidate": facts.front_candidate,
            "front_candidate_before_action": front_candidate_before_action,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "picked_up_object": picked_up_object,
            "picked_up_label": (facts.carried_candidate if picked_up_object else "none"),
            "pickup_step": self._pickup_step,
            "wrong_object": wrong_object,
            "wrong_object_label": (facts.carried_candidate if wrong_object else "none"),
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


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Fetch returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Fetch returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid Fetch returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Fetch returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Fetch returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    target_color, target_type = _target(mission)
    target_code = _OBJECT_CODES[target_type]
    target_visible = bool(
        numpy.any((image[:, :, 0] == target_code) & (image[:, :, 1] == target_color))
    )
    visible_candidates = tuple(
        sorted(
            {
                label
                for object_code, color_code in image[:, :, :2].reshape(-1, 2)
                if (
                    label := _candidate_label(
                        int(object_code),
                        int(color_code),
                    )
                )
                != "none"
            }
        )
    )
    front_candidate = _candidate_label(
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
    )
    carried_candidate = _candidate_label(
        int(image[3, 6, 0]),
        int(image[3, 6, 1]),
    )
    target_label = f"{_COLORS[target_color]}_{target_type}"
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
            target_type=target_type,
            target_visible=target_visible,
            target_in_front=front_candidate == target_label,
            visible_candidates=visible_candidates,
            front_candidate=front_candidate,
            carried_candidate=carried_candidate,
        ),
    )


def _target(mission: str) -> tuple[int, str]:
    remainder = next(
        (
            mission.removeprefix(prefix)
            for prefix in _MISSION_PREFIXES
            if mission.startswith(prefix)
        ),
        None,
    )
    if remainder is None:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    parts = remainder.split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in _OBJECT_CODES:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    return _COLORS.index(parts[0]), parts[1]


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid Fetch returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid Fetch returned a non-finite {name}")
    return number


def _candidate_label(object_code: int, color_code: int) -> str:
    object_type = _OBJECT_TYPES_BY_CODE.get(object_code)
    if object_type is None or not 0 <= color_code < len(_COLORS):
        return "none"
    return f"{_COLORS[color_code]}_{object_type}"


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid Fetch public observation is invalid")
    return image.data, direction


__all__ = ["FetchEnvironment"]
