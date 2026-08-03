"""One fresh Gymnasium Pusher-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import PusherConfig

_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "upper_arm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
)
_OBSERVATION_NAMES = (
    *(f"{name}_angle" for name in _JOINT_NAMES),
    *(f"{name}_angular_velocity" for name in _JOINT_NAMES),
    "fingertip_x",
    "fingertip_y",
    "fingertip_z",
    "object_x",
    "object_y",
    "object_z",
    "goal_x",
    "goal_y",
    "goal_z",
)
_MODEL_TIMESTEP_SECONDS = 0.01
_MAX_EPISODE_STEPS = 100


class PusherEnvironment:
    """The seeded strict adapter around configured Pusher-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: PusherConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not PusherConfig:
            raise TypeError("config must be PusherConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Pusher configuration belongs in PusherConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Pusher-v5",
                frame_skip=config.frame_skip,
                reward_near_weight=config.reward_near_weight,
                reward_dist_weight=config.reward_dist_weight,
                reward_control_weight=config.reward_control_weight,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._initial_object_position: tuple[float, float, float] | None = None
        self._initial_object_goal_distance: float | None = None
        self._minimum_object_goal_distance = math.inf
        self._maximum_object_goal_distance = -math.inf
        self._minimum_fingertip_object_distance = math.inf
        self._maximum_object_displacement = 0.0
        self._cumulative_action_squared_norm = 0.0
        self._cumulative_reward_distance = 0.0
        self._cumulative_reward_near = 0.0
        self._cumulative_reward_control = 0.0
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        initial_object = _point(public_observation, "object")
        initial_goal = _point(public_observation, "goal")
        initial_distance = _distance(initial_object, initial_goal)
        self._initial_object_position = initial_object
        self._initial_object_goal_distance = initial_distance
        self._minimum_object_goal_distance = initial_distance
        self._maximum_object_goal_distance = initial_distance
        self._minimum_fingertip_object_distance = _distance(
            _point(public_observation, "fingertip"),
            initial_object,
        )
        self._started = True
        return public_observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(applied_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Pusher returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        reward_terms = _reward_metrics(info)
        initial_object = self._initial_object_position
        initial_distance = self._initial_object_goal_distance
        if initial_object is None or initial_distance is None:
            raise RuntimeError("Pusher initial diagnostics are unavailable")
        fingertip = _point(public_observation, "fingertip")
        object_position = _point(public_observation, "object")
        goal = _point(public_observation, "goal")
        object_goal_distance = _distance(object_position, goal)
        fingertip_object_distance = _distance(fingertip, object_position)
        object_displacement = _distance(object_position, initial_object)
        self._steps += 1
        self._minimum_object_goal_distance = min(
            self._minimum_object_goal_distance,
            object_goal_distance,
        )
        self._maximum_object_goal_distance = max(
            self._maximum_object_goal_distance,
            object_goal_distance,
        )
        self._minimum_fingertip_object_distance = min(
            self._minimum_fingertip_object_distance,
            fingertip_object_distance,
        )
        self._maximum_object_displacement = max(
            self._maximum_object_displacement,
            object_displacement,
        )
        action_squared_norm = float(numpy.square(applied_action).sum())
        self._cumulative_action_squared_norm += action_squared_norm
        self._cumulative_reward_distance += reward_terms["reward_distance"]
        self._cumulative_reward_near += reward_terms["reward_near"]
        self._cumulative_reward_control += reward_terms["reward_control"]
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            action_squared_norm=action_squared_norm,
            reward=public_reward,
            reward_terms=reward_terms,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            initial_object=initial_object,
            initial_object_goal_distance=initial_distance,
            minimum_object_goal_distance=self._minimum_object_goal_distance,
            maximum_object_goal_distance=self._maximum_object_goal_distance,
            minimum_fingertip_object_distance=(self._minimum_fingertip_object_distance),
            maximum_object_displacement=self._maximum_object_displacement,
            cumulative_action_squared_norm=(self._cumulative_action_squared_norm),
            cumulative_reward_distance=self._cumulative_reward_distance,
            cumulative_reward_near=self._cumulative_reward_near,
            cumulative_reward_control=self._cumulative_reward_control,
            cumulative_return=self._cumulative_return,
        )
        self._done = terminated or truncated
        return Step(
            observation=public_observation,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 7:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -2.0 <= item <= 2.0:
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (23,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Pusher returned an invalid observation")
    return {
        name: _number(item, name=name) for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Pusher returned invalid reward metrics")
    required = {"reward_dist", "reward_ctrl", "reward_near"}
    if not required.issubset(value):
        raise RuntimeError("Pusher omitted reward metrics")
    return {
        "reward_distance": _number(
            value["reward_dist"],
            name="reward distance",
        ),
        "reward_control": _number(
            value["reward_ctrl"],
            name="reward control",
        ),
        "reward_near": _number(
            value["reward_near"],
            name="reward near",
        ),
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    *,
    action_squared_norm: float,
    reward: float,
    reward_terms: dict[str, float],
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: PusherConfig,
    initial_object: tuple[float, float, float],
    initial_object_goal_distance: float,
    minimum_object_goal_distance: float,
    maximum_object_goal_distance: float,
    minimum_fingertip_object_distance: float,
    maximum_object_displacement: float,
    cumulative_action_squared_norm: float,
    cumulative_reward_distance: float,
    cumulative_reward_near: float,
    cumulative_reward_control: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    fingertip = _point(observation, "fingertip")
    object_position = _point(observation, "object")
    goal = _point(observation, "goal")
    fingertip_object_distance = _distance(fingertip, object_position)
    object_goal_distance = _distance(object_position, goal)
    object_displacement = _distance(object_position, initial_object)
    expected_distance_reward = -config.reward_dist_weight * object_goal_distance
    expected_near_reward = -config.reward_near_weight * fingertip_object_distance
    expected_control_reward = -config.reward_control_weight * action_squared_norm
    expected_terms = {
        "reward_distance": expected_distance_reward,
        "reward_near": expected_near_reward,
        "reward_control": expected_control_reward,
    }
    for name, expected in expected_terms.items():
        if not math.isclose(
            reward_terms[name],
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Pusher {name} semantics drifted")
    reconstructed_reward = sum(reward_terms.values())
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Pusher reward decomposition drifted")
    if terminated:
        raise RuntimeError("Pusher natural-termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Pusher time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_reward_distance + cumulative_reward_near + cumulative_reward_control,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Pusher cumulative reward decomposition drifted")
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "action_squared_norm": action_squared_norm,
        "cumulative_action_squared_norm": cumulative_action_squared_norm,
        "mean_action_squared_norm": cumulative_action_squared_norm / step_count,
        "fingertip_object_delta_x": object_position[0] - fingertip[0],
        "fingertip_object_delta_y": object_position[1] - fingertip[1],
        "fingertip_object_delta_z": object_position[2] - fingertip[2],
        "fingertip_object_distance": fingertip_object_distance,
        "minimum_fingertip_object_distance": minimum_fingertip_object_distance,
        "object_goal_delta_x": goal[0] - object_position[0],
        "object_goal_delta_y": goal[1] - object_position[1],
        "object_goal_delta_z": goal[2] - object_position[2],
        "initial_object_goal_distance": initial_object_goal_distance,
        "object_goal_distance": object_goal_distance,
        "minimum_object_goal_distance": minimum_object_goal_distance,
        "maximum_object_goal_distance": maximum_object_goal_distance,
        "object_goal_distance_reduction": (initial_object_goal_distance - object_goal_distance),
        "object_goal_fraction_remaining": (object_goal_distance / initial_object_goal_distance),
        "object_moved_toward_goal": (object_goal_distance < initial_object_goal_distance),
        "object_displacement_x": object_position[0] - initial_object[0],
        "object_displacement_y": object_position[1] - initial_object[1],
        "object_displacement_z": object_position[2] - initial_object[2],
        "object_displacement": object_displacement,
        "maximum_object_displacement": maximum_object_displacement,
        "reward_distance": reward_terms["reward_distance"],
        "reward_near": reward_terms["reward_near"],
        "reward_control": reward_terms["reward_control"],
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_distance": cumulative_reward_distance,
        "cumulative_reward_near": cumulative_reward_near,
        "cumulative_reward_control": cumulative_reward_control,
        "cumulative_return": cumulative_return,
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _point(
    observation: dict[str, PolicyValue],
    prefix: str,
) -> tuple[float, float, float]:
    values: list[float] = []
    for axis in ("x", "y", "z"):
        value = observation.get(f"{prefix}_{axis}")
        if type(value) is not float:
            raise RuntimeError(f"Pusher returned invalid {prefix} position")
        values.append(value)
    return values[0], values[1], values[2]


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Pusher returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Pusher returned a non-finite {name}")
    return number


__all__ = ["PusherEnvironment"]
