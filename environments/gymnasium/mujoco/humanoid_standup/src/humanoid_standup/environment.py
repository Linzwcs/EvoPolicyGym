"""One fresh Gymnasium HumanoidStandup-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import HumanoidStandupConfig

_JOINTS = (
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder_1",
    "right_shoulder_2",
    "right_elbow",
    "left_shoulder_1",
    "left_shoulder_2",
    "left_elbow",
)
_STATE_FIELDS = (
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    *(f"{joint}_angle" for joint in _JOINTS),
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    *(f"{joint}_angular_velocity" for joint in _JOINTS),
)
_BODIES = (
    "torso",
    "lower_waist",
    "pelvis",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_upper_arm",
    "right_lower_arm",
    "left_upper_arm",
    "left_lower_arm",
)
_INERTIA_COMPONENTS = (
    "inertia_upper_0",
    "inertia_upper_1",
    "inertia_upper_2",
    "inertia_upper_3",
    "inertia_upper_4",
    "inertia_upper_5",
    "mass_times_com_offset_x",
    "mass_times_com_offset_y",
    "mass_times_com_offset_z",
    "mass",
)
_BODY_VELOCITY_COMPONENTS = (
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
)
_EXTERNAL_FORCE_COMPONENTS = (
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
)


class HumanoidStandupEnvironment:
    """The seeded strict adapter around configured HumanoidStandup-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HumanoidStandupConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HumanoidStandupConfig:
            raise TypeError("config must be HumanoidStandupConfig")
        if episode.scenario is not None:
            raise ValueError(
                "HumanoidStandup configuration belongs in "
                "HumanoidStandupConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        impact_lower = (
            float("-inf")
            if config.impact_cost_range[0] is None
            else config.impact_cost_range[0]
        )
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "HumanoidStandup-v5",
                frame_skip=config.frame_skip,
                ctrl_cost_weight=config.ctrl_cost_weight,
                impact_cost_weight=config.impact_cost_weight,
                impact_cost_range=(
                    impact_lower,
                    config.impact_cost_range[1],
                ),
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
                include_cinert_in_observation=(
                    config.include_cinert_in_observation
                ),
                include_cvel_in_observation=(
                    config.include_cvel_in_observation
                ),
                include_qfrc_actuator_in_observation=(
                    config.include_qfrc_actuator_in_observation
                ),
                include_cfrc_ext_in_observation=(
                    config.include_cfrc_ext_in_observation
                ),
            ),
        )
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation, config=self._config)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        observation, reward, terminated, truncated, info = (
            self._environment.step(_action(action))
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "HumanoidStandup returned invalid termination flags"
            )
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation, config=self._config),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
            metrics=_metrics(info),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 17:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -0.4 <= item <= 0.4
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    config: HumanoidStandupConfig,
) -> dict[str, PolicyValue]:
    expected_length = (
        len(_STATE_FIELDS)
        + (
            0
            if config.exclude_current_positions_from_observation
            else 2
        )
        + 130 * config.include_cinert_in_observation
        + 78 * config.include_cvel_in_observation
        + 17 * config.include_qfrc_actuator_in_observation
        + 78 * config.include_cfrc_ext_in_observation
    )
    if (
        type(value) is not numpy.ndarray
        or value.shape != (expected_length,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError(
            "HumanoidStandup returned an invalid observation"
        )
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        observation["torso_x_position"] = _number(
            value[0],
            name="torso x position",
        )
        observation["torso_y_position"] = _number(
            value[1],
            name="torso y position",
        )
        offset = 2
    for name, item in zip(
        _STATE_FIELDS,
        value[offset : offset + len(_STATE_FIELDS)],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    offset += len(_STATE_FIELDS)
    if config.include_cinert_in_observation:
        observation["body_inertias"] = _body_matrix(
            value[offset : offset + 130],
            components=_INERTIA_COMPONENTS,
            name="body inertia",
        )
        offset += 130
    if config.include_cvel_in_observation:
        observation["body_velocities"] = _body_matrix(
            value[offset : offset + 78],
            components=_BODY_VELOCITY_COMPONENTS,
            name="body velocity",
        )
        offset += 78
    if config.include_qfrc_actuator_in_observation:
        observation["actuator_forces"] = {
            joint: _number(
                value[offset + index],
                name=f"{joint} actuator force",
            )
            for index, joint in enumerate(_JOINTS)
        }
        offset += 17
    if config.include_cfrc_ext_in_observation:
        observation["external_forces"] = _body_matrix(
            value[offset : offset + 78],
            components=_EXTERNAL_FORCE_COMPONENTS,
            name="external force",
        )
    return observation


def _body_matrix(
    value: NDArray[numpy.float64],
    *,
    components: tuple[str, ...],
    name: str,
) -> PolicyValue:
    if value.shape != (len(_BODIES) * len(components),):
        raise RuntimeError(
            f"HumanoidStandup returned invalid {name} values"
        )
    bodies: dict[str, PolicyValue] = {}
    for body_index, body in enumerate(_BODIES):
        start = body_index * len(components)
        bodies[body] = {
            component: _number(
                value[start + component_index],
                name=f"{body} {name} {component}",
            )
            for component_index, component in enumerate(components)
        }
    return bodies


def _metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("HumanoidStandup returned invalid metrics")
    scalar_names = (
        "x_position",
        "y_position",
        "z_distance_from_origin",
        "reward_linup",
        "reward_quadctrl",
        "reward_impact",
    )
    if not set((*scalar_names, "tendon_length", "tendon_velocity")).issubset(
        value
    ):
        raise RuntimeError("HumanoidStandup omitted public metrics")
    renamed = {
        "reward_linup": "reward_upward",
        "reward_quadctrl": "reward_control",
    }
    metrics: dict[str, PolicyValue] = {
        renamed.get(name, name): _number(
            value[name],
            name=name.replace("_", " "),
        )
        for name in scalar_names
    }
    metrics["reward_constant"] = 1.0
    metrics["tendon_lengths"] = _tendons(
        value["tendon_length"],
        name="tendon length",
    )
    metrics["tendon_velocities"] = _tendons(
        value["tendon_velocity"],
        name="tendon velocity",
    )
    return metrics


def _tendons(value: object, *, name: str) -> PolicyValue:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (2,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError(
            f"HumanoidStandup returned invalid {name} values"
        )
    return {
        "left_hip_to_knee": _number(value[0], name=f"left {name}"),
        "right_hip_to_knee": _number(value[1], name=f"right {name}"),
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(
            f"HumanoidStandup returned an invalid {name}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"HumanoidStandup returned a non-finite {name}"
        )
    return number


__all__ = ["HumanoidStandupEnvironment"]
