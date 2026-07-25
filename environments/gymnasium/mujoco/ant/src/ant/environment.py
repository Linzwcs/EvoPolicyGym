"""One fresh Gymnasium Ant-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import AntConfig

_BODY_FIELDS = (
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    "front_left_hip_angle",
    "front_left_ankle_angle",
    "front_right_hip_angle",
    "front_right_ankle_angle",
    "back_left_hip_angle",
    "back_left_ankle_angle",
    "back_right_hip_angle",
    "back_right_ankle_angle",
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    "front_left_hip_angular_velocity",
    "front_left_ankle_angular_velocity",
    "front_right_hip_angular_velocity",
    "front_right_ankle_angular_velocity",
    "back_left_hip_angular_velocity",
    "back_left_ankle_angular_velocity",
    "back_right_hip_angular_velocity",
    "back_right_ankle_angular_velocity",
)
_CONTACT_BODIES = (
    "torso",
    "front_left_leg",
    "front_left_aux",
    "front_left_ankle",
    "front_right_leg",
    "front_right_aux",
    "front_right_ankle",
    "back_left_leg",
    "back_left_aux",
    "back_left_ankle",
    "back_right_leg",
    "back_right_aux",
    "back_right_ankle",
)
_CONTACT_COMPONENTS = (
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
)


class AntEnvironment:
    """The seeded strict adapter around configured Ant-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: AntConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not AntConfig:
            raise TypeError("config must be AntConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Ant configuration belongs in AntConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._exclude_positions = (
            config.exclude_current_positions_from_observation
        )
        self._include_contact = config.include_cfrc_ext_in_observation
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Ant-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                contact_cost_weight=config.contact_cost_weight,
                healthy_reward=config.healthy_reward,
                main_body=config.main_body,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
                contact_force_range=config.contact_force_range,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
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
        return _observation(
            observation,
            exclude_positions=self._exclude_positions,
            include_contact=self._include_contact,
        )

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
            raise RuntimeError("Ant returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_observation(
                observation,
                exclude_positions=self._exclude_positions,
                include_contact=self._include_contact,
            ),
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
    if type(value) is not list or len(value) != 8:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    exclude_positions: bool,
    include_contact: bool,
) -> dict[str, PolicyValue]:
    expected_length = (
        27
        + (0 if exclude_positions else 2)
        + (78 if include_contact else 0)
    )
    if (
        type(value) is not numpy.ndarray
        or value.shape != (expected_length,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Ant returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
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
        _BODY_FIELDS,
        value[offset : offset + len(_BODY_FIELDS)],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    offset += len(_BODY_FIELDS)
    if include_contact:
        observation["contact_forces"] = _contact_forces(
            value[offset:],
        )
    return observation


def _contact_forces(value: NDArray[numpy.float64]) -> PolicyValue:
    if value.shape != (78,):
        raise RuntimeError("Ant returned invalid contact forces")
    contacts: dict[str, PolicyValue] = {}
    for body_index, body in enumerate(_CONTACT_BODIES):
        start = body_index * len(_CONTACT_COMPONENTS)
        contacts[body] = {
            component: _number(
                value[start + component_index],
                name=f"{body} contact {component}",
            )
            for component_index, component in enumerate(_CONTACT_COMPONENTS)
        }
    return contacts


def _metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError("Ant returned invalid metrics")
    names = (
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "reward_forward",
        "reward_ctrl",
        "reward_contact",
        "reward_survive",
    )
    if not set(names).issubset(value):
        raise RuntimeError("Ant omitted public metrics")
    return {
        (
            "reward_control" if name == "reward_ctrl" else name
        ): _number(value[name], name=name.replace("_", " "))
        for name in names
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Ant returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Ant returned a non-finite {name}")
    return number


__all__ = ["AntEnvironment"]
