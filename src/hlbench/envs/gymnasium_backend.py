"""Generic Gymnasium environment backend."""

from __future__ import annotations

import math
from typing import Any

from hlbench.core.scenario import ScenarioSpec
from hlbench.core.task import EnvContract
from hlbench.envs.base import StepResult
from hlbench.envs.space_schema import space_to_schema, validate_action
from hlbench.envs.wrappers import PublicObservationWrapper


class GymnasiumEnvInstance:
    def __init__(self, env: Any, wrapper: PublicObservationWrapper) -> None:
        self.env = env
        self.wrapper = wrapper
        self._action_schema = space_to_schema(env.action_space)

    @property
    def action_count(self) -> int | None:
        return int(self.env.action_space.n) if hasattr(self.env.action_space, "n") else None

    @property
    def action_schema(self) -> dict[str, Any]:
        return self._action_schema

    def reset(self, seed: int, config: dict[str, Any] | None = None) -> Any:
        del config
        observation, _info = self.env.reset(seed=seed)
        return self.wrapper.observation(observation, action_count=self.action_count)

    def step(self, action: Any) -> StepResult:
        action = validate_action(self.action_schema, action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            observation=self.wrapper.observation(observation, action_count=self.action_count),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.wrapper.info(info),
        )

    def close(self) -> None:
        self.env.close()


class GymnasiumBackend:
    name = "gymnasium"

    def describe(self, scenario: ScenarioSpec) -> EnvContract:
        env = self._make_raw_env(scenario)
        try:
            reward_range = getattr(env, "reward_range", (None, None))
            low, high = _clean_bound(reward_range[0]), _clean_bound(reward_range[1])
            return EnvContract(
                backend=self.name,
                env_id=scenario.env_id,
                observation_schema=_with_meanings(
                    space_to_schema(env.observation_space),
                    scenario.observation_meanings,
                    "dimensions",
                ),
                action_schema=_with_meanings(
                    space_to_schema(env.action_space),
                    scenario.action_meanings,
                    _action_meaning_key(space_to_schema(env.action_space)),
                ),
                reward_range=(low, high),
                termination={
                    "terminated": "Environment-defined terminal state.",
                    "truncated": f"Episode hit the scenario or environment step limit ({scenario.max_steps}).",
                },
                public_info_schema={"type": "dict", "visibility": "train rollouts only"},
            )
        finally:
            env.close()

    def make(self, scenario: ScenarioSpec) -> GymnasiumEnvInstance:
        return GymnasiumEnvInstance(
            self._make_raw_env(scenario),
            PublicObservationWrapper(mode=scenario.observation_mode),
        )

    def _make_raw_env(self, scenario: ScenarioSpec) -> Any:
        try:
            import gymnasium as gym
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError("Gymnasium backend requires the `gymnasium` package.") from exc
        try:
            return gym.make(scenario.env_id, **scenario.env_kwargs)
        except ModuleNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(f"could not create Gymnasium environment {scenario.env_id!r}: {exc}") from exc


def _with_meanings(schema: dict[str, Any], meanings: tuple[dict[str, Any], ...], key: str) -> dict[str, Any]:
    if not meanings:
        return schema
    merged = dict(schema)
    merged[key] = [dict(item) for item in meanings]
    return merged


def _action_meaning_key(schema: dict[str, Any]) -> str:
    return "dimensions" if schema.get("type") == "box" else "actions"


def _clean_bound(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
