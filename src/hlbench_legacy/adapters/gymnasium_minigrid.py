"""Gymnasium Minigrid adapter with a policy-safe observation surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert common Gym/NumPy values into JSON-compatible data."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_record(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "info": self.info,
        }


class GymnasiumMinigridAdapter:
    """Thin adapter around Minigrid's Gymnasium environments.

    The policy receives the public observation returned by the environment
    (`image`, `direction`, `mission`) plus action-count metadata. It does not
    receive the full grid, object coordinates, or evaluator seed splits.
    """

    def __init__(self, env_id: str, max_steps: int | None = None) -> None:
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401  # Registers MiniGrid env IDs.
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Gymnasium Minigrid dependencies are missing. "
                "Install with `python -m pip install -e .[dev]`."
            ) from exc

        kwargs: dict[str, Any] = {}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        self.env = gym.make(env_id, **kwargs)
        self.env_id = env_id

    @property
    def action_count(self) -> int:
        return int(self.env.action_space.n)

    def reset(self, seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
        del config
        observation, _info = self.env.reset(seed=seed)
        return self._public_observation(observation)

    def step(self, action: int) -> StepResult:
        if not isinstance(action, int):
            raise TypeError(f"action must be an int, got {type(action).__name__}")
        if action < 0 or action >= self.action_count:
            raise ValueError(f"invalid action {action}; expected [0, {self.action_count})")
        observation, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            observation=self._public_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self._public_info(info),
        )

    def close(self) -> None:
        self.env.close()

    def _public_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        public = {
            "image": _jsonable(observation.get("image")),
            "direction": _jsonable(observation.get("direction")),
            "mission": _jsonable(observation.get("mission")),
            "action_count": self.action_count,
        }
        return public

    def _public_info(self, info: dict[str, Any]) -> dict[str, Any]:
        # Minigrid info is usually empty. Keep only JSON-compatible public keys.
        return _jsonable(info)

