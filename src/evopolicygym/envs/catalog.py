"""Environment registry."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import Env


@dataclass(frozen=True, slots=True)
class Registry:
    """In-memory environment catalog."""

    envs: dict[str, Env]

    def get(self, name: str) -> Env:
        try:
            return self.envs[name]
        except KeyError as exc:
            raise KeyError(f"unknown environment: {name}") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self.envs))

    @classmethod
    def of(cls, *envs: Env) -> Registry:
        return cls({env.task.name: env for env in envs})
