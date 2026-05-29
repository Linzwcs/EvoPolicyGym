"""Env registry: register_env() and get_env().

Each env declares everything the server needs to know:
- gymnasium factory
- obs/action spaces (mirrored into env_meta)
- max_episode_steps
- expert_baseline / random_baseline (server-internal, NOT exposed to agent)
- n_env_instances (default 256, env may override)
- obs_storage ("inline" or "external")
- reward_components (optional dict, see SPEC.md §1.1)
- paths to train.json and heldout.json

See docs/architecture.md §4.5 for details.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvDefinition:
    """Immutable per-env registration record."""

    env_id: str
    env_version: str
    factory: Callable[[], Any]  # () -> gymnasium.Env
    obs_space: dict[str, Any]
    action_space: dict[str, Any]
    max_episode_steps: int
    # Server-internal; never exposed via /info, /task, or env_meta.
    expert_baseline: float
    random_baseline: float
    train_seeds_path: Path
    heldout_seeds_path: Path
    n_env_instances: int = 256
    obs_storage: str = "inline"  # "inline" or "external"
    reward_components: dict[str, str] | None = None
    # Optional per-env TASK.md source file; Server reads it on demand for
    # ``GET /task``. Never staged into the workspace. If None or missing,
    # the server returns a minimal placeholder.
    task_md_path: Path | None = None
    # Optional per-env starter Policy. Server copies it into
    # ``workspace/system/policy.py`` at run init *if no policy.py exists
    # yet*. The starter MUST be a valid (but optionally bad) Policy so
    # turn 0's first submit can succeed and the agent immediately sees
    # the contract for ``__init__`` / ``reset`` / ``act``.
    starter_policy_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def public_env_meta(self) -> dict[str, Any]:
        """Return the subset of env_meta exposed via GET /info.

        Excludes server-internal fields (baselines, seed paths).
        See SPEC.md §1.1.
        """
        meta: dict[str, Any] = {
            "obs_space": self.obs_space,
            "action_space": self.action_space,
            "max_episode_steps": self.max_episode_steps,
            "n_env_instances": self.n_env_instances,
            "obs_storage": self.obs_storage,
        }
        if self.reward_components:
            meta["reward_components"] = self.reward_components
        return meta


_REGISTRY: dict[str, EnvDefinition] = {}


def register_env(
    *,
    env_id: str,
    env_version: str,
    factory: Callable[[], Any],
    obs_space: dict[str, Any],
    action_space: dict[str, Any],
    max_episode_steps: int,
    expert_baseline: float,
    random_baseline: float,
    train_seeds_path: Path,
    heldout_seeds_path: Path,
    n_env_instances: int = 256,
    obs_storage: str = "inline",
    reward_components: dict[str, str] | None = None,
    task_md_path: Path | None = None,
    starter_policy_path: Path | None = None,
    **extras: Any,
) -> None:
    """Register an env. Typically called from envs/<id>/__init__.py."""
    if env_id in _REGISTRY:
        raise ValueError(f"env {env_id!r} already registered")
    if obs_storage not in ("inline", "external"):
        raise ValueError(f"obs_storage must be 'inline' or 'external', got {obs_storage!r}")
    _REGISTRY[env_id] = EnvDefinition(
        env_id=env_id,
        env_version=env_version,
        factory=factory,
        obs_space=obs_space,
        action_space=action_space,
        max_episode_steps=max_episode_steps,
        expert_baseline=expert_baseline,
        random_baseline=random_baseline,
        train_seeds_path=train_seeds_path,
        heldout_seeds_path=heldout_seeds_path,
        n_env_instances=n_env_instances,
        obs_storage=obs_storage,
        reward_components=reward_components,
        task_md_path=task_md_path,
        starter_policy_path=starter_policy_path,
        extras=extras,
    )


def get_env(env_id: str) -> EnvDefinition:
    """Look up a registered env. Raises KeyError if unknown."""
    if env_id not in _REGISTRY:
        raise KeyError(f"env {env_id!r} not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[env_id]


def list_envs() -> list[str]:
    """All registered env IDs (sorted)."""
    return sorted(_REGISTRY)
