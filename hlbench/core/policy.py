"""Policy loading and interface helpers."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any


class PolicyLoadError(RuntimeError):
    """Raised when a candidate policy cannot be imported."""


def load_policy(policy_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("hlbench_candidate_policy", policy_path)
    if spec is None or spec.loader is None:
        raise PolicyLoadError(f"could not import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        policy_cls = getattr(module, "Policy")
    except AttributeError as exc:
        raise PolicyLoadError(f"{policy_path} must define class Policy") from exc
    return policy_cls()


def reset_policy(policy: Any, *, task_config: dict[str, Any]) -> None:
    if not hasattr(policy, "reset"):
        return
    try:
        policy.reset(task_config=task_config)
    except TypeError:
        try:
            policy.reset(task_config)
        except TypeError:
            policy.reset()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
