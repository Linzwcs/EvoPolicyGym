"""Shared filesystem paths."""

from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1] if PACKAGE_ROOT.parent.name == "src" else PACKAGE_ROOT.parent
SCENARIOS_ROOT = PACKAGE_ROOT / "scenarios"
RUNS_ROOT = REPO_ROOT / "runs"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "unnamed"


def run_root(*, model_name: str, env_name: str, run_id: str) -> Path:
    return RUNS_ROOT / slugify(model_name) / slugify(env_name) / slugify(run_id)
