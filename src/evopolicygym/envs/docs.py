"""Task document helpers for built-in environments."""

from __future__ import annotations

from pathlib import Path


def task(path: str) -> str:
    """Read the task markdown colocated with an environment module."""

    return (Path(path).with_name("task.md")).read_text(encoding="utf-8")
