"""Filesystem layout helpers for EvoPolicyGym run roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Layout:
    """Build canonical run paths under a runs directory."""

    base: Path = Path("runs")

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", Path(self.base))

    def run(self, *, model: str, env: str, exp: str) -> Path:
        return self.base / slug(model, "model") / slug(env, "env") / slug(exp, "exp")


def root(base: str | Path, *, model: str, env: str, exp: str) -> Path:
    """Return `base/model/env/exp` with filesystem-safe path segments."""

    return Layout(Path(base)).run(model=model, env=env, exp=exp)


def slug(value: str, name: str = "value") -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{name} must not be empty")
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    text = text.strip("._-")
    if not text:
        raise ValueError(f"{name} must contain an alphanumeric character")
    return text
