"""Harness preset values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessPreset:
    train_episodes: int | None = 2
    timeout_seconds: int = 1800


PRESETS: dict[str, HarnessPreset] = {
    "smoke": HarnessPreset(train_episodes=1, timeout_seconds=300),
    "default": HarnessPreset(),
}


def get_preset(name: str) -> HarnessPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}") from exc
