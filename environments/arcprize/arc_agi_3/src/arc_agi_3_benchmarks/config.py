"""Typed Host-selected ARC-AGI-3 game collections."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Full versioned IDs returned by the official discovery endpoint on 2026-08-01.
# Keeping the versions here prevents a public Benchmark identity from drifting when
# ARC Prize publishes a new implementation under the same four-character slug.
ARC_AGI_3_PUBLIC_GAMES = (
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
)

_GAME_ID = re.compile(r"^[A-Za-z0-9]{4}(?:-[A-Za-z0-9]+)?$")


@dataclass(frozen=True, slots=True)
class ArcAgi3Config:
    """Configuration fixing one ARC-AGI-3 game collection."""

    profile: str = "public-25"
    custom_game_ids: tuple[str, ...] = ()
    max_episode_steps: int = 1_000

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in {"public-25", "custom"}:
            raise ValueError("profile must be 'public-25' or 'custom'")
        if type(self.custom_game_ids) is not tuple:
            raise TypeError("custom_game_ids must be an exact tuple")
        if any(type(game_id) is not str for game_id in self.custom_game_ids):
            raise TypeError("custom_game_ids must contain exact strings")
        if self.profile == "public-25" and self.custom_game_ids:
            raise ValueError("custom_game_ids may be set only for the custom profile")
        if self.profile == "custom" and not self.custom_game_ids:
            raise ValueError("custom_game_ids must be non-empty for the custom profile")
        if type(self.max_episode_steps) is not int or not 1 <= self.max_episode_steps <= 100_000:
            raise ValueError("max_episode_steps must be an integer between 1 and 100000")

        game_ids = self.game_ids
        if any(_GAME_ID.fullmatch(game_id) is None for game_id in game_ids):
            raise ValueError(
                "game IDs must be four alphanumeric characters with an "
                "optional alphanumeric version suffix"
            )
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("game IDs must not contain duplicates")
        base_ids = tuple(game_id.split("-", 1)[0] for game_id in game_ids)
        if len(set(base_ids)) != len(base_ids):
            raise ValueError("game IDs must not repeat a four-character slug")

    @property
    def game_ids(self) -> tuple[str, ...]:
        if self.profile == "public-25":
            return ARC_AGI_3_PUBLIC_GAMES
        return self.custom_game_ids


__all__ = [
    "ARC_AGI_3_PUBLIC_GAMES",
    "ArcAgi3Config",
]
