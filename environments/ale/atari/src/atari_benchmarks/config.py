"""Typed redistributable ALE Atari configuration."""

from dataclasses import dataclass

ATARI_PROFILES = ("Tetris",)


@dataclass(frozen=True, slots=True)
class AtariConfig:
    """Configuration fixing one redistributable Atari game."""

    game: str = "Tetris"

    def __post_init__(self) -> None:
        if type(self.game) is not str:
            raise TypeError("game must be an exact string")
        if self.game not in ATARI_PROFILES:
            raise ValueError(
                "only the ROM-backed Tetris profile is currently portable"
            )


__all__ = ["ATARI_PROFILES", "AtariConfig"]
