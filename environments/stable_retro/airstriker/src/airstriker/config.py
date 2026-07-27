"""Typed Stable-Retro Airstriker configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AirstrikerConfig:
    """Configuration fixing the one redistributable Stable-Retro game."""

    game: str = "Airstriker-Genesis-v0"
    state: str = "Level1"

    def __post_init__(self) -> None:
        if type(self.game) is not str:
            raise TypeError("game must be an exact string")
        if self.game != "Airstriker-Genesis-v0":
            raise ValueError("only the ROM-backed Airstriker game is portable")
        if type(self.state) is not str:
            raise TypeError("state must be an exact string")
        if self.state != "Level1":
            raise ValueError("only the bundled Level1 state is supported")


__all__ = ["AirstrikerConfig"]
