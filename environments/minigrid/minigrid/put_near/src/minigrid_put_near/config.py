"""Typed, public MiniGrid PutNear environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES: dict[str, tuple[str, int, int]] = {
    "6x6-N2": ("MiniGrid-PutNear-6x6-N2-v0", 6, 2),
    "8x8-N3": ("MiniGrid-PutNear-8x8-N3-v0", 8, 3),
}


@dataclass(frozen=True, slots=True)
class PutNearConfig:
    """Parameters that define one MiniGrid PutNear Benchmark identity."""

    profile: str = "8x8-N3"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def environment_id(self) -> str:
        """Return the upstream Gymnasium registration."""

        return _PROFILES[self.profile][0]

    @property
    def size(self) -> int:
        """Return the square grid size."""

        return _PROFILES[self.profile][1]

    @property
    def object_count(self) -> int:
        """Return the generated object count."""

        return _PROFILES[self.profile][2]

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream horizon."""

        return 5 * self.size


__all__ = ["PutNearConfig"]

