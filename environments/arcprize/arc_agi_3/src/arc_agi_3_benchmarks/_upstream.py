"""Narrow structural contracts around the pinned official ARC toolkit."""

from __future__ import annotations

from typing import Any, Protocol

from arc_agi import Arcade, EnvironmentScorecard, OperationMode  # type: ignore[import-untyped]
from arcengine import FrameDataRaw, GameAction


class EnvironmentWrapperLike(Protocol):
    @property
    def observation_space(self) -> FrameDataRaw | None: ...

    def reset(self) -> FrameDataRaw | None: ...

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameDataRaw | None: ...


class ArcadeLike(Protocol):
    def create_scorecard(
        self,
        source_url: str | None = None,
        tags: list[str] | None = None,
        opaque: Any | None = None,
    ) -> str: ...

    def make(
        self,
        game_id: str,
        seed: int = 0,
        scorecard_id: str | None = None,
        save_recording: bool = False,
        include_frame_data: bool = True,
        render_mode: str | None = None,
        renderer: Any | None = None,
    ) -> EnvironmentWrapperLike | None: ...

    def close_scorecard(
        self,
        scorecard_id: str | None = None,
    ) -> EnvironmentScorecard | None: ...


def create_arcade(
    *,
    arc_api_key: str,
    arc_base_url: str,
    environments_dir: str,
    recordings_dir: str,
) -> Arcade:
    """Create the Host-owned official client without exposing its settings."""

    return Arcade(
        arc_api_key=arc_api_key,
        arc_base_url=arc_base_url,
        operation_mode=OperationMode.NORMAL,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )


__all__ = ["ArcadeLike", "EnvironmentWrapperLike", "create_arcade"]
