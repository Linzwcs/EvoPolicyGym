"""Runtime fixes for optional MiniGrid package assets."""

from __future__ import annotations

from pathlib import Path

_WFC_PREFIX = "MiniGrid-WFC-"
_WFC_PATTERNS = {
    "MazeSimple": "SimpleMaze.png",
    "DungeonMazeScaled": "ScaledMaze.png",
    "RoomsFabric": "Fabric.png",
    "ObstaclesBlackdots": "Blackdots.png",
    "ObstaclesAngular": "Angular.png",
    "ObstaclesHogs3": "Hogs.png",
}
_WFC_ASSET_DIR = Path(__file__).with_name("assets") / "minigrid_wfc_patterns"


def patch_minigrid_wfc_assets(env_id: str) -> None:
    """Point MiniGrid WFC presets at vendored pattern PNGs when package data is missing."""

    if not env_id.startswith(_WFC_PREFIX):
        return
    try:
        from minigrid.envs.wfc import config
    except Exception:
        return

    for preset_name, filename in _WFC_PATTERNS.items():
        preset = config.WFC_PRESETS_ALL.get(preset_name)
        if preset is None:
            continue
        if Path(preset.pattern_path).exists():
            continue
        asset = _WFC_ASSET_DIR / filename
        if asset.exists():
            preset.pattern_path = asset


__all__ = ["patch_minigrid_wfc_assets"]
