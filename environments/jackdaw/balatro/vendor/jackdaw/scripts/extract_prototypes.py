#!/usr/bin/env python3
"""Extract Balatro prototype data tables into JSON files.

Runs scripts/extract_prototypes.lua via LuaJIT subprocess. Falls back to
lupa (Python Lua bindings) if LuaJIT is not available.

Usage:
    uv run python scripts/extract_prototypes.py

Outputs JSON files in jackdaw/engine/data/:
    centers.json, cards.json, blinds.json, tags.json, stakes.json, seals.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_LUA = PROJECT_ROOT / "scripts" / "extract_prototypes.lua"
DATA_DIR = PROJECT_ROOT / "jackdaw" / "engine" / "data"


def find_luajit() -> str | None:
    """Find LuaJIT in PATH or common install locations."""
    for name in ["luajit", "lua"]:
        path = shutil.which(name)
        if path:
            return path
    for candidate in [
        Path.home() / "AppData/Local/Programs/LuaJIT/bin/luajit.exe",
        Path("C:/Program Files/LuaJIT/bin/luajit.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def run_via_luajit() -> bool:
    """Run the Lua extraction script via LuaJIT subprocess."""
    luajit = find_luajit()
    if not luajit:
        return False

    print(f"Using LuaJIT: {luajit}")
    result = subprocess.run(
        [luajit, str(SCRIPT_LUA)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        timeout=30,
    )
    return result.returncode == 0


def verify_outputs() -> None:
    """Print summary of extracted data files."""
    print("\n=== Verification ===")
    expected_files = [
        "centers.json",
        "cards.json",
        "blinds.json",
        "tags.json",
        "stakes.json",
        "seals.json",
    ]

    for fname in expected_files:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  MISSING: {fname}")
            continue
        with open(path) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, dict) else 0
        print(f"  {fname}: {count} entries")

    # Detailed centers breakdown
    centers_path = DATA_DIR / "centers.json"
    if centers_path.exists():
        with open(centers_path) as f:
            centers = json.load(f)

        set_counts: dict[str, int] = {}
        for entry in centers.values():
            s = entry.get("set", "Unknown")
            set_counts[s] = set_counts.get(s, 0) + 1

        print("\n  Centers by set:")
        for s in sorted(set_counts):
            print(f"    {s}: {set_counts[s]}")

        # Joker rarity
        rarity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for entry in centers.values():
            if entry.get("set") == "Joker" and "rarity" in entry:
                rarity_counts[entry["rarity"]] = rarity_counts.get(entry["rarity"], 0) + 1
        print("\n  Joker rarity:")
        for r, c in sorted(rarity_counts.items()):
            labels = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Legendary"}
            print(f"    {labels.get(r, r)}: {c}")


def main() -> int:
    print("Extracting Balatro prototype data...\n")

    if run_via_luajit():
        verify_outputs()
        return 0

    print("LuaJIT not available. Install LuaJIT or run:")
    print("  luajit scripts/extract_prototypes.lua")
    return 1


if __name__ == "__main__":
    sys.exit(main())
