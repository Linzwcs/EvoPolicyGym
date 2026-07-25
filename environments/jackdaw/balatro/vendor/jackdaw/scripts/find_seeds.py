#!/usr/bin/env python3
"""Find seeds that collectively cover all tags and boss blinds fastest.

Usage:
    uv run python scripts/find_seeds.py
    uv run python scripts/find_seeds.py --max-seeds 5000 --max-ante 8

This script runs actual games through the engine (using greedy_play_agent)
to see which tags and bosses appear at each ante. This is necessary because
the RNG state depends on the full game flow — skipping/playing blinds, shop
interactions, card draws, etc. all advance the RNG.

Output: a minimal set of (seed, ante) pairs that see every tag and boss blind.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from jackdaw.engine.actions import (
    GamePhase,
    get_legal_actions,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent

# ---------------------------------------------------------------------------
# All tags and boss blinds we need to cover
# ---------------------------------------------------------------------------

ALL_TAGS = {
    # Ante 1+
    "tag_economy",
    "tag_skip",
    "tag_charm",
    "tag_boss",
    "tag_d_six",
    "tag_juggle",
    "tag_coupon",
    "tag_voucher",
    "tag_double",
    "tag_investment",
    "tag_uncommon",
    "tag_rare",
    "tag_foil",
    "tag_holo",
    "tag_polychrome",
    # Ante 2+
    "tag_garbage",
    "tag_handy",
    "tag_orbital",
    "tag_top_up",
    "tag_buffoon",
    "tag_ethereal",
    "tag_meteor",
    "tag_standard",
    "tag_negative",
}

ALL_BOSSES = {
    # Ante 1+
    "bl_club",
    "bl_goad",
    "bl_head",
    "bl_hook",
    "bl_manacle",
    "bl_pillar",
    "bl_psychic",
    "bl_window",
    # Ante 2+
    "bl_arm",
    "bl_fish",
    "bl_flint",
    "bl_house",
    "bl_mark",
    "bl_mouth",
    "bl_needle",
    "bl_wall",
    "bl_water",
    "bl_wheel",
    # Ante 3+
    "bl_tooth",
    "bl_eye",
    # Ante 4+
    "bl_plant",
    # Ante 5+
    "bl_serpent",
    # Ante 6+
    "bl_ox",
    # Showdown (ante 8)
    "bl_final_acorn",
    "bl_final_bell",
    "bl_final_heart",
    "bl_final_leaf",
    "bl_final_vessel",
}


def _cheat_agent(gs: dict, legal: list) -> Any:
    """Agent that plays through all blinds using chip cheat.

    Selects every blind (never skips) and cheats chips after one hand
    so greedy_play_agent can always beat the blind. Matches the validator's
    play-through strategy.
    """
    phase = gs.get("phase")

    # After playing one hand, cheat chips to beat the blind
    if phase == GamePhase.SELECTING_HAND:
        blind = gs.get("blind")
        chips = gs.get("chips", 0)
        cr = gs.get("current_round", {})
        if blind and chips > 0 and cr.get("hands_left", 0) <= 3:
            gs["chips"] = getattr(blind, "chips", 600) - 1

    return greedy_play_agent(gs, legal)


def scan_seed(seed: str, max_ante: int) -> dict[str, Any]:
    """Play a full game, skipping Small/Big blinds, recording tags and bosses.

    Cheats by giving extra hands on boss blinds so the greedy agent can
    always survive to higher antes.

    Returns {"tags": set, "bosses": set, "tag_ante": dict, "boss_ante": dict}
    """
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"

    tags_seen: set[str] = set()
    bosses_seen: set[str] = set()
    tag_ante: dict[str, int] = {}
    tag_pos: dict[str, str] = {}  # tag_key -> "Small" or "Big"
    boss_ante: dict[str, int] = {}

    max_actions = 10000
    actions_taken = 0

    while actions_taken < max_actions:
        phase = gs.get("phase")
        if phase == GamePhase.GAME_OVER:
            break

        ante = gs.get("round_resets", {}).get("ante", 1)
        if ante > max_ante:
            break

        # At BLIND_SELECT, record the tags and boss for this ante
        if phase == GamePhase.BLIND_SELECT:
            blind_on_deck = gs.get("blind_on_deck", "")

            # Record tags (available when Small/Big are on deck)
            if blind_on_deck in ("Small", "Big"):
                blind_tags = gs.get("round_resets", {}).get("blind_tags", {})
                for pos, tag_key in blind_tags.items():
                    if tag_key and tag_key not in tags_seen:
                        tags_seen.add(tag_key)
                        tag_ante[tag_key] = ante
                        tag_pos[tag_key] = pos

            # Record boss (available when Boss is on deck)
            if blind_on_deck == "Boss":
                blind_choices = gs.get("round_resets", {}).get("blind_choices", {})
                boss_key = blind_choices.get("Boss", "")
                if boss_key and boss_key not in bosses_seen:
                    bosses_seen.add(boss_key)
                    boss_ante[boss_key] = ante

        # Use skip agent (skips Small/Big, cheats hands, plays Boss greedily)
        legal = get_legal_actions(gs)
        if not legal:
            break

        action = _cheat_agent(gs, legal)
        step(gs, action)
        actions_taken += 1

        # Early exit if won
        if gs.get("won") and gs.get("phase") == GamePhase.SHOP:
            break

    return {
        "tags": tags_seen,
        "bosses": bosses_seen,
        "tag_ante": tag_ante,
        "tag_pos": tag_pos,
        "boss_ante": boss_ante,
    }


def find_minimal_seeds(
    max_seeds: int = 5000,
    max_ante: int = 8,
) -> list[dict[str, Any]]:
    """Greedy set-cover: find seeds that collectively cover all tags + bosses."""

    uncovered_tags = set(ALL_TAGS)
    uncovered_bosses = set(ALL_BOSSES)

    # Phase 1: scan all seeds
    print(f"Scanning {max_seeds} seeds (playing to ante {max_ante})...")
    t0 = time.time()

    seed_data: list[dict[str, Any]] = []
    for i in range(max_seeds):
        seed = f"FIND_{i}"
        result = scan_seed(seed, max_ante)
        result["seed"] = seed
        seed_data.append(result)

        # Progress
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1}/{max_seeds} ({rate:.0f} seeds/sec)")

    elapsed = time.time() - t0
    print(f"  Scanned {max_seeds} seeds in {elapsed:.1f}s ({max_seeds / elapsed:.0f} seeds/sec)")

    # Phase 2: greedy set cover
    selected: list[dict[str, Any]] = []

    while uncovered_tags or uncovered_bosses:
        best_seed = None
        best_score = 0
        best_new_tags: set[str] = set()
        best_new_bosses: set[str] = set()

        for sd in seed_data:
            new_tags = sd["tags"] & uncovered_tags
            new_bosses = sd["bosses"] & uncovered_bosses
            score = len(new_tags) + len(new_bosses)
            if score > best_score:
                best_score = score
                best_seed = sd
                best_new_tags = new_tags
                best_new_bosses = new_bosses

        if best_seed is None or best_score == 0:
            break

        uncovered_tags -= best_new_tags
        uncovered_bosses -= best_new_bosses
        selected.append(
            {
                "seed": best_seed["seed"],
                "new_tags": sorted(best_new_tags),
                "new_bosses": sorted(best_new_bosses),
                "tag_ante": {k: best_seed["tag_ante"][k] for k in best_new_tags},
                "tag_pos": {k: best_seed["tag_pos"][k] for k in best_new_tags},
                "boss_ante": {k: best_seed["boss_ante"][k] for k in best_new_bosses},
                "max_ante_needed": max(
                    max((best_seed["tag_ante"].get(t, 1) for t in best_new_tags), default=1),
                    max((best_seed["boss_ante"].get(b, 1) for b in best_new_bosses), default=1),
                ),
            }
        )

        seed_data.remove(best_seed)

    covered_tags = ALL_TAGS - uncovered_tags
    covered_bosses = ALL_BOSSES - uncovered_bosses
    print(
        f"\nCoverage: {len(covered_tags)}/{len(ALL_TAGS)} tags, "
        f"{len(covered_bosses)}/{len(ALL_BOSSES)} bosses"
    )

    if uncovered_tags:
        print(f"  Missing tags: {sorted(uncovered_tags)}")
    if uncovered_bosses:
        print(f"  Missing bosses: {sorted(uncovered_bosses)}")

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Find seeds covering all tags and boss blinds")
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=5000,
        help="Seeds to scan (default: 5000)",
    )
    parser.add_argument(
        "--max-ante",
        type=int,
        default=8,
        help="Max ante to simulate (default: 8)",
    )
    args = parser.parse_args()

    selected = find_minimal_seeds(
        max_seeds=args.max_seeds,
        max_ante=args.max_ante,
    )

    print(f"\n{'=' * 70}")
    print(f"MINIMAL SEED SET ({len(selected)} seeds)")
    print(f"{'=' * 70}")

    for i, s in enumerate(selected, 1):
        print(f"\n--- Seed {i}: {s['seed']} (play to ante {s['max_ante_needed']}) ---")
        if s["new_tags"]:
            print(f"  Tags ({len(s['new_tags'])}):")
            for t in s["new_tags"]:
                print(f"    {t} (ante {s['tag_ante'][t]}, {s['tag_pos'][t]})")
        if s["new_bosses"]:
            print(f"  Bosses ({len(s['new_bosses'])}):")
            for b in s["new_bosses"]:
                print(f"    {b} (ante {s['boss_ante'][b]})")

    # Python dict literal for boss_blinds.py
    print(f"\n{'=' * 70}")
    print("_BOSS_SEEDS for boss_blinds.py")
    print(f"{'=' * 70}")
    print("_BOSS_SEEDS: dict[str, tuple[str, int]] = {")
    for s in selected:
        if s["new_bosses"]:
            print(f"    # Seed {s['seed']}")
            for b in s["new_bosses"]:
                print(f'    "{b}": ("{s["seed"]}", {s["boss_ante"][b]}),')
    print("}")


if __name__ == "__main__":
    main()
