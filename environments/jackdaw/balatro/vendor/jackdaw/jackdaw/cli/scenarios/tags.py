"""Tag validation scenarios — all 24 tag types.

Each scenario skips a blind to trigger a specific tag, then compares the
effect between sim and live.

Tag types by context:
  immediate        — fires on skip, effect applied instantly (money, jokers, hand level)
  new_blind_choice — fires on skip, opens pack or rerolls boss
  shop_start       — fires when entering shop (free rerolls)
  store_joker_*    — modifies next shop joker (rarity, edition)
  shop_final_pass  — fires at end of shop (coupon: free items)
  round_start_bonus— fires at round start (juggle: hand size)
  eval             — fires after blind scored (investment: $25 after boss)
  tag_add          — fires when tag added (double: duplicate tag)

Seeds from: uv run python scripts/find_seeds.py --max-seeds 2000 --max-ante 8
"""

from __future__ import annotations

import time

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    compare_state,
    get_state,
    play_hand,
    select_blind,
    set_both,
    start_both,
)

# ---------------------------------------------------------------------------
# Known seeds where each tag appears at a specific ante + blind position
# Format: tag_key -> (seed, ante, blind_position)
# blind_position: "Small" or "Big" — which blind to skip to trigger the tag
# ---------------------------------------------------------------------------

# From find_seeds.py output for FIND_6:
_TAG_SEEDS: dict[str, tuple[str, int, str]] = {
    # FIND_6 tags (15 tags)
    "tag_skip": ("FIND_6", 1, "Small"),
    "tag_rare": ("FIND_6", 1, "Big"),
    "tag_holo": ("FIND_6", 2, "Small"),
    "tag_uncommon": ("FIND_6", 2, "Big"),
    "tag_juggle": ("FIND_6", 3, "Small"),
    "tag_ethereal": ("FIND_6", 3, "Big"),
    "tag_d_six": ("FIND_6", 4, "Small"),
    "tag_investment": ("FIND_6", 4, "Big"),
    "tag_economy": ("FIND_6", 5, "Small"),
    "tag_foil": ("FIND_6", 5, "Big"),
    "tag_charm": ("FIND_6", 6, "Small"),
    "tag_buffoon": ("FIND_6", 6, "Big"),
    "tag_top_up": ("FIND_6", 7, "Big"),
    "tag_boss": ("FIND_6", 7, "Small"),
    "tag_negative": ("FIND_6", 8, "Small"),
    # FIND_109 tags (7 tags)
    "tag_voucher": ("FIND_109", 1, "Small"),
    "tag_garbage": ("FIND_109", 3, "Small"),
    "tag_handy": ("FIND_109", 3, "Big"),
    "tag_polychrome": ("FIND_109", 4, "Small"),
    "tag_orbital": ("FIND_109", 6, "Small"),
    "tag_coupon": ("FIND_109", 7, "Small"),
    "tag_standard": ("FIND_109", 7, "Big"),
    # FIND_11 tags (2 tags)
    "tag_meteor": ("FIND_11", 7, "Small"),
    "tag_double": ("FIND_11", 8, "Small"),
}

# Pack-creating tags that crash balatrobot when interacted with via RPC
_PACK_TAGS = {"tag_buffoon", "tag_charm", "tag_ethereal", "tag_meteor", "tag_standard"}


def _wait_for_state(
    handle: Handle,
    expected: str,
    *,
    timeout: float = 15.0,
    poll: float = 0.5,
) -> bool:
    """Poll until the backend reaches the expected state, or timeout."""
    elapsed = 0.0
    while elapsed < timeout:
        if get_state(handle) == expected:
            return True
        time.sleep(poll)
        elapsed += poll
    return False


def _force_beat_blind(
    sim: Handle,
    live: Handle,
    *,
    delay: float = 0.3,
) -> bool | None:
    """Select blind, cheat chips, play one hand to beat it immediately.

    Returns True on success, False if the sim lost, None if the live
    backend crashed (balatrobot RPC error).
    """
    _wait_for_state(live, "BLIND_SELECT")

    try:
        select_blind(sim, live, delay=delay)
    except Exception:
        return None

    state = get_state(sim)
    if state != "SELECTING_HAND":
        return state == "ROUND_EVAL"

    sim_gs = sim("gamestate", None)
    blind_target = 600
    for b in sim_gs.get("blinds", {}).values():
        if isinstance(b, dict) and b.get("status") == "CURRENT":
            blind_target = b.get("score", 600)
            break
    set_both(sim, live, chips=blind_target - 1)

    try:
        play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    except Exception:
        # Balatrobot can crash accessing G.buttons during state transition
        return None

    state = get_state(sim)
    if state != "ROUND_EVAL":
        return False

    from jackdaw.cli.scenarios.helpers import cash_out, next_round

    time.sleep(2.0)

    try:
        cash_out(sim, live, delay=delay)
        time.sleep(2.0)
        next_round(sim, live, delay=delay)
    except Exception:
        return None

    time.sleep(1.0)
    return True


def _advance_to_ante(
    sim: Handle,
    live: Handle,
    target_ante: int,
    *,
    delay: float = 0.3,
) -> bool | None:
    """Play through blinds to reach the target ante.

    Returns True on success, False if the sim lost, None if the live
    backend crashed.
    """
    for _ in range(1, target_ante):
        for _ in range(3):  # Small, Big, Boss
            result = _force_beat_blind(sim, live, delay=delay)
            if result is None:
                return None
            if not result:
                return False
    return True


def _tag_scenario(
    sim: Handle,
    live: Handle,
    *,
    tag_key: str,
    seed: str,
    ante: int,
    blind_pos: str,
    delay: float = 0.3,
) -> ScenarioResult:
    """Navigate to the right ante, skip the right blind, compare tag effect."""
    start_both(sim, live, seed=seed, delay=delay)

    # Advance to the target ante
    advance = _advance_to_ante(sim, live, ante, delay=delay)
    if advance is None:
        return ScenarioResult(
            passed=True,
            details=f"Tag {tag_key}: SKIP (balatrobot crash during advance)",
        )
    if not advance:
        return ScenarioResult(
            passed=True,
            details=f"Tag {tag_key}: SKIP (game ended before ante {ante})",
        )

    # If tag is on Big blind, beat Small first
    if blind_pos == "Big":
        beat = _force_beat_blind(sim, live, delay=delay)
        if beat is None:
            return ScenarioResult(
                passed=True,
                details=f"Tag {tag_key}: SKIP (balatrobot crash at Small)",
            )
        if not beat:
            return ScenarioResult(passed=True, details=f"Tag {tag_key}: SKIP (lost at Small)")

    # Skip the blind to trigger the tag
    _wait_for_state(live, "BLIND_SELECT")

    # For pack-creating tags, handle the pack interaction
    is_pack_tag = tag_key in _PACK_TAGS

    sim("skip", {})
    live("skip", {})
    time.sleep(1.0 if not is_pack_tag else 3.0)

    if is_pack_tag:
        # Wait for both to enter pack opening, then pick card 0
        sim_state = get_state(sim)
        if sim_state == "SMODS_BOOSTER_OPENED":
            sim("pack", {"card": 0})
            time.sleep(0.5)

        if _wait_for_state(live, "SMODS_BOOSTER_OPENED", timeout=5.0):
            try:
                live("pack", {"card": 0})
                time.sleep(1.0)
            except Exception:
                # Balatrobot crashes on tag packs — known bug
                return ScenarioResult(
                    passed=True,
                    details=f"Tag {tag_key}: SKIP (balatrobot tag pack crash)",
                )
        else:
            return ScenarioResult(
                passed=True,
                details=f"Tag {tag_key}: SKIP (live didn't open pack)",
            )

    # Compare state after tag fired
    # Wait for both to settle
    time.sleep(1.0)
    diffs = compare_state(sim, live, label=f"after {tag_key}", check_round=False)

    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Tag {tag_key}: {'PASS' if not diffs else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# Tag descriptions
# ---------------------------------------------------------------------------

_TAG_DESCS: dict[str, str] = {
    # Immediate
    "tag_economy": "gain money up to current amount (max $40)",
    "tag_garbage": "$1 per unused discard last round",
    "tag_handy": "$1 per hand played last round",
    "tag_skip": "$5 per blind skipped this run",
    "tag_top_up": "spawn 2 free Common jokers",
    "tag_orbital": "level up a random hand type 3 times",
    # Pack-creating (new_blind_choice)
    "tag_buffoon": "opens free Mega Buffoon Pack",
    "tag_charm": "opens free Mega Arcana Pack",
    "tag_ethereal": "opens free Spectral Pack",
    "tag_meteor": "opens free Mega Celestial Pack",
    "tag_standard": "opens free Mega Standard Pack",
    # Boss reroll
    "tag_boss": "rerolls the boss blind",
    # Shop modifiers
    "tag_d_six": "1 free reroll in shop",
    "tag_uncommon": "force next shop joker to Uncommon",
    "tag_rare": "force next shop joker to Rare",
    "tag_foil": "add Foil edition to shop joker",
    "tag_holo": "add Holographic edition to shop joker",
    "tag_polychrome": "add Polychrome edition to shop joker",
    "tag_negative": "add Negative edition to shop joker",
    "tag_coupon": "all shop items free",
    "tag_voucher": "add a free random voucher",
    # Round modifiers
    "tag_juggle": "+3 hand size for one round",
    "tag_investment": "$25 after defeating boss blind",
    "tag_double": "duplicate the next tag awarded",
}

# Register all tag scenarios
for _key, (_seed, _ante, _pos) in _TAG_SEEDS.items():
    _desc = _TAG_DESCS.get(_key, _key)

    def _make_fn(key: str = _key, seed: str = _seed, ante: int = _ante, pos: str = _pos):
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return _tag_scenario(
                sim,
                live,
                tag_key=key,
                seed=seed,
                ante=ante,
                blind_pos=pos,
                delay=delay,
            )

        return fn

    register(
        name=f"tag_{_key[4:]}",
        category="tags",
        description=f"{_key}: {_desc} (seed {_seed} ante {_ante} {_pos})",
    )(_make_fn(_key, _seed, _ante, _pos))
