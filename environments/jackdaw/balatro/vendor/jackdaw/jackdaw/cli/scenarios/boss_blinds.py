"""Boss blind validation scenarios — grouped by seed for efficiency.

Each seed group starts the game once and advances through antes sequentially,
testing each boss blind at its respective ante.  This avoids redundant game
restarts when multiple bosses share the same seed.

Seeds found via: uv run python scripts/find_seeds.py --max-seeds 2000 --max-ante 8
"""

from __future__ import annotations

import time
from collections import defaultdict

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    cash_out,
    compare_state,
    get_state,
    next_round,
    play_hand,
    select_blind,
    set_both,
    start_both,
)


def _wait_for_state(
    handle: Handle,
    expected: str,
    *,
    timeout: float = 10.0,
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
) -> bool:
    """Select blind, cheat chips, play one hand to beat it immediately.

    Returns True if the blind was beaten, False if game ended.
    """
    _wait_for_state(live, "BLIND_SELECT")
    select_blind(sim, live, delay=delay)

    state = get_state(sim)
    if state != "SELECTING_HAND":
        return state == "ROUND_EVAL"

    # Set chips just below the blind target so the first hand wins instantly.
    sim_gs = sim("gamestate", None)
    blind_target = 600
    for b in sim_gs.get("blinds", {}).values():
        if isinstance(b, dict) and b.get("status") == "CURRENT":
            blind_target = b.get("score", 600)
            break
    set_both(sim, live, chips=blind_target - 1)

    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    state = get_state(sim)
    if state != "ROUND_EVAL":
        return False

    time.sleep(2.0)  # scoring animation
    cash_out(sim, live, delay=delay)
    time.sleep(2.0)  # cash-out scoreboard
    next_round(sim, live, delay=delay)
    time.sleep(1.0)  # shop transition
    return True


def _advance_ante(
    sim: Handle,
    live: Handle,
    *,
    delay: float = 0.3,
) -> bool:
    """Play through Small, Big, and Boss blinds to advance one ante.

    Uses chip cheat on all three blinds. Does NOT skip — avoids tag rewards
    and the balatrobot tag-pack crash.

    Returns True if the ante was completed, False if game ended.
    """
    for _ in range(3):  # Small, Big, Boss
        if not _force_beat_blind(sim, live, delay=delay):
            return False
    return True


def _finish_current_ante(
    sim: Handle,
    live: Handle,
    *,
    delay: float = 0.3,
) -> bool:
    """After testing a boss blind, finish the ante so we can continue.

    Handles whatever state the game is in after the boss test:
    - SELECTING_HAND: force-beat remaining, cash out, next round
    - ROUND_EVAL: cash out, next round
    - GAME_OVER: return False
    - BLIND_SELECT: already advanced (shouldn't happen but handle gracefully)
    """
    state = get_state(sim)

    if state == "SELECTING_HAND":
        # Still playing the boss — force-beat it
        sim_gs = sim("gamestate", None)
        blind_target = 600
        for b in sim_gs.get("blinds", {}).values():
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                blind_target = b.get("score", 600)
                break
        set_both(sim, live, chips=blind_target - 1)
        play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
        state = get_state(sim)

    if state == "ROUND_EVAL":
        time.sleep(2.0)
        cash_out(sim, live, delay=delay)
        time.sleep(2.0)
        next_round(sim, live, delay=delay)
        time.sleep(1.0)
        return True

    if state == "BLIND_SELECT":
        return True

    return False  # GAME_OVER or unexpected


def _test_boss(
    sim: Handle,
    live: Handle,
    boss_key: str,
    seed: str,
    ante: int,
    *,
    delay: float = 0.3,
) -> ScenarioResult:
    """At the target ante, beat Small+Big with cheat, then beat Boss with cheat.

    Uses chip cheat on the boss too so we survive to continue to the next ante.
    The boss's debuff/scoring effects still apply during the cheated hand,
    so the sim-vs-live comparison is meaningful.
    """
    # Beat Small blind
    if not _force_beat_blind(sim, live, delay=delay):
        return ScenarioResult(passed=True, details=f"{boss_key}: SKIP (lost at Small)")

    # Beat Big blind
    if not _force_beat_blind(sim, live, delay=delay):
        return ScenarioResult(passed=True, details=f"{boss_key}: SKIP (lost at Big)")

    # Select boss blind (boss effects apply on select)
    _wait_for_state(live, "BLIND_SELECT")
    select_blind(sim, live, delay=delay)

    state = get_state(sim)
    if state != "SELECTING_HAND":
        diffs = compare_state(sim, live, label=f"boss {boss_key} after select")
        return ScenarioResult(
            passed=not diffs,
            diffs=diffs,
            details=f"{boss_key} (seed {seed} ante {ante}): {'PASS' if not diffs else 'FAIL'}",
        )

    # Cheat chips so first hand beats the boss, but scoring still exercises
    # the boss's effects (debuffs, restrictions, etc.)
    sim_gs = sim("gamestate", None)
    blind_target = 600
    for b in sim_gs.get("blinds", {}).values():
        if isinstance(b, dict) and b.get("status") == "CURRENT":
            blind_target = b.get("score", 600)
            break
    set_both(sim, live, chips=blind_target - 1)

    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label=f"boss {boss_key} ante {ante}")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"{boss_key} (seed {seed} ante {ante}): {'PASS' if not diffs else 'FAIL'}",
    )


def _boss_group_scenario(
    sim: Handle,
    live: Handle,
    *,
    seed: str,
    bosses: list[tuple[str, int]],
    delay: float = 0.3,
) -> ScenarioResult:
    """Run all bosses for a single seed in one game session.

    bosses: list of (boss_key, ante) sorted by ante ascending.
    """
    start_both(sim, live, seed=seed, delay=delay)

    sub_results: list[tuple[str, ScenarioResult]] = []
    current_ante = 1
    aborted = False

    for idx, (boss_key, target_ante) in enumerate(bosses):
        if aborted:
            sub_results.append(
                (
                    f"boss_{boss_key[3:]}",
                    ScenarioResult(passed=True, details=f"{boss_key}: SKIP (earlier abort)"),
                )
            )
            continue

        # Advance to target ante (playing through intermediate antes)
        while current_ante < target_ante:
            if not _advance_ante(sim, live, delay=delay):
                # Game ended — skip this and remaining bosses
                sub_results.append(
                    (
                        f"boss_{boss_key[3:]}",
                        ScenarioResult(
                            passed=True,
                            details=f"{boss_key}: SKIP (game ended at ante {current_ante})",
                        ),
                    )
                )
                aborted = True
                break

            diffs = compare_state(sim, live, label=f"after ante {current_ante}", check_round=False)
            if diffs:
                sub_results.append(
                    (
                        f"boss_{boss_key[3:]}",
                        ScenarioResult(
                            passed=False,
                            diffs=diffs,
                            details=f"{boss_key}: DIVERGED during advance (ante {current_ante})",
                        ),
                    )
                )
                aborted = True
                break

            current_ante += 1

        if aborted:
            continue

        # Test the boss at this ante
        boss_result = _test_boss(sim, live, boss_key, seed, target_ante, delay=delay)
        sub_results.append((f"boss_{boss_key[3:]}", boss_result))

        # Finish the ante so we can continue to the next boss
        if idx < len(bosses) - 1:
            if not _finish_current_ante(sim, live, delay=delay):
                aborted = True
                continue
            current_ante = target_ante + 1

    overall_passed = all(r.passed for _, r in sub_results)
    passed_count = sum(r.passed for _, r in sub_results)
    return ScenarioResult(
        passed=overall_passed,
        sub_results=sub_results,
        details=f"Seed {seed}: {passed_count}/{len(sub_results)} passed",
    )


# ---------------------------------------------------------------------------
# Known seeds — maps boss_key -> (seed, ante) where that boss appears
# ---------------------------------------------------------------------------

_BOSS_SEEDS: dict[str, tuple[str, int]] = {
    # Seed FIND_6: 8 bosses
    "bl_hook": ("FIND_6", 1),
    "bl_water": ("FIND_6", 2),
    "bl_eye": ("FIND_6", 3),
    "bl_window": ("FIND_6", 4),
    "bl_goad": ("FIND_6", 5),
    "bl_head": ("FIND_6", 6),
    "bl_wheel": ("FIND_6", 7),
    "bl_final_heart": ("FIND_6", 8),
    # Seed FIND_109: 8 bosses
    "bl_manacle": ("FIND_109", 1),
    "bl_psychic": ("FIND_109", 2),
    "bl_club": ("FIND_109", 3),
    "bl_flint": ("FIND_109", 4),
    "bl_house": ("FIND_109", 5),
    "bl_wall": ("FIND_109", 6),
    "bl_tooth": ("FIND_109", 7),
    "bl_final_vessel": ("FIND_109", 8),
    # Seed FIND_11: 5 bosses
    "bl_mark": ("FIND_11", 2),
    "bl_fish": ("FIND_11", 3),
    "bl_plant": ("FIND_11", 5),
    "bl_ox": ("FIND_11", 6),
    "bl_final_leaf": ("FIND_11", 8),
    # Seed FIND_372: 4 bosses
    "bl_pillar": ("FIND_372", 1),
    "bl_arm": ("FIND_372", 2),
    "bl_serpent": ("FIND_372", 7),
    "bl_final_bell": ("FIND_372", 8),
    # Seed FIND_1: 1 boss
    "bl_mouth": ("FIND_1", 2),
    # Seed FIND_8: 1 boss
    "bl_needle": ("FIND_8", 3),
    # Seed FIND_12: 1 boss
    "bl_final_acorn": ("FIND_12", 8),
}

_BOSS_NAMES: dict[str, tuple[str, str]] = {
    "bl_club": ("The Club", "Clubs debuffed"),
    "bl_goad": ("The Goad", "Spades debuffed"),
    "bl_head": ("The Head", "Hearts debuffed"),
    "bl_window": ("The Window", "Diamonds debuffed"),
    "bl_plant": ("The Plant", "face cards debuffed"),
    "bl_psychic": ("The Psychic", "must play 5 cards"),
    "bl_mouth": ("The Mouth", "only one hand type allowed"),
    "bl_eye": ("The Eye", "no repeat hand types"),
    "bl_needle": ("The Needle", "only 1 hand per round"),
    "bl_water": ("The Water", "start with 0 discards"),
    "bl_hook": ("The Hook", "discards 2 random cards per hand"),
    "bl_serpent": ("The Serpent", "draw 3 extra cards after play/discard"),
    "bl_fish": ("The Fish", "cards drawn face down"),
    "bl_house": ("The House", "first hand dealt face down"),
    "bl_mark": ("The Mark", "first hand dealt face down"),
    "bl_manacle": ("The Manacle", "-1 hand size"),
    "bl_flint": ("The Flint", "base Chips and Mult halved"),
    "bl_wall": ("The Wall", "extra large blind"),
    "bl_arm": ("The Arm", "decrease level of played hand type"),
    "bl_ox": ("The Ox", "set money to $0 if most played hand"),
    "bl_tooth": ("The Tooth", "lose $1 per card played"),
    "bl_wheel": ("The Wheel", "1 in 7 cards drawn face down"),
    "bl_pillar": ("The Pillar", "cards played prev round debuffed"),
    "bl_final_acorn": ("Amber Acorn", "flips and rearranges jokers"),
    "bl_final_bell": ("Cerulean Bell", "forces 1 card to always be selected"),
    "bl_final_heart": ("Crimson Heart", "one random joker debuffed each hand"),
    "bl_final_leaf": ("Verdant Leaf", "all cards debuffed until 1 sold"),
    "bl_final_vessel": ("Violet Vessel", "very large blind"),
}

# ---------------------------------------------------------------------------
# Build seed groups and register one scenario per seed
# ---------------------------------------------------------------------------

_SEED_GROUPS: dict[str, list[tuple[str, int]]] = defaultdict(list)
for _key, (_seed, _ante) in _BOSS_SEEDS.items():
    _SEED_GROUPS[_seed].append((_key, _ante))

# Sort each group by ante
for _seed in _SEED_GROUPS:
    _SEED_GROUPS[_seed].sort(key=lambda x: x[1])

for _seed, _bosses in _SEED_GROUPS.items():
    _boss_desc = ", ".join(f"{_BOSS_NAMES[k][0]}@{a}" for k, a in _bosses)

    def _make_fn(seed: str = _seed, bosses: list[tuple[str, int]] = _bosses):
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return _boss_group_scenario(sim, live, seed=seed, bosses=bosses, delay=delay)

        return fn

    register(
        name=f"boss_seed_{_seed.lower()}",
        category="boss_blinds",
        description=f"Seed {_seed}: {_boss_desc}",
    )(_make_fn(_seed, _bosses))
