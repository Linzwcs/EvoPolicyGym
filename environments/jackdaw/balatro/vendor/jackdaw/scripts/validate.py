#!/usr/bin/env python3
"""Unified validation tool for the jackdaw Balatro simulator.

Subcommands::

    uv run scripts/validate.py seed --seed TESTSEED --back b_red --stake 1
    uv run scripts/validate.py crash --count 200 --agent random
    uv run scripts/validate.py live --host 127.0.0.1 --port 12346
    uv run scripts/validate.py benchmark --count 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

sys.path.insert(0, ".")

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

DECK_MAP = {
    "b_red": "RED",
    "b_blue": "BLUE",
    "b_yellow": "YELLOW",
    "b_green": "GREEN",
    "b_black": "BLACK",
    "b_magic": "MAGIC",
    "b_nebula": "NEBULA",
    "b_ghost": "GHOST",
    "b_abandoned": "ABANDONED",
    "b_checkered": "CHECKERED",
    "b_zodiac": "ZODIAC",
    "b_painted": "PAINTED",
    "b_anaglyph": "ANAGLYPH",
    "b_plasma": "PLASMA",
    "b_erratic": "ERRATIC",
}

STAKE_MAP = {
    1: "WHITE",
    2: "RED",
    3: "GREEN",
    4: "BLACK",
    5: "BLUE",
    6: "PURPLE",
    7: "ORANGE",
    8: "GOLD",
}

TAG_NAME_TO_KEY = {
    "Economy Tag": "tag_economy",
    "Investment Tag": "tag_investment",
    "Double Tag": "tag_double",
    "Skip Tag": "tag_skip",
    "Garbage Tag": "tag_garbage",
    "Handy Tag": "tag_handy",
    "Orbital Tag": "tag_orbital",
    "Top-up Tag": "tag_top_up",
    "Boss Tag": "tag_boss",
    "Buffoon Tag": "tag_buffoon",
    "Charm Tag": "tag_charm",
    "Ethereal Tag": "tag_ethereal",
    "Meteor Tag": "tag_meteor",
    "Standard Tag": "tag_standard",
    "Juggle Tag": "tag_juggle",
    "D6 Tag": "tag_d_six",
    "Uncommon Tag": "tag_uncommon",
    "Rare Tag": "tag_rare",
    "Foil Tag": "tag_foil",
    "Holographic Tag": "tag_holo",
    "Polychrome Tag": "tag_polychrome",
    "Negative Tag": "tag_negative",
    "Voucher Tag": "tag_voucher",
    "Coupon Tag": "tag_coupon",
}

BOSS_NAME_TO_KEY = {
    "The Hook": "bl_hook",
    "The Club": "bl_club",
    "The Goad": "bl_goad",
    "The Head": "bl_head",
    "The Window": "bl_window",
    "The Wall": "bl_wall",
    "The Wheel": "bl_wheel",
    "The Arm": "bl_arm",
    "The Eye": "bl_eye",
    "The Mouth": "bl_mouth",
    "The Plant": "bl_plant",
    "The Serpent": "bl_serpent",
    "The Pillar": "bl_pillar",
    "The Needle": "bl_needle",
    "The Water": "bl_water",
    "The Manacle": "bl_manacle",
    "The Tooth": "bl_tooth",
    "The Fish": "bl_fish",
    "The Mark": "bl_mark",
    "The Flint": "bl_flint",
    "Amber Acorn": "bl_final_acorn",
    "Verdant Leaf": "bl_final_leaf",
    "Violet Vessel": "bl_final_vessel",
    "Crimson Heart": "bl_final_heart",
    "Cerulean Bell": "bl_final_bell",
    "The Ox": "bl_ox",
    "The House": "bl_house",
    "Psychic": "bl_psychic",
}

# ---------------------------------------------------------------------------
# RPC helpers (shared by seed and live subcommands)
# ---------------------------------------------------------------------------

_bot_url: str = "http://127.0.0.1:12346"


def rpc(method: str, params: dict | None = None) -> dict:
    """Send a JSON-RPC request to balatrobot."""
    import httpx

    payload: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params:
        payload["params"] = params
    resp = httpx.post(_bot_url, json=payload, timeout=10.0)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC {method}: {data['error']}")
    return data.get("result", {})


def require_balatrobot() -> None:
    """Check that balatrobot is reachable, exit if not."""
    try:
        rpc("health")
    except Exception as e:
        print(f"Cannot reach balatrobot at {_bot_url}: {e}")
        print("Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>")
        sys.exit(1)


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------


def bot_hand_keys(bot: dict) -> list[str]:
    return [c["key"] for c in bot.get("hand", {}).get("cards", [])]


def bot_deck_keys(bot: dict) -> list[str]:
    return [c["key"] for c in bot.get("cards", {}).get("cards", [])]


def sim_hand_keys(gs: dict) -> list[str]:
    return [c.card_key for c in gs.get("hand", []) if hasattr(c, "card_key") and c.card_key]


def sim_deck_keys(gs: dict) -> list[str]:
    return [c.card_key for c in gs.get("deck", []) if hasattr(c, "card_key") and c.card_key]


def compare_states(sim: dict, bot: dict, label: str) -> list[str]:
    """Compare sim game state to balatrobot state, return list of diffs."""
    diffs: list[str] = []

    def cmp(name: str, s: Any, live: Any) -> None:
        if s != live:
            diffs.append(f"{name}: sim={s} live={live}")

    cmp("money", sim.get("dollars", 0), bot.get("money", 0))
    cmp("ante", sim["round_resets"]["ante"], bot.get("ante_num", 1))

    cr = sim.get("current_round", {})
    br = bot.get("round", {})
    cmp("chips", sim.get("chips", 0), br.get("chips", 0))

    # Only compare hands/discards when in a round
    sim_phase = sim.get("phase")
    if sim_phase not in (GamePhase.BLIND_SELECT, None):
        cmp("hands_left", cr.get("hands_left", 0), br.get("hands_left", 0))
        cmp("discards_left", cr.get("discards_left", 0), br.get("discards_left", 0))

    # Hand cards (as sets — display order may differ)
    sh = set(sim_hand_keys(sim))
    lh = set(bot_hand_keys(bot))
    cmp("hand_cards", sh, lh)

    # Deck size
    cmp("deck_size", len(sim.get("deck", [])), bot.get("cards", {}).get("count", 0))

    status = "OK" if not diffs else f"{len(diffs)} diffs"
    print(f"  [{label}] {status}")
    for d in diffs:
        print(f"    {d}")
    return diffs


def init_sim(seed: str, back_key: str = "b_red", stake: int = 1) -> dict:
    """Initialize a simulator run ready for blind select."""
    gs = initialize_run(back_key, stake, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs["jokers"] = []
    gs["consumables"] = []
    return gs


def start_bot_run(seed: str, back_key: str = "b_red", stake: int = 1) -> None:
    """Start a run in balatrobot."""
    rpc("menu")
    time.sleep(1.5)
    deck_name = DECK_MAP.get(back_key, "RED")
    stake_name = STAKE_MAP.get(stake, "WHITE")
    rpc("start", {"deck": deck_name, "stake": stake_name, "seed": seed})
    time.sleep(0.5)


# ===================================================================
# Subcommand: seed
# ===================================================================


def _seed_run_validation(seed: str, back_key: str, stake: int) -> dict[str, Any]:
    """Play through Small->Big->Boss with identical actions in both."""
    print(f"\n{'=' * 60}")
    print(f"Seed: {seed}  back={back_key}  stake={stake}")
    print(f"{'=' * 60}")

    start_bot_run(seed, back_key, stake)
    sim = init_sim(seed, back_key, stake)

    all_diffs: list[list[str]] = []
    step_count = 0

    # Compare initial state
    bot = rpc("gamestate")
    all_diffs.append(compare_states(sim, bot, "init"))

    # --- Also compare init-only fields (boss, tags, deck order) ---
    _compare_init_fields(sim, bot)

    blinds = ["Small", "Big", "Boss"]
    for blind_name in blinds:
        # Select blind
        print(f"\n--- Select {blind_name} Blind ---")
        step(sim, SelectBlind())
        rpc("select")
        time.sleep(0.3)
        bot = rpc("gamestate")
        all_diffs.append(compare_states(sim, bot, f"after_select_{blind_name.lower()}"))
        step_count += 1

        # Play hands until blind beaten or game over
        hand_num = 0
        while sim.get("phase") == GamePhase.SELECTING_HAND:
            hand = sim.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break

            # Get balatrobot's hand order and play the same cards
            bot_now = rpc("gamestate")
            bot_hand = bot_hand_keys(bot_now)
            bot_play_keys = bot_hand[: min(5, len(bot_hand))]

            # Find those same card keys in our sim hand
            sim_indices = []
            used: set[int] = set()
            for bk in bot_play_keys:
                for i, c in enumerate(hand):
                    if i not in used and hasattr(c, "card_key") and c.card_key == bk:
                        sim_indices.append(i)
                        used.add(i)
                        break

            if len(sim_indices) != len(bot_play_keys):
                sim_indices = list(range(n))
                print(f"  Play hand {hand_num}: FALLBACK (card mismatch)")
            else:
                print(f"  Play hand {hand_num}: {bot_play_keys}")

            bot_indices = list(range(len(bot_play_keys)))
            step(sim, PlayHand(card_indices=tuple(sim_indices)))
            rpc("play", {"cards": bot_indices})
            time.sleep(0.3)
            bot = rpc("gamestate")
            all_diffs.append(
                compare_states(sim, bot, f"after_play_{blind_name.lower()}_{hand_num}")
            )
            step_count += 1
            hand_num += 1

            if sim.get("phase") == GamePhase.GAME_OVER:
                print("  GAME OVER")
                return _seed_summary(seed, all_diffs, step_count, "game_over")

        if sim.get("phase") != GamePhase.ROUND_EVAL:
            print(f"  Unexpected phase: {sim.get('phase')}")
            return _seed_summary(seed, all_diffs, step_count, "unexpected_phase")

        # Cash out
        print("  Cash out")
        step(sim, CashOut())
        rpc("cash_out")
        time.sleep(0.3)
        bot = rpc("gamestate")
        all_diffs.append(compare_states(sim, bot, f"after_cashout_{blind_name.lower()}"))
        step_count += 1

        # Next round (except after Boss)
        if blind_name != "Boss":
            print("  Next round")
            step(sim, NextRound())
            rpc("next_round")
            time.sleep(0.3)
            bot = rpc("gamestate")
            all_diffs.append(compare_states(sim, bot, f"after_nextround_{blind_name.lower()}"))
            step_count += 1

    return _seed_summary(seed, all_diffs, step_count, "complete")


def _compare_init_fields(sim: dict, bot: dict) -> None:
    """Print extended init comparison (boss, tags, deck order)."""
    # Boss
    sim_boss = sim["round_resets"]["blind_choices"]["Boss"]
    live_boss_name = bot.get("blinds", {}).get("boss", {}).get("name", "")
    live_boss_key = BOSS_NAME_TO_KEY.get(live_boss_name, f"?{live_boss_name}")
    status = "MATCH" if sim_boss == live_boss_key else "DIFF"
    print(f"  boss:       sim={sim_boss:<16}  live={live_boss_key:<16}  [{status}]")

    # Tags
    rr = sim["round_resets"]
    sim_small_tag = rr.get("blind_tags", {}).get("Small", "?")
    sim_big_tag = rr.get("blind_tags", {}).get("Big", "?")
    live_blinds = bot.get("blinds", {})
    live_small_name = live_blinds.get("small", {}).get("tag_name", "")
    live_big_name = live_blinds.get("big", {}).get("tag_name", "")
    live_small_tag = TAG_NAME_TO_KEY.get(live_small_name, f"?{live_small_name}")
    live_big_tag = TAG_NAME_TO_KEY.get(live_big_name, f"?{live_big_name}")

    status = "MATCH" if sim_small_tag == live_small_tag else "DIFF"
    print(f"  small_tag:  sim={sim_small_tag:<20}  live={live_small_tag:<20}  [{status}]")
    status = "MATCH" if sim_big_tag == live_big_tag else "DIFF"
    print(f"  big_tag:    sim={sim_big_tag:<20}  live={live_big_tag:<20}  [{status}]")

    # Deck order (first 10)
    sim_dk = [c.card_key for c in sim["deck"][:10]]
    live_cards = bot.get("cards", {}).get("cards", [])
    live_dk = [c["key"] for c in live_cards[:10]]
    print("  Deck order (first 10):")
    for i in range(min(10, max(len(sim_dk), len(live_dk)))):
        sk = sim_dk[i] if i < len(sim_dk) else "?"
        lk = live_dk[i] if i < len(live_dk) else "?"
        status = "MATCH" if sk == lk else "DIFF"
        print(f"    [{i}] sim={sk:<6}  live={lk:<6}  [{status}]")


def _seed_summary(seed: str, all_diffs: list, step_count: int, status: str) -> dict:
    total_diffs = sum(len(d) for d in all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    return {
        "seed": seed,
        "steps": step_count,
        "total_diffs": total_diffs,
        "clean_steps": clean,
        "total_steps": len(all_diffs),
        "status": status,
    }


def cmd_seed(args: argparse.Namespace) -> None:
    """Run sim and balatrobot side-by-side, compare at every decision point."""
    require_balatrobot()

    results = []
    for i in range(args.seeds):
        seed = f"{args.seed}{i}" if args.seeds > 1 else args.seed
        result = _seed_run_validation(seed, args.back, args.stake)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["total_diffs"] == 0 else "FAIL"
        print(
            f"  {r['seed']}: {status} ({r['clean_steps']}/{r['total_steps']} clean, "
            f"{r['total_diffs']} diffs, {r['status']})"
        )

    total_clean = sum(r["total_diffs"] == 0 for r in results)
    print(f"\nResult: {total_clean}/{len(results)} seeds fully clean")


# ===================================================================
# Subcommand: crash
# ===================================================================


def cmd_crash(args: argparse.Namespace) -> None:
    """Run N simulated games, report crashes and stats."""
    agent = random_agent
    wins = 0
    crashes = 0
    total_actions = 0
    total_rounds = 0
    max_actions = 0

    print(f"Running {args.count} games with {args.agent} agent...")
    t0 = time.time()
    for i in range(args.count):
        try:
            gs = simulate_run(
                "b_red",
                1,
                f"CRASH_{args.agent}_{i}",
                agent,
                max_actions=2000,
            )
            if gs.get("won"):
                wins += 1
            total_actions += gs["actions_taken"]
            total_rounds += gs.get("round", 0)
            max_actions = max(max_actions, gs["actions_taken"])
        except Exception as e:
            crashes += 1
            print(f"  CRASH at seed {i}: {e}")
    elapsed = time.time() - t0
    runs_per_sec = args.count / max(elapsed, 0.001)

    # Determinism check — use greedy_play_agent (deterministic) so the
    # comparison is not confounded by Python's random module state.
    print("\nDeterminism check...")
    gs1 = simulate_run("b_red", 1, "DETERMINISM_CHECK", greedy_play_agent, max_actions=500)
    gs2 = simulate_run("b_red", 1, "DETERMINISM_CHECK", greedy_play_agent, max_actions=500)
    deterministic = (
        gs1.get("dollars") == gs2.get("dollars")
        and gs1.get("round") == gs2.get("round")
        and gs1.get("won") == gs2.get("won")
    )
    print(f"  Same seed -> same result: {deterministic}")

    # Results
    print(f"\n{'=' * 60}")
    print(f"Runs:        {args.count}")
    print(f"Wins:        {wins}")
    print(f"Crashes:     {crashes}")
    print(f"Avg actions: {total_actions / max(args.count, 1):.1f}")
    print(f"Max actions: {max_actions}")
    print(f"Avg rounds:  {total_rounds / max(args.count, 1):.1f}")
    print(f"Time:        {elapsed:.2f}s ({runs_per_sec:.0f} runs/sec)")

    ok = crashes == 0 and deterministic
    print(
        f"\n{'PASS' if ok else 'FAIL'}: {crashes} crashes, "
        f"determinism={'OK' if deterministic else 'FAIL'}"
    )


# ===================================================================
# Subcommand: live
# ===================================================================


def cmd_live(args: argparse.Namespace) -> None:
    """Connect to a running balatrobot, mirror state, compare in real time."""
    require_balatrobot()

    seed = args.seed

    start_bot_run(seed)
    sim = init_sim(seed)

    all_diffs: list[list[str]] = []

    # Compare initial state
    live = rpc("gamestate")
    all_diffs.append(compare_states(sim, live, "init"))

    # Select Small Blind
    print("\n--- Select Small Blind ---")
    step(sim, SelectBlind())
    rpc("select")
    time.sleep(0.3)
    live = rpc("gamestate")
    all_diffs.append(compare_states(sim, live, "after_select_small"))

    # Play first 5 cards
    print("\n--- Play hand (0-4) ---")
    hand_cards = sim.get("hand", [])
    if len(hand_cards) >= 5:
        step(sim, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        rpc("play", {"cards": [0, 1, 2, 3, 4]})
        time.sleep(0.5)
        live = rpc("gamestate")
        all_diffs.append(compare_states(sim, live, "after_play"))

    # Check if we beat the blind
    if sim.get("phase") == GamePhase.ROUND_EVAL:
        print("\n--- Cash out ---")
        step(sim, CashOut())
        rpc("cash_out")
        time.sleep(0.3)
        live = rpc("gamestate")
        all_diffs.append(compare_states(sim, live, "after_cashout"))

        if sim.get("phase") == GamePhase.SHOP:
            print(
                f"  Shop cards: sim={len(sim.get('shop_cards', []))} "
                f"live={len(live.get('shop', {}).get('cards', []))}"
            )

            print("\n--- Next round ---")
            step(sim, NextRound())
            rpc("next_round")
            time.sleep(0.3)
            live = rpc("gamestate")
            all_diffs.append(compare_states(sim, live, "after_next_round"))

            if sim.get("phase") == GamePhase.BLIND_SELECT:
                print("\n--- Select Big Blind ---")
                step(sim, SelectBlind())
                rpc("select")
                time.sleep(0.3)
                live = rpc("gamestate")
                all_diffs.append(compare_states(sim, live, "after_select_big"))
    elif sim.get("phase") == GamePhase.SELECTING_HAND:
        print("  (blind not beaten, still playing)")
    elif sim.get("phase") == GamePhase.GAME_OVER:
        print("  (game over)")

    # Summary
    print(f"\n{'=' * 60}")
    total_steps = len(all_diffs)
    clean = sum(1 for d in all_diffs if not d)
    total_diffs = sum(len(d) for d in all_diffs)
    print(f"Steps compared: {total_steps}")
    print(f"Clean: {clean}/{total_steps}")
    print(f"Total discrepancies: {total_diffs}")


# ===================================================================
# Subcommand: benchmark
# ===================================================================


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Pure performance measurement."""
    agent = random_agent
    count = args.count

    print(f"Benchmarking {count} runs...")
    t0 = time.time()
    total_actions = 0
    for i in range(count):
        gs = simulate_run("b_red", 1, f"BENCH_{i}", agent, max_actions=2000)
        total_actions += gs["actions_taken"]
    elapsed = time.time() - t0

    runs_per_sec = count / max(elapsed, 0.001)
    actions_per_sec = total_actions / max(elapsed, 0.001)

    print(f"\n{'=' * 60}")
    print(f"Runs:            {count}")
    print(f"Total actions:   {total_actions}")
    print(f"Avg actions/run: {total_actions / count:.1f}")
    print(f"Time:            {elapsed:.2f}s")
    print(f"Runs/sec:        {runs_per_sec:.0f}")
    print(f"Actions/sec:     {actions_per_sec:.0f}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified validation tool for the jackdaw Balatro simulator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- seed --
    p_seed = sub.add_parser(
        "seed",
        help="Run sim and balatrobot side-by-side, compare at every decision point",
    )
    p_seed.add_argument("--seed", default="TESTSEED")
    p_seed.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds to test (appends 0..N-1 to seed)"
    )
    p_seed.add_argument("--back", default="b_red", help="Deck back key")
    p_seed.add_argument("--stake", type=int, default=1, help="Stake level (1-8)")
    p_seed.add_argument("--host", default="127.0.0.1")
    p_seed.add_argument("--port", type=int, default=12346)

    # -- crash --
    p_crash = sub.add_parser(
        "crash",
        help="Run N simulated games, report crashes and stats",
    )
    p_crash.add_argument("--count", type=int, default=200)
    p_crash.add_argument("--agent", default="random", choices=["random"], help="Agent type")

    # -- live --
    p_live = sub.add_parser(
        "live",
        help="Connect to running balatrobot, mirror state, compare in real time",
    )
    p_live.add_argument("--seed", default="LIVETEST")
    p_live.add_argument("--host", default="127.0.0.1")
    p_live.add_argument("--port", type=int, default=12346)

    # -- benchmark --
    p_bench = sub.add_parser(
        "benchmark",
        help="Pure performance measurement",
    )
    p_bench.add_argument("--count", type=int, default=1000)

    args = parser.parse_args()

    # Set bot URL from host/port if provided
    global _bot_url
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 12346)
    _bot_url = f"http://{host}:{port}"

    commands = {
        "seed": cmd_seed,
        "crash": cmd_crash,
        "live": cmd_live,
        "benchmark": cmd_benchmark,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
