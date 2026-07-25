"""Shared helpers for validation scenarios.

Every helper takes ``sim_handle`` and ``live_handle`` callables (the
``Backend.handle`` method) so scenarios don't depend on concrete backend
types.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from jackdaw.cli.scenarios import ScenarioResult

# Type alias for backend handle functions
Handle = Callable[[str, dict[str, Any] | None], dict[str, Any]]

# Default delay between actions (overridden per-run via ``delay`` kwarg)
_DEFAULT_DELAY = 0.3


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def start_both(
    sim: Handle,
    live: Handle,
    *,
    seed: str = "SCENARIO",
    deck: str = "RED",
    stake: str = "WHITE",
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Start a new run on both backends with the same params."""
    live("menu", None)
    time.sleep(delay)
    params = {"deck": deck, "stake": stake, "seed": seed}
    sim("start", params)
    live("start", params)
    time.sleep(delay)


def select_blind(sim: Handle, live: Handle, *, delay: float = _DEFAULT_DELAY) -> None:
    """Select the current blind on both backends."""
    sim("select", {})
    live("select", {})
    time.sleep(delay)


def skip_blind(sim: Handle, live: Handle, *, delay: float = _DEFAULT_DELAY) -> None:
    """Skip the current blind on both backends."""
    sim("skip", {})
    live("skip", {})
    time.sleep(delay)


def add_both(
    sim: Handle,
    live: Handle,
    **params: Any,
) -> None:
    """Call ``add`` on both backends with the same params."""
    sim("add", params)
    live("add", params)


def set_both(
    sim: Handle,
    live: Handle,
    **params: Any,
) -> None:
    """Call ``set`` on both backends with the same params."""
    sim("set", params)
    live("set", params)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def play_hand(
    sim: Handle,
    live: Handle,
    cards: list[int],
    *,
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Play a hand on both backends."""
    params = {"cards": cards}
    sim("play", params)
    live("play", params)
    time.sleep(max(delay, 1.5))  # scoring animation


def discard(
    sim: Handle,
    live: Handle,
    cards: list[int],
    *,
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Discard cards on both backends."""
    params = {"cards": cards}
    sim("discard", params)
    live("discard", params)
    time.sleep(delay)


def use_consumable(
    sim: Handle,
    live: Handle,
    consumable: int,
    cards: list[int] | None = None,
    *,
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Use a consumable on both backends."""
    params: dict[str, Any] = {"consumable": consumable}
    if cards is not None:
        params["cards"] = cards
    sim("use", params)
    live("use", params)
    time.sleep(delay)


def cash_out(sim: Handle, live: Handle, *, delay: float = _DEFAULT_DELAY) -> None:
    """Cash out on both backends."""
    sim("cash_out", {})
    live("cash_out", {})
    time.sleep(max(delay, 1.5))


def next_round(sim: Handle, live: Handle, *, delay: float = _DEFAULT_DELAY) -> None:
    """Advance to next round on both backends."""
    sim("next_round", {})
    live("next_round", {})
    time.sleep(delay)


def sell_joker(sim: Handle, live: Handle, index: int, *, delay: float = _DEFAULT_DELAY) -> None:
    """Sell a joker on both backends."""
    sim("sell", {"joker": index})
    live("sell", {"joker": index})
    time.sleep(delay)


def sell_consumable(
    sim: Handle, live: Handle, index: int, *, delay: float = _DEFAULT_DELAY
) -> None:
    """Sell a consumable on both backends."""
    sim("sell", {"consumable": index})
    live("sell", {"consumable": index})
    time.sleep(delay)


def buy_card(sim: Handle, live: Handle, index: int, *, delay: float = _DEFAULT_DELAY) -> None:
    """Buy a shop card on both backends."""
    sim("buy", {"card": index})
    live("buy", {"card": index})
    time.sleep(delay)


def reroll_shop(sim: Handle, live: Handle, *, delay: float = _DEFAULT_DELAY) -> None:
    """Reroll the shop on both backends."""
    sim("reroll", {})
    live("reroll", {})
    time.sleep(delay)


def get_state(handle: Handle) -> str:
    """Get the current game state string from a backend."""
    resp = handle("gamestate", None)
    return resp.get("state", "")


def play_through_blind(
    sim: Handle,
    live: Handle,
    *,
    max_hands: int = 4,
    delay: float = _DEFAULT_DELAY,
) -> bool:
    """Play hands until the blind is beaten or the game ends.

    Returns True if the blind was beaten (state becomes ROUND_EVAL),
    False if the game ended (GAME_OVER) or hands ran out.
    """
    for _ in range(max_hands):
        state = get_state(sim)
        if state != "SELECTING_HAND":
            break
        play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    state = get_state(sim)
    return state == "ROUND_EVAL"


def advance_past_blind(
    sim: Handle,
    live: Handle,
    *,
    delay: float = _DEFAULT_DELAY,
) -> bool:
    """Select blind, play through it, cash out, advance to next round.

    Returns True if successful, False if game ended.
    """
    select_blind(sim, live, delay=delay)
    if not play_through_blind(sim, live, delay=delay):
        return False
    cash_out(sim, live, delay=delay)
    next_round(sim, live, delay=delay)
    return True


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _card_keys(area: dict | list) -> list[str]:
    """Extract card keys from a serialized area."""
    if isinstance(area, dict):
        return [c.get("key", "") for c in area.get("cards", [])]
    return []


def compare_state(
    sim: Handle,
    live: Handle,
    *,
    label: str = "",
    check_round: bool = True,
    check_shop: bool = False,
) -> list[str]:
    """Compare gamestate between sim and live, return list of diffs."""
    sim_resp = sim("gamestate", None)
    live_resp = live("gamestate", None)

    diffs: list[str] = []

    def cmp(name: str, s: Any, lv: Any) -> None:
        if s != lv:
            diffs.append(f"{name}: sim={s} live={lv}")

    cmp("state", sim_resp.get("state", ""), live_resp.get("state", ""))
    cmp("money", sim_resp.get("money", 0), live_resp.get("money", 0))
    cmp("ante_num", sim_resp.get("ante_num", 1), live_resp.get("ante_num", 1))

    if check_round:
        sim_round = sim_resp.get("round", {})
        live_round = live_resp.get("round", {})
        state = sim_resp.get("state", "")
        if state in ("SELECTING_HAND", "ROUND_EVAL"):
            cmp("chips", sim_round.get("chips", 0), live_round.get("chips", 0))
            cmp("hands_left", sim_round.get("hands_left", 0), live_round.get("hands_left", 0))
            cmp(
                "discards_left",
                sim_round.get("discards_left", 0),
                live_round.get("discards_left", 0),
            )

    # Hand cards (as sets — order may differ)
    sim_hand = set(_card_keys(sim_resp.get("hand", {})))
    live_hand = set(_card_keys(live_resp.get("hand", {})))
    cmp("hand_cards", sim_hand, live_hand)

    # Deck size
    cmp(
        "deck_size",
        sim_resp.get("cards", {}).get("count", 0),
        live_resp.get("cards", {}).get("count", 0),
    )

    # Jokers (ordered)
    cmp("jokers", _card_keys(sim_resp.get("jokers", {})), _card_keys(live_resp.get("jokers", {})))

    # Consumables (as sets)
    cmp(
        "consumables",
        set(_card_keys(sim_resp.get("consumables", {}))),
        set(_card_keys(live_resp.get("consumables", {}))),
    )

    if check_shop:
        cmp("shop", _card_keys(sim_resp.get("shop", {})), _card_keys(live_resp.get("shop", {})))

    return diffs


# ---------------------------------------------------------------------------
# Scenario runner pattern
# ---------------------------------------------------------------------------


def run_joker_scenario(
    sim: Handle,
    live: Handle,
    *,
    joker_key: str,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
    edition: str | None = None,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add joker, play hand[0:5], compare.

    This is the workhorse for most joker scenarios.
    """
    if not seed:
        seed = f"J_{joker_key.upper()}"

    start_both(sim, live, seed=seed, delay=delay)
    select_blind(sim, live, delay=delay)

    add_kwargs: dict[str, Any] = {"key": joker_key}
    if edition:
        add_kwargs["edition"] = edition
    add_both(sim, live, **add_kwargs)

    # Play the first 5 cards from hand
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    diffs = compare_state(sim, live, label=f"after play with {joker_key}")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"Joker {joker_key}: {'PASS' if not diffs else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# Hand presets — known card sets for triggering specific joker conditions
# ---------------------------------------------------------------------------

HAND_PRESETS: dict[str, list[str]] = {
    # Poker hand types
    "PAIR": ["H_A", "D_A", "S_K", "C_Q", "H_J"],
    "THREE_KIND": ["H_A", "D_A", "S_A", "C_K", "H_Q"],
    "FOUR_KIND": ["H_A", "D_A", "S_A", "C_A", "H_K"],
    "TWO_PAIR": ["H_A", "D_A", "S_K", "C_K", "H_Q"],
    "FULL_HOUSE": ["H_A", "D_A", "S_A", "C_K", "H_K"],
    "STRAIGHT": ["H_5", "D_6", "S_7", "C_8", "H_9"],
    "FLUSH_HEARTS": ["H_2", "H_5", "H_8", "H_J", "H_A"],
    "FLUSH_DIAMONDS": ["D_2", "D_5", "D_8", "D_J", "D_A"],
    "FLUSH_CLUBS": ["C_2", "C_5", "C_8", "C_J", "C_A"],
    "FLUSH_SPADES": ["S_2", "S_5", "S_8", "S_J", "S_A"],
    # Rank-specific
    "FACE_CARDS": ["H_J", "D_Q", "S_K", "C_J", "H_Q"],
    "EVEN_RANKS": ["H_2", "D_4", "S_6", "C_8", "H_T"],
    "ODD_RANKS": ["H_3", "D_5", "S_7", "C_9", "H_A"],
    "FIBONACCI": ["H_A", "D_2", "S_3", "C_5", "H_8"],
    "WITH_ACES": ["H_A", "D_A", "S_3", "C_5", "H_8"],
    "WITH_TENS_FOURS": ["H_T", "D_4", "S_T", "C_4", "H_7"],
    "KINGS_QUEENS": ["H_K", "D_K", "S_Q", "C_Q", "H_J"],
    "HACK_RANKS": ["H_2", "D_3", "S_4", "C_5", "H_2"],
    "WITH_EIGHTS": ["H_8", "D_8", "S_3", "C_5", "H_A"],
    "WITH_SIXES": ["H_6", "D_6", "S_8", "C_9", "H_T"],
    "WITH_WEE_TWOS": ["H_2", "D_2", "S_3", "C_5", "H_A"],
    # Multi-condition
    "ALL_SUITS": ["H_A", "D_K", "C_Q", "S_J", "H_T"],
    "STRAIGHT_ACE": ["H_A", "D_T", "S_J", "C_Q", "H_K"],
    "STRAIGHT_FLUSH": ["H_5", "H_6", "H_7", "H_8", "H_9"],
    "SPADES_CLUBS": ["S_2", "S_5", "C_8", "C_J", "S_A"],
    "HIGH_CARD": ["H_2", "D_5", "S_8", "C_J", "H_A"],
    # Variable-count hands
    "THREE_CARDS": ["H_A", "D_K", "S_Q"],
    "FOUR_CARDS": ["H_A", "D_K", "S_Q", "C_J"],
}


def get_hand_count(handle: Handle) -> int:
    """Get the number of cards currently in hand."""
    resp = handle("gamestate", None)
    return resp.get("hand", {}).get("count", 0)


def run_joker_with_setup(
    sim: Handle,
    live: Handle,
    *,
    joker_key: str,
    hand_preset: str | list[str] | None = None,
    play_count: int | None = None,
    pre_discard: list[str] | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
    edition: str | None = None,
) -> ScenarioResult:
    """Enhanced joker scenario: inject specific cards, optionally discard, play.

    Args:
        joker_key: The joker center key (e.g. "j_jolly").
        hand_preset: Preset name from HAND_PRESETS, or a list of card keys.
            If None, falls back to standard play [0..4].
        play_count: Number of injected cards to play. Defaults to len(cards).
        pre_discard: Card keys to inject and discard before the main hand.
        seed: Run seed.
        delay: Action delay.
        edition: Optional joker edition.
    """
    if not seed:
        seed = f"J_{joker_key.upper()}"

    start_both(sim, live, seed=seed, delay=delay)
    select_blind(sim, live, delay=delay)

    # Add the joker
    add_kwargs: dict[str, Any] = {"key": joker_key}
    if edition:
        add_kwargs["edition"] = edition
    add_both(sim, live, **add_kwargs)

    # No preset — fall back to standard behavior
    if hand_preset is None:
        play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
        diffs = compare_state(sim, live, label=f"after play with {joker_key}")
        return ScenarioResult(
            passed=len(diffs) == 0,
            diffs=diffs,
            details=f"Joker {joker_key}: {'PASS' if not diffs else 'FAIL'}",
        )

    # Resolve preset
    cards = HAND_PRESETS[hand_preset] if isinstance(hand_preset, str) else hand_preset
    if play_count is None:
        play_count = len(cards)

    # Pre-discard phase (for discard-triggered jokers)
    if pre_discard:
        for card_key in pre_discard:
            add_both(sim, live, key=card_key)
        count = get_hand_count(sim)
        discard_indices = list(range(count - len(pre_discard), count))
        discard(sim, live, discard_indices, delay=delay)

    # Inject hand cards
    for card_key in cards:
        add_both(sim, live, key=card_key)

    # Play the last N injected cards
    count = get_hand_count(sim)
    play_indices = list(range(count - play_count, count))
    play_hand(sim, live, play_indices, delay=delay)

    diffs = compare_state(sim, live, label=f"after play with {joker_key}")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"Joker {joker_key}: {'PASS' if not diffs else 'FAIL'}",
    )


def run_consumable_scenario(
    sim: Handle,
    live: Handle,
    *,
    consumable_key: str,
    targets: list[int] | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add consumable, use it, compare."""
    if not seed:
        seed = f"C_{consumable_key.upper()}"

    start_both(sim, live, seed=seed, delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key=consumable_key)
    use_consumable(sim, live, 0, cards=targets, delay=delay)

    diffs = compare_state(sim, live, label=f"after using {consumable_key}")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"Consumable {consumable_key}: {'PASS' if not diffs else 'FAIL'}",
    )


def run_modifier_scenario(
    sim: Handle,
    live: Handle,
    *,
    card_key: str,
    enhancement: str = "c_base",
    edition: str | None = None,
    seal: str | None = None,
    seed: str = "",
    delay: float = _DEFAULT_DELAY,
) -> ScenarioResult:
    """Standard pattern: start game, select blind, add modified card, play it, compare."""
    if not seed:
        seed = f"M_{enhancement}_{card_key}"

    start_both(sim, live, seed=seed, delay=delay)
    select_blind(sim, live, delay=delay)

    add_kwargs: dict[str, Any] = {"key": card_key}
    if enhancement != "c_base":
        add_kwargs["enhancement"] = enhancement
    if edition:
        add_kwargs["edition"] = edition
    if seal:
        add_kwargs["seal"] = seal
    add_both(sim, live, **add_kwargs)

    # Play hand including the added card (it should be at the end of hand)
    # We play indices 0-4 which will include the dealt hand
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    label = f"{enhancement}"
    if edition:
        label += f"+{edition}"
    if seal:
        label += f"+{seal}"

    diffs = compare_state(sim, live, label=f"after play with {label} {card_key}")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"Modifier {label} on {card_key}: {'PASS' if not diffs else 'FAIL'}",
    )
