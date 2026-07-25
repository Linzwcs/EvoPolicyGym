"""Full run initialization chain matching ``game.lua:start_run``.

Public API
----------
:func:`init_game_object`  — create the default ``G.GAME`` state dict.
:func:`initialize_run`    — complete run init (stake → back → challenge → deck → ante).
:func:`start_round`       — per-round reset from ``round_resets``.

Source references
-----------------
- ``game.lua:1862``  — ``init_game_object``
- ``game.lua:2018``  — ``Game:start_run``
- ``state_events.lua:290`` — ``new_round``
- ``misc_functions.lua:1868`` — ``get_starting_params``
- ``common_events.lua:2271`` — targeting card reset functions
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.back import Back
from jackdaw.engine.challenges import apply_challenge
from jackdaw.engine.data.prototypes import BLINDS
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.round_lifecycle import reset_round_targets
from jackdaw.engine.stakes import apply_stake_modifiers
from jackdaw.engine.tags import assign_ante_blinds
from jackdaw.engine.vouchers import apply_voucher

# ---------------------------------------------------------------------------
# Default starting parameters — misc_functions.lua:1868
# ---------------------------------------------------------------------------


def get_starting_params() -> dict[str, Any]:
    """Return default starting parameters matching ``get_starting_params()`` in Lua."""
    return {
        "dollars": 4,
        "hand_size": 8,
        "discards": 3,
        "hands": 4,
        "reroll_cost": 5,
        "joker_slots": 5,
        "ante_scaling": 1,
        "consumable_slots": 2,
        "no_faces": False,
        "erratic_suits_and_ranks": False,
    }


# ---------------------------------------------------------------------------
# init_game_object — game.lua:1862-2016
# ---------------------------------------------------------------------------


def init_game_object() -> dict[str, Any]:
    """Create the default ``G.GAME`` state.

    Matches ``Game:init_game_object()`` (game.lua:1862-2016).  Returns a
    dict with every field the game engine expects.
    """
    bosses_used: dict[str, int] = {k: 0 for k, v in BLINDS.items() if v.boss is not None}

    return {
        "won": False,
        "round_scores": {
            "furthest_ante": 0,
            "furthest_round": 0,
            "hand": 0,
            "poker_hand": "",
            "new_collection": 0,
            "cards_played": 0,
            "cards_discarded": 0,
            "times_rerolled": 0,
            "cards_purchased": 0,
        },
        "joker_usage": {},
        "consumeable_usage": {},
        "hand_usage": {},
        "last_tarot_planet": None,
        "win_ante": 8,
        "stake": 1,
        "modifiers": {},
        "starting_params": get_starting_params(),
        "banned_keys": {},
        "round": 0,
        "probabilities": {"normal": 1},
        "bosses_used": bosses_used,
        "pseudorandom": {},
        "starting_deck_size": 52,
        "ecto_minus": 1,
        "pack_size": 2,
        "skips": 0,
        "STOP_USE": 0,
        "edition_rate": 1,
        "joker_rate": 20,
        "tarot_rate": 4,
        "planet_rate": 4,
        "spectral_rate": 0,
        "playing_card_rate": 0,
        "consumeable_buffer": 0,
        "joker_buffer": 0,
        "discount_percent": 0,
        "interest_cap": 25,
        "interest_amount": 1,
        "inflation": 0,
        "hands_played": 0,
        "unused_discards": 0,
        "perishable_rounds": 5,
        "rental_rate": 3,
        "blind": None,
        "chips": 0,
        "dollars": 0,
        "max_jokers": 0,
        "bankrupt_at": 0,
        "current_boss_streak": 0,
        "base_reroll_cost": 5,
        "blind_on_deck": None,
        "sort": "desc",
        "previous_round": {"dollars": 4},
        "tags": {},
        "tag_tally": 0,
        "pool_flags": {},
        "used_jokers": {},
        "used_vouchers": {},
        "current_round": {
            "current_hand": {
                "chips": 0,
                "mult": 0,
                "handname": "",
                "hand_level": "",
            },
            "used_packs": [],
            "cards_flipped": 0,
            "idol_card": {"suit": "Spades", "rank": "Ace"},
            "mail_card": {"rank": "Ace"},
            "ancient_card": {"suit": "Spades"},
            "castle_card": {"suit": "Spades"},
            "hands_left": 0,
            "hands_played": 0,
            "discards_left": 0,
            "discards_used": 0,
            "dollars": 0,
            "reroll_cost": 5,
            "reroll_cost_increase": 0,
            "jokers_purchased": 0,
            "free_rerolls": 0,
            "round_dollars": 0,
            "most_played_poker_hand": "High Card",
        },
        "round_resets": {
            "hands": 1,
            "discards": 1,
            "reroll_cost": 1,
            "temp_reroll_cost": None,
            "temp_handsize": None,
            "ante": 1,
            "blind_ante": 1,
            "blind_states": {
                "Small": "Select",
                "Big": "Upcoming",
                "Boss": "Upcoming",
            },
            "blind_choices": {"Small": "bl_small", "Big": "bl_big"},
            "boss_rerolled": False,
        },
        "round_bonus": {
            "next_hands": 0,
            "discards": 0,
        },
        "shop": {"joker_max": 2},
    }


# ---------------------------------------------------------------------------
# initialize_run — game.lua:2018-2410
# ---------------------------------------------------------------------------


def initialize_run(
    back_key: str,
    stake: int,
    seed: str,
    *,
    challenge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a complete run state.

    Follows the exact sequence from ``Game:start_run`` (game.lua:2042-2375):

    1. Create base game_state via :func:`init_game_object`
    2. Set stake
    3. Apply stake modifiers (:func:`~jackdaw.engine.stakes.apply_stake_modifiers`)
    4. Apply back mutations (:meth:`~jackdaw.engine.back.Back.apply_to_run`)
    5. Apply challenge overrides (jokers, consumables, vouchers, rules, restrictions)
    6. Transfer ``starting_params`` → ``round_resets`` and ``dollars``
    7. Initialize :class:`~jackdaw.engine.rng.PseudoRandom`
    8. Generate first ante (boss, tags, voucher)
    9. Build deck (:func:`~jackdaw.engine.deck_builder.build_deck`)
    10. Set card area limits
    11. Shuffle deck
    12. Reset targeting cards (idol, mail, ancient, castle)

    Returns the fully populated game_state dict.
    """
    # -----------------------------------------------------------------------
    # 1. Create base game_state
    # -----------------------------------------------------------------------
    gs = init_game_object()

    # -----------------------------------------------------------------------
    # 2. Set stake
    # -----------------------------------------------------------------------
    gs["stake"] = stake

    # -----------------------------------------------------------------------
    # 3. Apply stake modifiers
    # -----------------------------------------------------------------------
    apply_stake_modifiers(stake, gs)

    # -----------------------------------------------------------------------
    # 4. Apply back mutations
    # -----------------------------------------------------------------------
    back = Back(back_key)
    gs["selected_back_key"] = back_key
    mutations = back.apply_to_run(gs)

    sp = gs["starting_params"]

    if "hands_delta" in mutations:
        sp["hands"] += mutations["hands_delta"]
    if "discards_delta" in mutations:
        sp["discards"] += mutations["discards_delta"]
    if "hand_size_delta" in mutations:
        sp["hand_size"] += mutations["hand_size_delta"]
    if "joker_slots_delta" in mutations:
        sp["joker_slots"] += mutations["joker_slots_delta"]
    if "consumable_slots_delta" in mutations:
        sp["consumable_slots"] += mutations["consumable_slots_delta"]
    if "dollars_delta" in mutations:
        sp["dollars"] += mutations["dollars_delta"]
    if "ante_scaling" in mutations:
        sp["ante_scaling"] = mutations["ante_scaling"]
    if "money_per_hand" in mutations:
        gs["money_per_hand"] = mutations["money_per_hand"]
    if "money_per_discard" in mutations:
        gs["money_per_discard"] = mutations["money_per_discard"]
    if "no_interest" in mutations:
        gs["no_interest"] = True
    if "spectral_rate" in mutations:
        gs["spectral_rate"] = mutations["spectral_rate"]
    if "reroll_discount" in mutations:
        sp["reroll_cost"] = max(0, sp["reroll_cost"] - mutations["reroll_discount"])

    # Propagate deck-building flags from back config → starting_params
    # so that build_deck can see them (it checks starting_params first).
    back_config = back.config
    if back_config.get("remove_faces"):
        sp["no_faces"] = True
    if back_config.get("randomize_rank_suit"):
        sp["erratic_suits_and_ranks"] = True

    # Starting consumables (Magic Deck, Ghost Deck) — stored for caller to add
    starting_consumables: list[str] = mutations.get("starting_consumables", [])
    gs["starting_consumables"] = starting_consumables

    # -----------------------------------------------------------------------
    # 5. Apply challenge overrides
    # -----------------------------------------------------------------------
    if challenge:
        gs["challenge"] = challenge.get("id")
        apply_challenge(challenge, gs)

    # -----------------------------------------------------------------------
    # 6. Transfer starting_params → round_resets + dollars
    # -----------------------------------------------------------------------
    rr = gs["round_resets"]
    rr["hands"] = sp["hands"]
    rr["discards"] = sp["discards"]
    rr["reroll_cost"] = sp["reroll_cost"]
    gs["dollars"] = sp["dollars"]
    gs["base_reroll_cost"] = sp["reroll_cost"]
    gs["current_round"]["reroll_cost"] = sp["reroll_cost"]

    # Card area limits — set from starting_params BEFORE voucher application
    # so that vouchers (Crystal Ball +1 consumable slot, Antimatter +1 joker
    # slot, etc.) can modify them correctly.
    gs["hand_size"] = sp["hand_size"]
    gs["joker_slots"] = sp["joker_slots"]
    gs["consumable_slots"] = sp["consumable_slots"]

    # -----------------------------------------------------------------------
    # 6b. Apply starting vouchers (Magic Deck, Nebula Deck, Zodiac Deck)
    # -----------------------------------------------------------------------
    # Vouchers are applied AFTER starting_params → round_resets transfer and
    # card area limit initialization, matching the Lua order where
    # Card:apply_to_run modifies the live game state.
    starting_vouchers: list[str] = mutations.get("starting_vouchers", [])
    for v_key in starting_vouchers:
        gs["used_vouchers"][v_key] = True
        apply_voucher(v_key, gs)

    # -----------------------------------------------------------------------
    # 7. Initialize RNG
    # -----------------------------------------------------------------------
    rng = PseudoRandom(seed)
    gs["rng"] = rng
    gs["seeded"] = True

    # -----------------------------------------------------------------------
    # 8. Generate first ante (boss, tags, voucher) — game.lua:2177-2180
    # -----------------------------------------------------------------------
    ante_result = assign_ante_blinds(1, rng, gs)
    gs["current_round"]["voucher"] = ante_result["voucher"]
    rr["blind_choices"]["Boss"] = ante_result["blind_choices"]["Boss"]

    # -----------------------------------------------------------------------
    # 9. Build deck
    # -----------------------------------------------------------------------
    from jackdaw.engine.deck_builder import build_deck

    deck = build_deck(back_key, rng, challenge, sp)
    gs["deck"] = deck
    gs["starting_deck_size"] = len(deck)

    # -----------------------------------------------------------------------
    # 10. Shuffle deck (RNG-driven, matches game.lua:2383)
    # -----------------------------------------------------------------------
    _shuffle_deck(deck, rng, gs["round_resets"]["ante"])

    # -----------------------------------------------------------------------
    # 12. Reset targeting cards
    # -----------------------------------------------------------------------
    reset_round_targets(rng, gs["round_resets"]["ante"], gs)

    # -----------------------------------------------------------------------
    # HandLevels (not in Lua start_run but needed for scoring)
    # -----------------------------------------------------------------------
    gs["hand_levels"] = HandLevels()

    # -----------------------------------------------------------------------
    # Card areas — initialize empty lists so callers don't have to
    # -----------------------------------------------------------------------
    gs.setdefault("jokers", [])
    gs.setdefault("consumables", [])
    gs.setdefault("hand", [])
    gs.setdefault("discard_pile", [])

    # -----------------------------------------------------------------------
    # Phase — the run starts at blind selection for Small blind
    # -----------------------------------------------------------------------
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"

    return gs


# ---------------------------------------------------------------------------
# start_round — state_events.lua:290-353
# ---------------------------------------------------------------------------


def start_round(game_state: dict[str, Any]) -> None:
    """Reset per-round state from ``round_resets``.

    Called at the start of each round (not each ante).  Mirrors
    ``new_round()`` in ``state_events.lua:290-353``.
    """
    rr = game_state["round_resets"]
    rb = game_state["round_bonus"]
    cr = game_state["current_round"]

    # Hands and discards from round_resets + round_bonus
    cr["hands_left"] = max(1, rr["hands"] + rb["next_hands"])
    cr["discards_left"] = max(0, rr["discards"] + rb["discards"])

    # Round-specific counters
    cr["hands_played"] = 0
    cr["discards_used"] = 0
    cr["reroll_cost_increase"] = 0
    cr["used_packs"] = []
    cr["jokers_purchased"] = 0

    # Free rerolls (Chaos the Clown gives free rerolls; default 0)
    jokers = game_state.get("jokers", [])
    chaos_count = sum(
        1
        for j in jokers
        if getattr(j, "center_key", None) == "j_chaos"
        or (isinstance(j, dict) and j.get("key") == "j_chaos")
    )
    cr["free_rerolls"] = chaos_count

    # Calculate reroll cost — common_events.lua:2263-2268
    _calculate_reroll_cost(game_state, skip_increment=True)

    # Clear round bonus (consumed this round)
    rb["next_hands"] = 0
    rb["discards"] = 0

    # Temp hand size (Juggle Tag) — apply then clear
    if rr.get("temp_handsize"):
        game_state["hand_size"] = game_state.get("hand_size", 8) + rr["temp_handsize"]
        rr["temp_handsize"] = None

    # Temp reroll cost (D6 Tag) — clear after round start
    if rr.get("temp_reroll_cost") is not None:
        rr["temp_reroll_cost"] = None
        _calculate_reroll_cost(game_state, skip_increment=True)

    # Reset played_this_round for all hand types
    hand_levels: HandLevels | None = game_state.get("hand_levels")
    if hand_levels is not None:
        hand_levels.reset_round_counts()

    # NOTE: targeting cards (idol, mail, ancient, castle) are NOT re-rolled
    # here.  Vanilla rolls them once at run start (game.lua:2385-2389) and
    # then at each round END (state_events.lua:273-276) — never at round
    # start.  Rolling here would double-advance the 'idol/mail/anc/cas'
    # streams.  See _handle_cash_out in game.py for the round-end roll.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shuffle_deck(deck: list, rng: PseudoRandom, ante: int) -> None:
    """In-place Fisher-Yates shuffle using the run RNG.

    **Initial run shuffle** (game.lua:2383) calls ``self.deck:shuffle()``
    with NO argument, which defaults to ``pseudoseed('shuffle')``.

    **Per-round shuffle** (state_events.lua:344) calls
    ``G.deck:shuffle('nr'..ante)`` = ``pseudoseed('nr1')``.

    This function is called for the initial shuffle — uses ``'shuffle'``.
    """
    seed_val = rng.seed("shuffle")
    rng.shuffle(deck, seed_val)


def _calculate_reroll_cost(gs: dict[str, Any], *, skip_increment: bool = False) -> None:
    """Calculate the current reroll cost.

    Mirrors ``calculate_reroll_cost`` in ``common_events.lua:2263-2268``.
    """
    cr = gs["current_round"]
    rr = gs["round_resets"]

    if cr.get("free_rerolls", 0) < 0:
        cr["free_rerolls"] = 0
    if cr.get("free_rerolls", 0) > 0:
        cr["reroll_cost"] = 0
        return

    cr["reroll_cost_increase"] = cr.get("reroll_cost_increase", 0)
    if not skip_increment:
        cr["reroll_cost_increase"] += 1

    base = (
        rr.get("temp_reroll_cost")
        if rr.get("temp_reroll_cost") is not None
        else rr.get("reroll_cost", 5)
    )
    cr["reroll_cost"] = base + cr["reroll_cost_increase"]
