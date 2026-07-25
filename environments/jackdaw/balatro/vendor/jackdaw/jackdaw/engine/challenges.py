"""Challenge definitions and application logic.

Ports the 20 challenge definitions from ``challenges.lua`` and the
application logic from ``game.lua:2063-2148``.

Public API
----------
:data:`CHALLENGES`        — ``{id: challenge_def}`` dict of all 20 challenges.
:func:`get_challenge`     — look up a challenge by id.
:func:`apply_challenge`   — apply a challenge to a game_state dict.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.vouchers import apply_voucher

# ---------------------------------------------------------------------------
# Challenge definitions — challenges.lua
# ---------------------------------------------------------------------------

CHALLENGES: dict[str, dict[str, Any]] = {
    "c_omelette_1": {
        "id": "c_omelette_1",
        "name": "The Omelette",
        "rules": {
            "custom": [
                {"id": "no_reward"},
                {"id": "no_extra_hand_money"},
                {"id": "no_interest"},
            ],
            "modifiers": [],
        },
        "jokers": [
            {"id": "j_egg"},
            {"id": "j_egg"},
            {"id": "j_egg"},
            {"id": "j_egg"},
            {"id": "j_egg"},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "v_seed_money"},
                {"id": "v_money_tree"},
                {"id": "j_to_the_moon"},
                {"id": "j_rocket"},
                {"id": "j_golden"},
                {"id": "j_satellite"},
            ],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_city_1": {
        "id": "c_city_1",
        "name": "15 Minute City",
        "rules": {"custom": [], "modifiers": []},
        "jokers": [
            {"id": "j_ride_the_bus", "eternal": True},
            {"id": "j_shortcut", "eternal": True},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {
            "type": "Challenge Deck",
            "cards": [
                {"s": s, "r": r}
                for s in ("D", "C", "H", "S")
                for r in ("4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "J", "Q", "K")
            ],
        },
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_rich_1": {
        "id": "c_rich_1",
        "name": "Rich get Richer",
        "rules": {
            "custom": [{"id": "chips_dollar_cap"}],
            "modifiers": [{"id": "dollars", "value": 100}],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [{"id": "v_seed_money"}, {"id": "v_money_tree"}],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_knife_1": {
        "id": "c_knife_1",
        "name": "On a Knife's Edge",
        "rules": {"custom": [], "modifiers": []},
        "jokers": [{"id": "j_ceremonial", "eternal": True, "pinned": True}],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_xray_1": {
        "id": "c_xray_1",
        "name": "X-ray Vision",
        "rules": {
            "custom": [{"id": "flipped_cards", "value": 4}],
            "modifiers": [],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_mad_world_1": {
        "id": "c_mad_world_1",
        "name": "Mad World",
        "rules": {
            "custom": [
                {"id": "no_extra_hand_money"},
                {"id": "no_interest"},
            ],
            "modifiers": [],
        },
        "jokers": [
            {"id": "j_pareidolia", "edition": "negative", "eternal": True},
            {"id": "j_business", "eternal": True},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {
            "type": "Challenge Deck",
            "cards": [
                {"s": s, "r": r}
                for s in ("D", "C", "H", "S")
                for r in ("2", "3", "4", "5", "6", "7", "8", "9")
            ],
        },
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [{"id": "bl_plant", "type": "blind"}],
        },
    },
    "c_luxury_1": {
        "id": "c_luxury_1",
        "name": "Luxury Tax",
        "rules": {
            "custom": [{"id": "minus_hand_size_per_X_dollar", "value": 5}],
            "modifiers": [{"id": "hand_size", "value": 10}],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_non_perishable_1": {
        "id": "c_non_perishable_1",
        "name": "Non-Perishable",
        "rules": {
            "custom": [{"id": "all_eternal"}],
            "modifiers": [],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "j_gros_michel"},
                {"id": "j_ice_cream"},
                {"id": "j_cavendish"},
                {"id": "j_turtle_bean"},
                {"id": "j_ramen"},
                {"id": "j_diet_cola"},
                {"id": "j_selzer"},
                {"id": "j_popcorn"},
                {"id": "j_mr_bones"},
                {"id": "j_invisible"},
                {"id": "j_luchador"},
            ],
            "banned_tags": [],
            "banned_other": [{"id": "bl_final_leaf", "type": "blind"}],
        },
    },
    "c_medusa_1": {
        "id": "c_medusa_1",
        "name": "Medusa",
        "rules": {"custom": [], "modifiers": []},
        "jokers": [{"id": "j_marble", "eternal": True}],
        "consumeables": [],
        "vouchers": [],
        "deck": {
            "type": "Challenge Deck",
            "cards": [
                {"s": s, "r": r, **({"e": "m_stone"} if r in ("J", "Q", "K") else {})}
                for s in ("D", "C", "H", "S")
                for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
            ],
        },
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_double_nothing_1": {
        "id": "c_double_nothing_1",
        "name": "Double or Nothing",
        "rules": {
            "custom": [{"id": "debuff_played_cards"}],
            "modifiers": [],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {
            "type": "Challenge Deck",
            "cards": [
                {"s": s, "r": r, "g": "Red"}
                for s in ("D", "C", "H", "S")
                for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
            ],
        },
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_typecast_1": {
        "id": "c_typecast_1",
        "name": "Typecast",
        "rules": {
            "custom": [
                {"id": "set_eternal_ante", "value": 4},
                {"id": "set_joker_slots_ante", "value": 4},
            ],
            "modifiers": [],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [{"id": "bl_final_leaf", "type": "blind"}],
        },
    },
    "c_inflation_1": {
        "id": "c_inflation_1",
        "name": "Inflation",
        "rules": {
            "custom": [{"id": "inflation"}],
            "modifiers": [],
        },
        "jokers": [{"id": "j_credit_card"}],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "v_clearance_sale"},
                {"id": "v_liquidation"},
            ],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_bram_poker_1": {
        "id": "c_bram_poker_1",
        "name": "Bram Poker",
        "rules": {
            "custom": [{"id": "no_shop_jokers"}],
            "modifiers": [],
        },
        "jokers": [{"id": "j_vampire", "eternal": True}],
        "consumeables": [{"id": "c_empress"}, {"id": "c_emperor"}],
        "vouchers": [{"id": "v_magic_trick"}, {"id": "v_illusion"}],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_fragile_1": {
        "id": "c_fragile_1",
        "name": "Fragile",
        "rules": {"custom": [], "modifiers": []},
        "jokers": [
            {"id": "j_oops", "eternal": True, "edition": "negative"},
            {"id": "j_oops", "eternal": True, "edition": "negative"},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {
            "type": "Challenge Deck",
            "cards": [
                {"s": s, "r": r, "e": "m_glass"}
                for s in ("D", "C", "H", "S")
                for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
            ],
        },
        "restrictions": {
            "banned_cards": [
                {"id": "c_magician"},
                {"id": "c_empress"},
                {"id": "c_heirophant"},
                {"id": "c_chariot"},
                {"id": "c_devil"},
                {"id": "c_tower"},
                {"id": "c_lovers"},
                {"id": "c_incantation"},
                {"id": "c_grim"},
                {"id": "c_familiar"},
                {
                    "id": "p_standard_normal_1",
                    "ids": [
                        "p_standard_normal_1",
                        "p_standard_normal_2",
                        "p_standard_normal_3",
                        "p_standard_normal_4",
                        "p_standard_jumbo_1",
                        "p_standard_jumbo_2",
                        "p_standard_mega_1",
                        "p_standard_mega_2",
                    ],
                },
                {"id": "j_marble"},
                {"id": "j_vampire"},
                {"id": "j_midas_mask"},
                {"id": "j_certificate"},
                {"id": "v_magic_trick"},
                {"id": "v_illusion"},
            ],
            "banned_tags": [{"id": "tag_standard"}],
            "banned_other": [],
        },
    },
    "c_monolith_1": {
        "id": "c_monolith_1",
        "name": "Monolith",
        "rules": {"custom": [], "modifiers": []},
        "jokers": [
            {"id": "j_obelisk", "eternal": True},
            {"id": "j_marble", "eternal": True, "edition": "negative"},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_blast_off_1": {
        "id": "c_blast_off_1",
        "name": "Blast Off",
        "rules": {
            "custom": [],
            "modifiers": [
                {"id": "hands", "value": 2},
                {"id": "discards", "value": 2},
                {"id": "joker_slots", "value": 4},
            ],
        },
        "jokers": [
            {"id": "j_constellation", "eternal": True},
            {"id": "j_rocket", "eternal": True},
        ],
        "consumeables": [],
        "vouchers": [{"id": "v_planet_merchant"}, {"id": "v_planet_tycoon"}],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "v_grabber"},
                {"id": "v_nacho_tong"},
                {"id": "j_burglar"},
            ],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_five_card_1": {
        "id": "c_five_card_1",
        "name": "Five-Card Draw",
        "rules": {
            "custom": [],
            "modifiers": [
                {"id": "hand_size", "value": 5},
                {"id": "joker_slots", "value": 7},
                {"id": "discards", "value": 6},
            ],
        },
        "jokers": [
            {"id": "j_card_sharp"},
            {"id": "j_joker"},
        ],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "j_juggler"},
                {"id": "j_troubadour"},
                {"id": "j_turtle_bean"},
            ],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_golden_needle_1": {
        "id": "c_golden_needle_1",
        "name": "Golden Needle",
        "rules": {
            "custom": [{"id": "discard_cost", "value": 1}],
            "modifiers": [
                {"id": "hands", "value": 1},
                {"id": "discards", "value": 6},
                {"id": "dollars", "value": 10},
            ],
        },
        "jokers": [{"id": "j_credit_card"}],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "v_grabber"},
                {"id": "v_nacho_tong"},
                {"id": "j_burglar"},
            ],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_cruelty_1": {
        "id": "c_cruelty_1",
        "name": "Cruelty",
        "rules": {
            "custom": [
                {"id": "no_reward_specific", "value": "Small"},
                {"id": "no_reward_specific", "value": "Big"},
            ],
            "modifiers": [{"id": "joker_slots", "value": 3}],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [],
            "banned_tags": [],
            "banned_other": [],
        },
    },
    "c_jokerless_1": {
        "id": "c_jokerless_1",
        "name": "Jokerless",
        "rules": {
            "custom": [{"id": "no_shop_jokers"}],
            "modifiers": [{"id": "joker_slots", "value": 0}],
        },
        "jokers": [],
        "consumeables": [],
        "vouchers": [],
        "deck": {"type": "Challenge Deck"},
        "restrictions": {
            "banned_cards": [
                {"id": "c_judgement"},
                {"id": "c_wraith"},
                {"id": "c_soul"},
                {"id": "v_antimatter"},
                {
                    "id": "p_buffoon_normal_1",
                    "ids": [
                        "p_buffoon_normal_1",
                        "p_buffoon_normal_2",
                        "p_buffoon_jumbo_1",
                        "p_buffoon_mega_1",
                    ],
                },
            ],
            "banned_tags": [
                {"id": "tag_rare"},
                {"id": "tag_uncommon"},
                {"id": "tag_holo"},
                {"id": "tag_polychrome"},
                {"id": "tag_negative"},
                {"id": "tag_foil"},
                {"id": "tag_buffoon"},
                {"id": "tag_top_up"},
            ],
            "banned_other": [
                {"id": "bl_final_acorn", "type": "blind"},
                {"id": "bl_final_heart", "type": "blind"},
                {"id": "bl_final_leaf", "type": "blind"},
            ],
        },
    },
}


def get_challenge(challenge_id: str) -> dict[str, Any] | None:
    """Look up a challenge definition by its id (e.g. ``'c_omelette_1'``)."""
    return CHALLENGES.get(challenge_id)


# ---------------------------------------------------------------------------
# apply_challenge — game.lua:2063-2148
# ---------------------------------------------------------------------------


def apply_challenge(
    challenge_def: dict[str, Any],
    game_state: dict[str, Any],
) -> None:
    """Apply challenge modifications to a run's game_state in-place.

    Ports the challenge application block from ``game.lua:2063-2148``.

    Order of operations
    -------------------
    1. Starting jokers — stored in ``game_state["challenge_jokers"]`` for the
       caller to instantiate (with eternal/pinned/edition flags).
    2. Starting consumables — appended to ``game_state["starting_consumables"]``.
    3. Starting vouchers — marked as used and effects applied via
       :func:`~jackdaw.engine.vouchers.apply_voucher`.
    4. Rule modifiers — override ``starting_params`` values directly.
    5. Custom rules — set ``game_state["modifiers"]`` fields or special flags.
    6. Restrictions — populate ``game_state["banned_keys"]``.

    Parameters
    ----------
    challenge_def:
        A challenge definition dict (from :data:`CHALLENGES`).
    game_state:
        Mutable run-state dict.  Must have ``starting_params``,
        ``used_vouchers``, ``modifiers``, and ``banned_keys`` sub-dicts.
    """
    sp = game_state["starting_params"]

    # ------------------------------------------------------------------
    # 1. Starting jokers — stored for caller to instantiate
    # ------------------------------------------------------------------
    jokers = challenge_def.get("jokers", [])
    if jokers:
        game_state["challenge_jokers"] = list(jokers)

    # ------------------------------------------------------------------
    # 2. Starting consumables
    # ------------------------------------------------------------------
    consumeables = challenge_def.get("consumeables", [])
    if consumeables:
        existing = game_state.get("starting_consumables", [])
        game_state["starting_consumables"] = existing + [c["id"] for c in consumeables]

    # ------------------------------------------------------------------
    # 3. Starting vouchers — mark used + apply effects
    # ------------------------------------------------------------------
    for v in challenge_def.get("vouchers", []):
        v_key = v["id"]
        game_state["used_vouchers"][v_key] = True
        apply_voucher(v_key, game_state)

    # ------------------------------------------------------------------
    # 4. Rule modifiers — override starting_params
    # ------------------------------------------------------------------
    rules = challenge_def.get("rules", {})
    for mod in rules.get("modifiers", []):
        sp[mod["id"]] = mod["value"]

    # ------------------------------------------------------------------
    # 5. Custom rules
    # ------------------------------------------------------------------
    mods = game_state.setdefault("modifiers", {})
    for rule in rules.get("custom", []):
        rid = rule["id"]
        rval = rule.get("value")

        if rid == "no_reward":
            nr = mods.setdefault("no_blind_reward", {})
            nr["Small"] = True
            nr["Big"] = True
            nr["Boss"] = True
        elif rid == "no_reward_specific":
            nr = mods.setdefault("no_blind_reward", {})
            nr[rval] = True
        elif rval is not None:
            mods[rid] = rval
        elif rid == "no_shop_jokers":
            game_state["joker_rate"] = 0
        else:
            mods[rid] = True

    # ------------------------------------------------------------------
    # 6. Restrictions — banned keys
    # ------------------------------------------------------------------
    banned = game_state.setdefault("banned_keys", {})
    restrictions = challenge_def.get("restrictions", {})
    for category in ("banned_cards", "banned_tags", "banned_other"):
        for entry in restrictions.get(category, []):
            banned[entry["id"]] = True
            if "ids" in entry:
                for sub_id in entry["ids"]:
                    banned[sub_id] = True
