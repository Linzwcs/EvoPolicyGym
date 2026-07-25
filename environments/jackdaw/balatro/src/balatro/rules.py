"""Public, version-pinned rules exposed to Policy authors."""

from __future__ import annotations

from typing import Any, cast

from evopolicygym.policy import PolicyValue
from jackdaw.engine import consumables as jackdaw_consumables
from jackdaw.engine import jokers as jackdaw_jokers
from jackdaw.engine.data.hands import HAND_BASE, HandType
from jackdaw.engine.data.prototypes import (
    BOOSTERS,
    ENHANCEMENTS,
    JOKERS,
    PLANETS,
    SPECTRALS,
    TAGS,
    TAROTS,
    VOUCHERS,
)

POLICY_GUIDE: dict[str, PolicyValue] = {
    "profile": (
        "One complete Red Deck, White Stake run. Red Deck grants one extra discard "
        "per round. Clear Small Blind, Big Blind, and Boss Blind in each Ante; "
        "defeat the Ante 8 Boss Blind to win. Objects whose effects are inactive in "
        "the pinned engine are excluded before pool selection and listed in "
        "Benchmark metadata.excluded_content."
    ),
    "information_contract": {
        "core_rules": (
            "This guide is the complete stable core contract. A Policy does not need "
            "an unseen card catalog to act."
        ),
        "visible_tooltips": (
            "Every currently visible Blind, Joker, Enhancement, consumable, Voucher, "
            "Booster, and Tag carries its implemented rule in the observation. Editions "
            "and Seals use their visible names and the exact definitions in card_modifiers. "
            "Unseen future draws, shop contents, and draw order remain hidden."
        ),
        "authority": (
            "For the current decision, observation.legal_actions is authoritative for "
            "available action kinds, target areas, target indices, and card-count bounds."
        ),
        "memory": (
            "One Policy instance may retain memory across act() calls within the same "
            "Episode, but every Episode starts with a fresh process, Policy, and state."
        ),
    },
    "objective": {
        "benchmark_reward": (
            "A transition's top-level reward is Benchmark progress: +1 when a Blind is "
            "actually cleared, plus a one-time +1000 when the run is won. Playing a hand, "
            "shopping, cashing out, or skipping without clearing a Blind normally gives 0."
        ),
        "game_money": (
            "Money is a separate spendable in-game resource. blind.dollar_reward is the "
            "Blind's base cash payout and is never the transition reward."
        ),
        "episode_score": (
            "The Episode score is the sum of transition rewards: Blinds cleared plus the "
            "1000-point win bonus. A Policy failure receives zero regardless of progress."
        ),
    },
    "run_loop": {
        "blind_sequence": (
            "Each Ante presents Small Blind, Big Blind, then an unskippable Boss Blind."
        ),
        "select_or_skip": (
            "Selecting a Blind starts it. Small and Big Blinds may instead be skipped for "
            "their visible Tag; a skipped Blind gives no Blind cash and no +1 Benchmark reward."
        ),
        "play": (
            "During an active Blind, play or discard 1 to 5 cards. Reach or exceed "
            "blind.target_chips before hands_left becomes zero."
        ),
        "cash_out": (
            "Clearing a Blind enters round_eval. cash_out accepts the visible round "
            "earnings and opens the shop."
        ),
        "shop": (
            "In the shop, buy or sell visible cards, redeem the Voucher, reroll, open "
            "Boosters, use consumables, reorder Jokers, or choose next_round."
        ),
        "termination": (
            "The run loses immediately when no hands remain and the active Blind target "
            "has not been reached. It wins after the Ante 8 Boss is cleared and cashed out."
        ),
    },
    "phases": {
        "blind_select": (
            "Inspect the offered Blind and skip Tag, then select_blind or, for Small/Big, "
            "skip_blind. Any other available actions are listed by legal_actions."
        ),
        "selecting_hand": (
            "Choose play_hand or discard using current hand indices; usable consumables, "
            "sorting, and reordering actions appear in legal_actions."
        ),
        "round_eval": "The Blind is cleared; inspect earnings and use cash_out.",
        "shop": (
            "Inspect shop cards, Vouchers, Boosters, money, slots, and costs. Buy, sell, "
            "reroll, open a Booster, or use next_round."
        ),
        "pack_opening": (
            "Choose a visible pack card with pick_pack_card, including required hand-card "
            "targets, or use skip_pack. choices_remaining reports remaining picks."
        ),
        "game_over": "Terminal state; no further action is requested.",
    },
    "actions": {
        "strictness": (
            "Actions are strict tagged objects. Include exactly the required fields; "
            "invalid actions are never repaired or replaced and cause Policy failure."
        ),
        "simple": (
            "select_blind, skip_blind, cash_out, reroll_shop, next_round, skip_pack, "
            "sort_hand_by_rank, and sort_hand_by_suit use only {'kind': NAME}."
        ),
        "hand_selection": (
            "play_hand and discard use {'kind': NAME, 'card_indices': [..]}. Indices "
            "refer to observation.hand in that same decision. Select 1 to 5 distinct cards."
        ),
        "entity_target": (
            "buy_card, sell_joker, sell_consumable, redeem_voucher, open_booster, and "
            "swap actions use {'kind': NAME, 'target_index': I}; consult target_area."
        ),
        "card_and_targets": (
            "use_consumable and pick_pack_card use {'kind': NAME, 'target_index': I, "
            "'card_indices': [..]}. legal_actions supplies exact valid targets and bounds."
        ),
        "order": (
            "Card selection order and Joker order can affect effects. Entity indices are "
            "ephemeral and must be read again after every action."
        ),
    },
    "observation": {
        "progress": "Ante, Blinds cleared, next/current Blind role, step count, and win state.",
        "resources": (
            "Spendable money, current Blind Chips, hands/discards left, reroll cost, "
            "and card-slot limits."
        ),
        "blind": (
            "Current or offered Blind, chip target, base dollar payout, Boss rule, debuff "
            "state, and visible skip Tag."
        ),
        "cards": (
            "hand, jokers, consumables, shop, and pack contain only currently visible cards. "
            "index is the action handle; rank/suit/chips identify playing cards; rule is the "
            "human-readable implemented tooltip; cost and sell_value are game dollars."
        ),
        "face_down": (
            "A face-down card deliberately hides its identity and rule. Do not infer hidden "
            "values from key, name, rank, suit, or chips."
        ),
        "last_hand": (
            "After play_hand, last_hand reports the detected hand, actual scoring cards, "
            "final Chips, final Mult, total score, dollars earned, and a bounded breakdown."
        ),
        "round_earnings": (
            "In round_eval, round_earnings itemizes the Blind dollar reward, unused-hand "
            "and discard bonuses, Joker dollars, interest, rental cost, and total dollars."
        ),
        "poker_hands": (
            "Current hand levels, base Chips, base Mult, and play counts, equivalent to Run Info."
        ),
        "deck": (
            "Only visible remaining-deck counts and composition are published; draw order is hidden."
        ),
    },
    "scoring": {
        "formula": "Each played hand adds floor(final Chips × final Mult) to the Blind score.",
        "rank_chips": (
            "When a playing card scores, Ace contributes 11 Chips, King/Queen/Jack 10, "
            "and numbered cards their number, before modifiers."
        ),
        "selected_vs_scoring": (
            "A Policy may select up to five cards, but only cards belonging to the "
            "detected poker hand score by default. Unmatched kickers do not contribute "
            "rank Chips unless an effect such as Splash says that all played cards score."
        ),
        "base_values": (
            "The detected poker hand and its current level provide base Chips and Mult; "
            "scoring cards, enhancements, editions, held-card effects, Jokers, and the "
            "Blind then modify them."
        ),
        "joker_order": (
            "Jokers resolve from left to right. Additive Chips and Mult are applied "
            "before later multiplicative X Mult effects, so Joker order can change score."
        ),
        "held_cards": (
            "Effects that say 'held in hand' apply to cards not selected for play, in their "
            "documented scoring context."
        ),
        "debuffs": (
            "A debuffed card or Joker contributes no normal effect. Inspect each visible "
            "entity's debuffed field and the active Blind rule."
        ),
    },
    "poker_hands": {
        "High Card": "Highest scoring card when no stronger hand is present.",
        "Pair": "Two cards of the same rank.",
        "Two Pair": "Two different pairs.",
        "Three of a Kind": "Three cards of the same rank.",
        "Straight": "Five consecutive ranks; Ace may be high or form A-2-3-4-5.",
        "Flush": "Five cards of the same suit.",
        "Full House": "Three cards of one rank and two cards of another rank.",
        "Four of a Kind": "Four cards of the same rank.",
        "Straight Flush": "Five consecutive ranks of the same suit.",
        "Five of a Kind": "Five cards of the same rank.",
        "Flush House": "A Full House whose five cards share one suit.",
        "Flush Five": "Five cards of the same rank and suit.",
    },
    "economy": {
        "money": "Money buys cards, packs, Vouchers, and shop rerolls.",
        "cash_out": (
            "Round cash is the Blind's dollar_reward plus normally $1 per unused hand, "
            "plus Joker/card earnings and interest, minus rental costs. It becomes "
            "spendable when cash_out is used."
        ),
        "interest": (
            "At normal White Stake rules, cash out grants $1 interest per $5 held, "
            "up to $5, unless a visible effect changes or disables interest."
        ),
        "sell_value": "Owned Jokers and consumables may be sold for their visible sell value.",
        "slots": (
            "A non-Negative Joker normally requires a free Joker slot. Consumables and "
            "pack choices similarly respect their visible slot and target constraints."
        ),
    },
    "card_modifiers": {
        "enhancements": {
            "Bonus Card": "+30 Chips when scored.",
            "Mult Card": "+4 Mult when scored.",
            "Wild Card": "Counts as every suit.",
            "Glass Card": "X2 Mult when scored; 1 in 4 chance to be destroyed.",
            "Steel Card": "X1.5 Mult while held in hand.",
            "Stone Card": "+50 Chips when scored; has no rank or suit for poker-hand detection.",
            "Gold Card": "Earn $3 if held in hand at end of round.",
            "Lucky Card": "When scored: 1 in 5 chance for +20 Mult and 1 in 15 chance for $20.",
        },
        "editions": {
            "foil": "+50 Chips when this card or Joker scores.",
            "holo": "+10 Mult when this card or Joker scores.",
            "polychrome": "X1.5 Mult when this card or Joker scores.",
            "negative": "An owned Negative Joker adds one Joker slot.",
        },
        "seals": {
            "Red": "Retrigger this card once in its scoring or held-card context.",
            "Blue": "If held at round end, create the Planet card for the last played hand if space allows.",
            "Gold": "Earn $3 when this card scores.",
            "Purple": "Create a Tarot card when this card is discarded if space allows.",
        },
    },
    "visible_rules": {
        "contract": (
            "Every visible gameplay object includes rule.summary and structured current "
            "parameters when applicable. Treat that rule as authoritative for this pinned engine."
        ),
        "dynamic_state": (
            "ability and rule.parameters contain the current mutable values for scaling, "
            "decaying, probabilistic, or hand-type-dependent Jokers."
        ),
        "catalog_scope": (
            "Only encountered objects need to be understood. The Instruction intentionally "
            "does not preload the full unseen Joker or consumable catalog."
        ),
    },
}

_RARITY_NAMES = {
    1: "Common",
    2: "Uncommon",
    3: "Rare",
    4: "Legendary",
}

_SUMMARY_OVERRIDES = {
    "j_cloud_9": "Cloud 9: earn $1 per 9 in the full deck at end of round.",
    "j_delayed_grat": (
        "Delayed Gratification: if no discards are used this round, earn $2 "
        "per remaining discard at end of round."
    ),
    "j_golden": "Golden Joker: earn $4 at end of round.",
    "j_rocket": (
        "Rocket: earn $1 at end of round; its payout permanently gains $2 "
        "after each Boss Blind defeated."
    ),
    "j_satellite": (
        "Satellite: earn $1 at end of round for each unique Planet card "
        "used during this run."
    ),
}

_ENHANCEMENT_RULES = {
    "m_bonus": "Bonus Card: +30 Chips when scored.",
    "m_mult": "Mult Card: +4 Mult when scored.",
    "m_wild": "Wild Card: counts as every suit.",
    "m_glass": "Glass Card: X2 Mult when scored; 1 in 4 chance to be destroyed.",
    "m_steel": "Steel Card: X1.5 Mult while held in hand.",
    "m_stone": "Stone Card: +50 Chips when scored; has no rank or suit for hand detection.",
    "m_gold": "Gold Card: earn $3 if held in hand at end of round.",
    "m_lucky": (
        "Lucky Card: when scored, 1 in 5 chance for +20 Mult and "
        "1 in 15 chance to earn $20."
    ),
}

_CONSUMABLE_SUMMARY_OVERRIDES = {
    "c_magician": "The Magician: enhance up to 2 selected cards into Lucky Cards.",
    "c_empress": "The Empress: enhance up to 2 selected cards into Mult Cards.",
    "c_heirophant": "The Hierophant: enhance up to 2 selected cards into Bonus Cards.",
    "c_lovers": "The Lovers: enhance 1 selected card into a Wild Card.",
    "c_chariot": "The Chariot: enhance 1 selected card into a Steel Card.",
    "c_justice": "Justice: enhance 1 selected card into a Glass Card.",
    "c_devil": "The Devil: enhance 1 selected card into a Gold Card.",
    "c_tower": "The Tower: enhance 1 selected card into a Stone Card.",
    "c_star": "The Star: convert up to 3 selected cards to Diamonds.",
    "c_moon": "The Moon: convert up to 3 selected cards to Clubs.",
    "c_sun": "The Sun: convert up to 3 selected cards to Hearts.",
    "c_world": "The World: convert up to 3 selected cards to Spades.",
    "c_talisman": "Talisman: add a Gold Seal to 1 selected card.",
    "c_deja_vu": "Deja Vu: add a Red Seal to 1 selected card.",
    "c_trance": "Trance: add a Blue Seal to 1 selected card.",
    "c_medium": "Medium: add a Purple Seal to 1 selected card.",
}

_VOUCHER_RULES = {
    "v_overstock_norm": "Overstock: permanently add 1 card slot to every shop.",
    "v_overstock_plus": "Overstock Plus: permanently add 1 more card slot to every shop.",
    "v_clearance_sale": "Clearance Sale: all shop cards and packs are 25% cheaper.",
    "v_liquidation": "Liquidation: all shop cards and packs are 50% cheaper.",
    "v_hone": "Hone: visible card editions appear at 2 times the normal rate.",
    "v_glow_up": "Glow Up: visible card editions appear at 4 times the normal rate.",
    "v_reroll_surplus": "Reroll Surplus: shop rerolls permanently cost $2 less.",
    "v_reroll_glut": "Reroll Glut: shop rerolls permanently cost another $2 less.",
    "v_crystal_ball": "Crystal Ball: permanently add 1 consumable slot.",
    "v_omen_globe": (
        "Omen Globe: excluded from this Benchmark because its pack effect is inactive."
    ),
    "v_telescope": (
        "Telescope: excluded from this Benchmark because its pack effect is inactive."
    ),
    "v_observatory": (
        "Observatory: excluded from this Benchmark because its scoring effect is inactive."
    ),
    "v_grabber": "Grabber: permanently add 1 hand per round.",
    "v_nacho_tong": "Nacho Tong: permanently add 1 more hand per round.",
    "v_wasteful": "Wasteful: permanently add 1 discard per round.",
    "v_recyclomancy": "Recyclomancy: permanently add 1 more discard per round.",
    "v_tarot_merchant": "Tarot Merchant: Tarot cards appear 2.4 times as often in shops.",
    "v_tarot_tycoon": "Tarot Tycoon: Tarot cards appear 8 times as often in shops.",
    "v_planet_merchant": "Planet Merchant: Planet cards appear 2.4 times as often in shops.",
    "v_planet_tycoon": "Planet Tycoon: Planet cards appear 8 times as often in shops.",
    "v_seed_money": "Seed Money: raise the money earning interest cap from $25 to $50.",
    "v_money_tree": "Money Tree: raise the money earning interest cap to $100.",
    "v_blank": "Blank: no immediate gameplay effect; it is the prerequisite for Antimatter.",
    "v_antimatter": "Antimatter: permanently add 1 Joker slot.",
    "v_magic_trick": "Magic Trick: playing cards may appear in shop card slots.",
    "v_illusion": (
        "Illusion: playing cards may appear in shop card slots; its additional modifier "
        "roll is not active in this pinned Jackdaw revision."
    ),
    "v_hieroglyph": "Hieroglyph: decrease Ante by 1 and permanently lose 1 hand per round.",
    "v_petroglyph": "Petroglyph: decrease Ante by 1 and permanently lose 1 discard per round.",
    "v_directors_cut": (
        "Director's Cut: excluded from this Benchmark because no Boss reroll action is active."
    ),
    "v_retcon": "Retcon: excluded from this Benchmark because no Boss reroll action is active.",
    "v_paint_brush": "Paint Brush: permanently increase hand size by 1.",
    "v_palette": "Palette: permanently increase hand size by 1 more.",
}

_TAG_RULES = {
    "tag_boss": "Boss Tag: immediately reroll the current Boss Blind.",
    "tag_buffoon": "Buffoon Tag: immediately open a free Mega Buffoon Pack.",
    "tag_charm": "Charm Tag: immediately open a free Mega Arcana Pack.",
    "tag_coupon": "Coupon Tag: all cards and Booster Packs in the next shop are free.",
    "tag_d_six": "D6 Tag: the next shop starts with one free reroll.",
    "tag_double": "Double Tag: duplicate the next non-Double Tag that is obtained.",
    "tag_economy": "Economy Tag: gain dollars equal to current money, capped at $40.",
    "tag_ethereal": "Ethereal Tag: immediately open a free Spectral Pack.",
    "tag_foil": "Foil Tag: the next eligible shop Joker becomes Foil and free.",
    "tag_garbage": "Garbage Tag: gain $1 per unused discard accumulated this run.",
    "tag_handy": "Handy Tag: gain $1 per hand played this run.",
    "tag_holo": "Holographic Tag: the next eligible shop Joker becomes Holographic and free.",
    "tag_investment": "Investment Tag: gain $25 after the next Boss Blind is defeated.",
    "tag_juggle": "Juggle Tag: increase hand size by 3 for the next round.",
    "tag_meteor": "Meteor Tag: immediately open a free Mega Celestial Pack.",
    "tag_negative": "Negative Tag: the next eligible shop Joker becomes Negative and free.",
    "tag_orbital": "Orbital Tag: level up one randomly selected poker hand by 3 levels.",
    "tag_polychrome": "Polychrome Tag: the next eligible shop Joker becomes Polychrome and free.",
    "tag_rare": "Rare Tag: excluded because its forced-Rare shop creation is inactive.",
    "tag_skip": "Skip Tag: gain $5 for each Blind skipped so far in this run.",
    "tag_standard": "Standard Tag: immediately open a free Mega Standard Pack.",
    "tag_top_up": "Top-up Tag: create up to 2 free Common Jokers, subject to Joker slots.",
    "tag_uncommon": "Uncommon Tag: excluded because its forced-Uncommon shop creation is inactive.",
    "tag_voucher": "Voucher Tag: excluded because its free-Voucher creation is inactive.",
}

_BLIND_RULES = {
    "bl_small": "No special rule.",
    "bl_big": "No special rule.",
    "bl_hook": "After each hand is played, discard two random cards from hand.",
    "bl_ox": "Playing the most-played poker hand of this run sets money to $0.",
    "bl_house": "The first hand is drawn face-down.",
    "bl_wall": "Requires a substantially larger chip target.",
    "bl_wheel": "Each drawn card has a 1 in 7 chance to be face-down.",
    "bl_arm": "Playing a poker hand decreases that hand type by one level, to a minimum of 1.",
    "bl_club": "All Club cards are debuffed.",
    "bl_fish": "Cards drawn after each played hand are face-down.",
    "bl_psychic": "Exactly five cards must be played in every hand.",
    "bl_goad": "All Spade cards are debuffed.",
    "bl_water": "Start this Blind with no discards.",
    "bl_window": "All Diamond cards are debuffed.",
    "bl_manacle": "Hand size is reduced by one.",
    "bl_eye": "No poker hand type may be played more than once this round.",
    "bl_mouth": "After the first hand, only that poker hand type may be played this round.",
    "bl_plant": "All face cards are debuffed.",
    "bl_serpent": "After playing or discarding, always draw exactly three cards.",
    "bl_pillar": "Cards already played during this Ante are debuffed.",
    "bl_needle": "Only one hand is available.",
    "bl_head": "All Heart cards are debuffed.",
    "bl_tooth": "Lose $1 for every card played.",
    "bl_flint": "Base Chips and base Mult are each reduced by half before other effects.",
    "bl_mark": "All face cards are drawn face-down.",
    "bl_final_acorn": "On Blind selection, all Jokers are flipped and shuffled.",
    "bl_final_leaf": "All cards are debuffed until one owned Joker is sold.",
    "bl_final_vessel": "Requires an extremely large chip target.",
    "bl_final_heart": "One random owned Joker is disabled for each hand.",
    "bl_final_bell": "One card is always forced into the selected hand.",
}


def visible_card_rule(
    *,
    key: str,
    card_set: str,
    parameters: dict[str, PolicyValue],
    visible_state: dict[str, PolicyValue] | None = None,
) -> PolicyValue:
    """Describe the implemented effect of one currently visible card."""

    if card_set == "Joker":
        prototype = JOKERS.get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown Joker {key}")
        rule: dict[str, PolicyValue] = {
            "summary": _joker_summary(key),
            "parameters": dict(parameters),
            "rarity": {
                "level": prototype.rarity,
                "name": _RARITY_NAMES.get(prototype.rarity, "Unknown"),
            },
        }
    else:
        summary, config = _non_joker_rule(key, card_set, parameters)
        merged_parameters = _public_config(config)
        merged_parameters.update(parameters)
        rule = {
            "summary": summary,
            "parameters": merged_parameters,
        }
    if visible_state:
        rule["visible_state"] = dict(visible_state)
    return rule


def blind_rule(key: str) -> str:
    """Return the public semantic rule for one Blind."""

    rule = _BLIND_RULES.get(key)
    if rule is None:
        raise RuntimeError(f"Jackdaw returned unknown Blind {key}")
    return rule


def visible_tag_rule(key: str) -> dict[str, PolicyValue]:
    """Describe one currently visible skip or awarded Tag."""

    prototype = TAGS.get(key)
    if prototype is None:
        raise RuntimeError(f"Jackdaw returned unknown Tag {key}")
    summary = _TAG_RULES.get(key)
    if summary is None:
        raise RuntimeError(f"Tag {key} has no public rule")
    return {
        "key": key,
        "name": prototype.name,
        "rule": {
            "summary": summary,
            "parameters": _public_config(prototype.config),
        },
    }


def _joker_summary(key: str) -> str:
    override = _SUMMARY_OVERRIDES.get(key)
    if override is not None:
        return override

    registry = cast(
        dict[str, Any],
        getattr(jackdaw_jokers, "_REGISTRY", {}),
    )
    handler = registry.get(key)
    document = getattr(handler, "__doc__", None)
    if type(document) is not str or not document.strip():
        raise RuntimeError(f"Jackdaw Joker {key} has no implemented rule description")
    summary = document.strip().splitlines()[0]
    summary = summary.split(" Source:", maxsplit=1)[0]
    for prefix in ("Passive/meta: ", "Passive: ", "scoring stub — "):
        summary = summary.removeprefix(prefix)
    replacements = {
        "setting_blind": "Blind selection",
        "open_booster": "opening a Booster Pack",
        "xMult": "X Mult",
        "chip_mod": "Chips",
        "t_chips": "conditional Chips",
        "t_mult": "conditional Mult",
        "s_mult": "per-card Mult",
        "sell_cost": "sell value",
    }
    for source, destination in replacements.items():
        summary = summary.replace(source, destination)
    return summary[:240]


def _non_joker_rule(
    key: str,
    card_set: str,
    parameters: dict[str, PolicyValue],
) -> tuple[str, dict[str, Any]]:
    if card_set == "Default":
        return (
            "Base playing card: contributes its visible rank Chips when it scores.",
            {},
        )

    if card_set == "Enhanced":
        prototype = ENHANCEMENTS.get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown Enhancement {key}")
        summary = _ENHANCEMENT_RULES.get(key)
        if summary is None:
            raise RuntimeError(f"Enhancement {key} has no public rule")
        return summary, prototype.config

    if card_set in {"Tarot", "Planet", "Spectral"}:
        prototype = {
            "Tarot": TAROTS,
            "Planet": PLANETS,
            "Spectral": SPECTRALS,
        }[card_set].get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown {card_set} {key}")
        return _consumable_summary(key, card_set), prototype.config

    if card_set == "Voucher":
        prototype = VOUCHERS.get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown Voucher {key}")
        summary = _VOUCHER_RULES.get(key)
        if summary is None:
            raise RuntimeError(f"Voucher {key} has no public rule")
        config = dict(prototype.config)
        if prototype.requires:
            config["requires"] = list(prototype.requires)
        return summary, config

    if card_set == "Booster":
        prototype = BOOSTERS.get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown Booster {key}")
        choices = _config_int(prototype.config, "choose", 1)
        cards = _config_int(prototype.config, "extra", 1)
        content = {
            "Arcana": "Tarot cards",
            "Celestial": "Planet cards",
            "Spectral": "Spectral cards",
            "Standard": "playing cards",
            "Buffoon": "Jokers",
        }.get(prototype.kind, f"{prototype.kind} cards")
        return (
            f"{prototype.name}: reveal {cards} {content} and choose {choices}.",
            prototype.config,
        )

    effect = parameters.get("effect")
    if type(effect) is str and effect:
        return effect, {}
    return "No additional implemented rule.", {}


def _consumable_summary(key: str, card_set: str) -> str:
    override = _CONSUMABLE_SUMMARY_OVERRIDES.get(key)
    if override is not None:
        return override

    if card_set == "Planet":
        prototype = PLANETS[key]
        hand_name = prototype.config.get("hand_type")
        if type(hand_name) is not str:
            raise RuntimeError(f"Planet {key} has no hand_type")
        try:
            hand_type = HandType(hand_name)
        except ValueError as error:
            raise RuntimeError(f"Planet {key} has invalid hand_type") from error
        values = HAND_BASE[hand_type]
        return (
            f"{prototype.name}: level up {hand_name} by 1, adding "
            f"{values.l_chips} base Chips and {values.l_mult} base Mult."
        )

    registry = cast(
        dict[str, Any],
        getattr(jackdaw_consumables, "_CONSUMABLE_REGISTRY", {}),
    )
    handler = registry.get(key)
    document = getattr(handler, "__doc__", None)
    if type(document) is not str or not document.strip():
        raise RuntimeError(f"Consumable {key} has no implemented rule description")
    summary = document.strip().splitlines()[0]
    summary = summary.split(" Source:", maxsplit=1)[0]
    return summary[:240]


def _public_config(config: dict[str, Any]) -> dict[str, PolicyValue]:
    return {
        str(key): _public_config_value(value, depth=0)
        for key, value in sorted(config.items())
    }


def _public_config_value(value: Any, *, depth: int) -> PolicyValue:
    if value is None:
        return None
    if type(value) is bool:
        return bool(value)
    if type(value) is int:
        return int(value)
    if type(value) is float:
        return float(value)
    if type(value) is str:
        return str(value)
    if depth >= 3:
        raise RuntimeError("Card rule config is too deeply nested")
    if type(value) is list:
        return [
            _public_config_value(item, depth=depth + 1)
            for item in value
        ]
    if type(value) is dict:
        return {
            str(key): _public_config_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise RuntimeError(f"Card rule config contains unsupported {type(value).__name__}")


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if type(value) is not int:
        raise RuntimeError(f"Card rule config {key} is not an integer")
    return value


__all__ = [
    "POLICY_GUIDE",
    "blind_rule",
    "visible_card_rule",
    "visible_tag_rule",
]
