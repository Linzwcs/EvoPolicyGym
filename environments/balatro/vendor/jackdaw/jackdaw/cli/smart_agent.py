"""Smarter validation agent that aims to survive deep into late-game.

Key improvements over the basic validation agent:
- Score-aware hand selection: estimates actual chips×mult, not just hand type rank
- Blind-aware discarding: only discards when estimated score can't beat the blind
- Economy management: preserves $5 interest floors, avoids wasteful spending
- Joker evaluation: prioritizes mult/chip jokers, avoids filling slots with junk
- Consumable targeting: smarter tarot/planet usage based on hand level priorities
- Hand-type focus: concentrates planet upgrades on 1-2 hand types for scaling
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Hand scoring estimate (fast, no RNG — for decision-making only)
# ---------------------------------------------------------------------------

# Approximate value of each hand type at level 1 (chips * mult)
_HAND_BASE_SCORE: dict[str, tuple[int, int]] = {
    "Flush Five": (160, 16),
    "Flush House": (140, 14),
    "Five of a Kind": (120, 12),
    "Straight Flush": (100, 8),
    "Four of a Kind": (60, 7),
    "Full House": (40, 4),
    "Flush": (35, 4),
    "Straight": (30, 4),
    "Three of a Kind": (30, 3),
    "Two Pair": (20, 2),
    "Pair": (10, 2),
    "High Card": (5, 1),
}

# Hand type priority order (index 0 = best)
_HAND_PRIORITY = [
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]


def _estimate_hand_score(
    played_cards: list[Any],
    jokers: list[Any],
    hand_levels: Any,
) -> tuple[str, float]:
    """Estimate score for a hand without RNG.  Returns (hand_type, score)."""
    from jackdaw.engine.data.hands import HandType
    from jackdaw.engine.hand_eval import evaluate_hand

    result = evaluate_hand(played_cards, jokers)
    hand_type = result.detected_hand

    # Get base chips/mult from hand level
    base_chips, base_mult = _HAND_BASE_SCORE.get(hand_type, (5, 1))
    level = 1
    if hand_levels and hand_type in hand_levels:
        hs = hand_levels[hand_type]
        level = getattr(hs, "level", 1)

    # Scale with level
    from jackdaw.engine.data.hands import HAND_BASE

    hd = HAND_BASE.get(HandType(hand_type))
    if hd:
        base_chips = hd.chips_at(level)
        base_mult = hd.mult_at(level)

    # Add chip value from scoring cards
    card_chips = 0
    for c in played_cards:
        if hasattr(c, "get_chip_bonus"):
            card_chips += c.get_chip_bonus()

    total_chips = base_chips + card_chips

    # Estimate joker contributions (rough)
    flat_chips = 0
    flat_mult = 0
    x_mult = 1.0

    def _num(val: Any, default: int | float = 0) -> int | float:
        """Safely extract a numeric value from ability fields."""
        if isinstance(val, (int, float)):
            return val
        return default

    for j in jokers:
        a = getattr(j, "ability", {})
        if not isinstance(a, dict):
            continue
        if getattr(j, "debuff", False):
            continue
        name = a.get("name", "")
        # Common chip jokers
        if name == "Blue Joker":
            flat_chips += _num(a.get("extra")) * 2  # rough estimate
        elif name == "Stencil":
            x_mult *= max(1, _num(a.get("x_mult"), 1))
        elif name == "Steel Joker":
            x_mult *= max(1, 1 + _num(a.get("extra")) * 0.2)
        elif name in ("Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker"):
            flat_mult += _num(a.get("t_mult")) + _num(a.get("extra"))
        elif name == "Jolly Joker" and hand_type == "Pair":
            flat_mult += _num(a.get("t_mult"), 8)
        elif name == "Zany Joker" and hand_type == "Three of a Kind":
            flat_mult += _num(a.get("t_mult"), 12)
        elif name == "Mad Joker" and hand_type == "Two Pair":
            flat_mult += _num(a.get("t_mult"), 10)
        elif name == "Crazy Joker" and hand_type == "Straight":
            flat_mult += _num(a.get("t_mult"), 12)
        elif name == "Droll Joker" and hand_type == "Flush":
            flat_mult += _num(a.get("t_mult"), 10)
        elif name == "Half Joker" and len(played_cards) <= 3:
            flat_mult += _num(a.get("extra"), 20)
        elif name == "Ride the Bus":
            flat_mult += _num(a.get("extra"))
        elif name == "Blackboard":
            extra = a.get("extra")
            if isinstance(extra, dict):
                x_mult *= _num(extra.get("x_mult"), 1)
        elif name == "The Duo" and hand_type in (
            "Pair",
            "Two Pair",
            "Full House",
            "Three of a Kind",
            "Four of a Kind",
            "Five of a Kind",
        ):
            x_mult *= 2
        elif name == "The Trio" and hand_type in (
            "Three of a Kind",
            "Full House",
            "Four of a Kind",
            "Five of a Kind",
        ):
            x_mult *= 2
        elif name == "The Family" and hand_type in ("Four of a Kind", "Five of a Kind"):
            x_mult *= 2
        # Generic: if joker has t_mult or t_chips
        elif _num(a.get("t_mult")) > 0:
            flat_mult += _num(a.get("t_mult"))
        elif _num(a.get("t_chips")) > 0:
            flat_chips += _num(a.get("t_chips"))
        # x_mult jokers
        if _num(a.get("x_mult")) > 1:
            x_mult *= _num(a.get("x_mult"))

    # Edition bonuses on jokers
    for j in jokers:
        ed = getattr(j, "edition", None)
        if not ed or not isinstance(ed, dict):
            continue
        if ed.get("foil"):
            flat_chips += 50
        if ed.get("holo"):
            flat_mult += 10
        if ed.get("polychrome"):
            x_mult *= 1.5

    total_mult = base_mult + flat_mult
    total_chips += flat_chips
    score = total_chips * total_mult * x_mult
    return hand_type, score


def _pick_best_hand_scored(
    hand_cards: list[Any],
    jokers: list[Any],
    hand_levels: Any,
) -> tuple[list[int], str, float]:
    """Pick best hand by estimated score.  Returns (indices, hand_type, score)."""
    n = len(hand_cards)
    if n == 0:
        return [], "High Card", 0

    best_indices = list(range(min(5, n)))
    best_type = "High Card"
    best_score = 0.0

    # Try all 5-card combos, plus smaller sizes (3,4) for types like Three of a Kind
    for size in (min(5, n), min(4, n), min(3, n), min(2, n), 1):
        if size <= 0 or size > n:
            continue
        for combo in combinations(range(n), size):
            cards = [hand_cards[i] for i in combo]
            hand_type, score = _estimate_hand_score(cards, jokers, hand_levels)
            if score > best_score:
                best_score = score
                best_indices = list(combo)
                best_type = hand_type

    return best_indices, best_type, best_score


# ---------------------------------------------------------------------------
# Joker evaluation for purchasing decisions
# ---------------------------------------------------------------------------

# Jokers roughly ranked by late-game value
_GOOD_JOKERS: set[str] = {
    # xMult scaling
    "Steel Joker",
    "Hologram",
    "Obelisk",
    "The Idol",
    "Photograph",
    "Ancient Joker",
    "Loyalty Card",
    "Stencil",
    "Blackboard",
    # Pair/type boosters
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    # Economy
    "Golden Joker",
    "Delayed Gratification",
    "Cloud 9",
    "Business Card",
    "Faceless Joker",
    "To the Moon",
    # Retrigger / copy
    "Dusk",
    "Hack",
    "Sock and Buskin",
    "Hanging Chad",
    # Chip/mult scaling
    "Blue Joker",
    "Green Joker",
    "Red Card",
    "Spare Trousers",
    "Ride the Bus",
    "Runner",
    "Ice Cream",
    "Constellation",
    "Swashbuckler",
    "Joker Stencil",
    # Utility
    "Chaos the Clown",
}

_GREAT_JOKERS: set[str] = {
    "Steel Joker",
    "Hologram",
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Blackboard",
    "Stencil",
    "Blueprint",
    "Brainstorm",
}


def _joker_value(card: Any) -> int:
    """Rate a joker 0-100 for purchase priority."""
    a = getattr(card, "ability", {})
    if not isinstance(a, dict):
        return 10
    name = a.get("name", "")
    if name in _GREAT_JOKERS:
        return 90
    if name in _GOOD_JOKERS:
        return 60
    # Any joker with x_mult
    if a.get("x_mult", 0) > 1:
        return 70
    # Any joker with decent flat mult
    if a.get("t_mult", 0) >= 4:
        return 40
    if a.get("t_chips", 0) >= 20:
        return 30
    return 15


# ---------------------------------------------------------------------------
# Consumable targeting
# ---------------------------------------------------------------------------


def _pick_consumable_targets(
    card: Any,
    hand: list[Any],
) -> tuple[int, ...] | None:
    """Return target indices for a consumable, or None if no targets needed."""
    from jackdaw.engine.card import _resolve_center

    key = getattr(card, "center_key", "")
    try:
        center = _resolve_center(key)
    except (KeyError, Exception):
        return None

    cfg = center.get("config") or {}
    if isinstance(cfg, list):
        cfg = {}
    max_h = cfg.get("max_highlighted")
    if not max_h or not hand:
        return None

    min_h = cfg.get("min_highlighted", 1)
    n = min(max_h, len(hand))
    if n < min_h:
        return None

    # For enhancement tarots, target the highest-value cards
    # For destruction spectrals, target the lowest-value cards
    effect = cfg.get("mod_conv") or center.get("effect", "")

    if effect in ("destroy",):
        # Target lowest value cards
        ranked = sorted(
            range(len(hand)),
            key=lambda i: hand[i].get_chip_bonus() if hasattr(hand[i], "get_chip_bonus") else 0,
        )
        return tuple(sorted(ranked[:n]))

    # Default: target first N cards (simplest valid choice)
    return tuple(range(n))


# ---------------------------------------------------------------------------
# The smart agent
# ---------------------------------------------------------------------------


def smart_agent(gs: dict[str, Any], legal_actions: list[Any]) -> Any:
    """Smart agent targeting deep runs (ante 8+)."""
    from jackdaw.engine.actions import (
        BuyCard,
        CashOut,
        Discard,
        GamePhase,
        NextRound,
        OpenBooster,
        PickPackCard,
        PlayHand,
        RedeemVoucher,
        Reroll,
        SelectBlind,
        SellCard,
        SkipPack,
        UseConsumable,
    )

    phase = gs.get("phase")
    if isinstance(phase, str):
        phase = GamePhase(phase)

    def _card_set(card: Any) -> str:
        a = getattr(card, "ability", None)
        return a.get("set", "") if isinstance(a, dict) else ""

    # -- BLIND_SELECT --
    if phase == GamePhase.BLIND_SELECT:
        for a in legal_actions:
            if isinstance(a, SelectBlind):
                return a
        return legal_actions[0]

    # -- SELECTING_HAND --
    if phase == GamePhase.SELECTING_HAND:
        hand = gs.get("hand", [])
        jokers = gs.get("jokers", [])
        hand_levels = gs.get("hand_levels")
        cr = gs.get("current_round", {})
        hands_left = cr.get("hands_left", 4)
        discards_left = cr.get("discards_left", 0)

        # Use planet consumables immediately
        for a in legal_actions:
            if isinstance(a, UseConsumable):
                consumables = gs.get("consumables", [])
                if a.card_index < len(consumables):
                    if _card_set(consumables[a.card_index]) == "Planet":
                        return a

        if not hand:
            return legal_actions[0]

        # Evaluate best hand
        best_indices, best_type, best_score = _pick_best_hand_scored(hand, jokers, hand_levels)

        # Decide: discard or play?
        # Only discard when hand is truly bad (High Card) AND we have
        # enough hands+discards to recover.  Being too eager to discard
        # wastes actions and can lose rounds.
        should_discard = False
        if discards_left > 0 and hands_left > 1:
            hand_type_rank = _HAND_PRIORITY.index(best_type) if best_type in _HAND_PRIORITY else 11
            if hand_type_rank >= 11:  # High Card only — discard
                should_discard = True
            elif hand_type_rank >= 10 and hands_left >= 3 and discards_left >= 2:
                # Pair with plenty of room — try for better
                should_discard = True

        if should_discard:
            best_set = set(best_indices)
            worst = [i for i in range(len(hand)) if i not in best_set]
            if worst:
                n = min(len(worst), 5)
                for a in legal_actions:
                    if isinstance(a, Discard) and not a.card_indices:
                        return Discard(card_indices=tuple(sorted(worst[:n])))

        # Play best hand
        for a in legal_actions:
            if isinstance(a, PlayHand) and not a.card_indices and hand:
                return PlayHand(card_indices=tuple(best_indices))

        return legal_actions[0]

    # -- ROUND_EVAL --
    if phase == GamePhase.ROUND_EVAL:
        # Use planet consumables before cashing out
        for a in legal_actions:
            if isinstance(a, UseConsumable):
                consumables = gs.get("consumables", [])
                if a.card_index < len(consumables):
                    if _card_set(consumables[a.card_index]) == "Planet":
                        return a
        for a in legal_actions:
            if isinstance(a, CashOut):
                return a
        return legal_actions[0]

    # -- SHOP --
    if phase == GamePhase.SHOP:
        dollars = gs.get("dollars", 0)
        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        consumables = gs.get("consumables", [])
        consumable_slots = gs.get("consumable_slots", 2)
        shop_cards = gs.get("shop_cards", [])
        cr = gs.get("current_round", {})
        ante = gs.get("ante", 1)

        has_joker_room = len(jokers) < joker_slots
        # Economy: in early game spend aggressively on jokers,
        # then preserve $5 interest floors once we have a build.
        if ante <= 2 and len(jokers) < 3:
            interest_floor = 0  # spend everything to build joker slots
        elif ante <= 3:
            interest_floor = 5
        elif ante <= 5:
            interest_floor = 5  # one interest tier
        else:
            interest_floor = 10  # two interest tiers in late game
        spendable = max(0, dollars - interest_floor)

        buy_actions = [a for a in legal_actions if isinstance(a, BuyCard)]
        reroll_cost = cr.get("reroll_cost", 5)
        free_rerolls = cr.get("free_rerolls", 0)
        times_rerolled = cr.get("times_rerolled", 0)
        need_jokers = has_joker_room and len(jokers) < 4

        # Check if shop has affordable jokers
        shop_has_joker = False
        best_joker_action = None
        best_joker_val = 0
        for a in buy_actions:
            if a.shop_index < len(shop_cards):
                card = shop_cards[a.shop_index]
                if _card_set(card) == "Joker" and card.cost <= dollars:
                    shop_has_joker = True
                    val = _joker_value(card)
                    if val > best_joker_val and card.cost <= spendable:
                        best_joker_val = val
                        best_joker_action = a

        # 1. Use owned planets first (free value)
        for a in legal_actions:
            if isinstance(a, UseConsumable):
                if a.card_index < len(consumables):
                    if _card_set(consumables[a.card_index]) == "Planet":
                        return a

        # 2. Use owned tarots if consumable slots are full
        if len(consumables) >= consumable_slots:
            for a in legal_actions:
                if isinstance(a, UseConsumable):
                    if a.card_index < len(consumables):
                        c = consumables[a.card_index]
                        if _card_set(c) == "Tarot":
                            targets = _pick_consumable_targets(c, gs.get("hand", []))
                            if targets is not None:
                                return UseConsumable(
                                    card_index=a.card_index,
                                    target_indices=targets,
                                )
                            return a

        # 3. Buy jokers — TOP PRIORITY when building roster
        if best_joker_action and need_jokers:
            return best_joker_action
        if best_joker_action and best_joker_val >= 25:
            return best_joker_action

        # 4. Reroll to find jokers when roster is thin
        if need_jokers and not shop_has_joker and len(jokers) < 3:
            if free_rerolls > 0 or (reroll_cost <= spendable and times_rerolled < 3):
                for a in legal_actions:
                    if isinstance(a, Reroll):
                        return a

        # 5. Redeem vouchers (strong persistent upgrades)
        for a in legal_actions:
            if isinstance(a, RedeemVoucher):
                vouchers = gs.get("shop_vouchers", [])
                if a.card_index < len(vouchers):
                    v = vouchers[a.card_index]
                    if v.cost <= spendable + interest_floor:
                        return a

        # 6. Open booster packs (good value, may contain jokers)
        for a in legal_actions:
            if isinstance(a, OpenBooster):
                boosters = gs.get("shop_boosters", [])
                if a.card_index < len(boosters):
                    b = boosters[a.card_index]
                    if b.cost <= spendable:
                        return a

        # 7. Buy planets if we have consumable slots
        if buy_actions and len(consumables) < consumable_slots:
            for a in buy_actions:
                if a.shop_index < len(shop_cards):
                    card = shop_cards[a.shop_index]
                    if _card_set(card) == "Planet" and card.cost <= spendable:
                        return a

        # 8. Buy tarots if we have slots
        if buy_actions and len(consumables) < consumable_slots:
            for a in buy_actions:
                if a.shop_index < len(shop_cards):
                    card = shop_cards[a.shop_index]
                    if _card_set(card) == "Tarot" and card.cost <= spendable:
                        return a

        # 9. Sell weak jokers to make room for better ones (late game)
        if not has_joker_room and ante >= 3:
            great_in_shop = any(
                a.shop_index < len(shop_cards)
                and _card_set(shop_cards[a.shop_index]) == "Joker"
                and _joker_value(shop_cards[a.shop_index]) >= 80
                for a in buy_actions
            )
            if great_in_shop:
                sell_actions = [
                    a for a in legal_actions if isinstance(a, SellCard) and a.area == "jokers"
                ]
                if sell_actions:
                    worst_val = 999
                    worst_action = None
                    for a in sell_actions:
                        if a.card_index < len(jokers):
                            val = _joker_value(jokers[a.card_index])
                            if val < worst_val:
                                worst_val = val
                                worst_action = a
                    if worst_action and worst_val < 50:
                        return worst_action

        # 12. Sell consumables to make room if needed
        if len(consumables) >= consumable_slots:
            for a in legal_actions:
                if isinstance(a, SellCard) and a.area == "consumables":
                    return a

        # 13. Free rerolls
        if free_rerolls > 0:
            for a in legal_actions:
                if isinstance(a, Reroll):
                    return a

        # 14. Leave shop
        for a in legal_actions:
            if isinstance(a, NextRound):
                return a
        return legal_actions[0]

    # -- PACK_OPENING --
    if phase == GamePhase.PACK_OPENING:
        pack_cards = gs.get("pack_cards", [])
        pack_hand = gs.get("hand", [])
        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        consumables = gs.get("consumables", [])
        consumable_slots = gs.get("consumable_slots", 2)
        has_joker_room = len(jokers) < joker_slots
        has_consumable_room = len(consumables) < consumable_slots

        best_pick = None
        best_priority = -1

        for a in legal_actions:
            if not isinstance(a, PickPackCard):
                continue
            if a.card_index >= len(pack_cards):
                continue
            card = pack_cards[a.card_index]
            cset = _card_set(card)

            if cset == "Joker":
                if has_joker_room:
                    p = 40 + _joker_value(card)
                else:
                    p = 1  # Skip jokers if full
            elif cset == "Planet":
                p = 70  # Planets are always good (auto-used or leveled)
            elif cset == "Spectral":
                p = 50 if has_consumable_room else 20
            elif cset == "Tarot":
                p = 30 if has_consumable_room else 10
            else:
                p = 5  # Playing cards — rarely worth taking

            if p > best_priority:
                best_priority = p
                best_pick = a

        if best_pick is not None:
            card = pack_cards[best_pick.card_index]
            targets = _pick_consumable_targets(card, pack_hand)
            if targets is not None:
                return PickPackCard(
                    card_index=best_pick.card_index,
                    target_indices=targets,
                )
            return best_pick

        for a in legal_actions:
            if isinstance(a, SkipPack):
                return a
        return legal_actions[0]

    # Fallback
    return legal_actions[0]
