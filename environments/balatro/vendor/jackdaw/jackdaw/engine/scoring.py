"""Scoring pipeline and eval_card wrapper.

Ports ``eval_card`` from ``common_events.lua:580`` and the scoring
pipeline from ``state_events.lua:571-1065``.

``score_hand_base``: Phases 1-4, 6-8, 12 without joker effects.
``score_hand``: Full pipeline including Phases 5, 7b-d, 8b-c, 9 with jokers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.hand_levels import HandLevels
    from jackdaw.engine.rng import PseudoRandom


def eval_card(
    card: Card,
    context: dict[str, Any] | None = None,
    *,
    rng: PseudoRandom | None = None,
    probabilities_normal: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a single card's scoring contribution.

    Matches ``eval_card`` (common_events.lua:580).

    The return dict contains only fields with non-zero values (matching
    the source's ``if value > 0 then ret.field = value`` pattern).

    Context keys:
        - ``cardarea``: ``"play"`` or ``"hand"`` — determines which
          scoring methods to call.
        - ``repetition_only``: If true, only check for retrigger seals.
        - ``edition``: If true, only return edition info (for joker
          edition pass in Phase 9a).

    For played cards (``cardarea="play"``):
        Returns: chips, mult, x_mult, p_dollars, edition.

    For held cards (``cardarea="hand"``):
        Returns: h_mult, x_mult (from h_x_mult).

    For repetition check (``repetition_only=True``):
        Returns: seals (with repetitions count).

    Args:
        card: The card to evaluate.
        context: Context dict with cardarea and flags.
        rng: PseudoRandom instance for Lucky Card rolls.
        probabilities_normal: ``G.GAME.probabilities.normal`` (default 1).
    """
    ctx = context or {}
    ret: dict[str, Any] = {}

    # Repetition-only mode: just check seals
    if ctx.get("repetition_only"):
        seals = card.calculate_seal(repetition=True)
        if seals:
            ret["seals"] = seals
        return ret

    cardarea = ctx.get("cardarea")

    # Played cards (cardarea == G.play → "play")
    if cardarea == "play":
        chips = card.get_chip_bonus()
        if chips > 0:
            ret["chips"] = chips

        mult = card.get_chip_mult(
            rng=rng,
            probabilities_normal=probabilities_normal,
        )
        if mult > 0:
            ret["mult"] = mult

        x_mult = card.get_chip_x_mult()
        if x_mult > 0:
            ret["x_mult"] = x_mult

        p_dollars = card.get_p_dollars(
            rng=rng,
            probabilities_normal=probabilities_normal,
        )
        if p_dollars > 0:
            ret["p_dollars"] = p_dollars

        # Joker effects on played cards (calculate_joker stub)
        # Will be: jokers = card.calculate_joker(context)
        # if jokers: ret["jokers"] = jokers

        edition = card.get_edition()
        if edition:
            ret["edition"] = edition

    # Held-in-hand cards (cardarea == G.hand → "hand")
    elif cardarea == "hand":
        h_mult = card.get_chip_h_mult()
        if h_mult > 0:
            ret["h_mult"] = h_mult

        # Source maps h_x_mult to ret.x_mult (not ret.h_x_mult)
        h_x_mult = card.get_chip_h_x_mult()
        if h_x_mult > 0:
            ret["x_mult"] = h_x_mult

        # Joker effects on held cards (calculate_joker stub)

    # Joker area (cardarea == G.jokers → "jokers")
    elif cardarea == "jokers":
        if ctx.get("edition"):
            edition = card.get_edition()
            if edition:
                ret["jokers"] = edition
        # else: calculate_joker stub

    return ret


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Result of the full scoring pipeline."""

    hand_type: str
    """Detected hand type (e.g. ``"Full House"``) or ``"NULL"``."""

    scoring_cards: list[Card]
    """Cards that scored (including Splash/Stone augmentation)."""

    chips: float
    """Final chip value (after all per-card and edition bonuses)."""

    mult: float
    """Final mult value (after all additive and multiplicative bonuses)."""

    total: int
    """``floor(chips * mult)`` — the score added to G.GAME.chips."""

    dollars_earned: int
    """Dollars earned from scoring (Gold Seal, Lucky Card, etc.)."""

    debuffed: bool
    """True if the hand was blocked by a boss blind."""

    breakdown: list[str] = field(default_factory=list)
    """Step-by-step log for debugging."""

    jokers_removed: list[Card] = field(default_factory=list)
    """Jokers that self-destructed during scoring (Ice Cream, etc.)."""

    cards_destroyed: list[Card] = field(default_factory=list)
    """Playing cards destroyed during Phase 11 (Glass shatter, etc.)."""

    saved: bool = False
    """True if Mr. Bones (or similar) prevented a game over."""

    joker_creates: list[dict] = field(default_factory=list)
    """Card creation descriptors from joker effects (Vagabond, 8-Ball, etc.)."""


# ---------------------------------------------------------------------------
# Base scoring pipeline (Phases 1-4, 6-8, 12 without joker effects)
# ---------------------------------------------------------------------------


def score_hand_base(
    played_cards: list[Card],
    held_cards: list[Card],
    hand_levels: Any,  # HandLevels
    blind: Any,  # Blind
    rng: PseudoRandom,
    *,
    probabilities_normal: float = 1.0,
    joker_flags: dict[str, bool] | None = None,
) -> ScoreResult:
    """Score a hand without joker effects.

    Implements Phases 1-4, 6-8, 12 of the scoring pipeline from
    ``state_events.lua:571-1065``.
    """
    from jackdaw.engine.hand_eval import evaluate_hand

    _ = joker_flags  # reserved for future joker flag passing
    dollars = 0
    breakdown: list[str] = []

    # === Phase 1-2: Hand detection ===
    eval_result = evaluate_hand(played_cards, jokers=None)
    hand_type = eval_result.detected_hand
    scoring_cards = eval_result.scoring_cards
    poker_hands = eval_result.poker_hands

    if hand_type == "NULL":
        return ScoreResult(
            hand_type="NULL",
            scoring_cards=[],
            chips=0,
            mult=0,
            total=0,
            dollars_earned=0,
            debuffed=False,
            breakdown=["No hand"],
        )

    # === Phase 3: Boss blind debuff check ===
    # Lua passes G.play.cards (all played cards, not just scoring subset)
    # to debuff_hand (state_events.lua:614).  Matters for The Psychic
    # which checks #cards >= h_size_ge against the full played hand.
    debuffed = blind.debuff_hand(played_cards, poker_hands, hand_type)
    if debuffed:
        return ScoreResult(
            hand_type=hand_type,
            scoring_cards=scoring_cards,
            chips=0,
            mult=0,
            total=0,
            dollars_earned=0,
            debuffed=True,
            breakdown=[f"Hand blocked by {blind.name}"],
        )

    # === Phase 3b: The Arm — demote played hand type by 1 (min L1) ===
    if (
        getattr(blind, "name", "") == "The Arm"
        and not getattr(blind, "disabled", False)
        and hand_levels[hand_type].level > 1
    ):
        hand_levels.level_up(hand_type, amount=-1)

    # === Phase 4: Base chips/mult from hand level ===
    base_chips, base_mult = hand_levels.get(hand_type)
    hand_chips = float(base_chips)
    mult = float(base_mult)
    breakdown.append(
        f"Base: {hand_type} L{hand_levels[hand_type].level}"
        f" -> {int(hand_chips)} chips, {int(mult)} mult"
    )

    # Record play
    hand_levels.record_play(hand_type)

    # === Phase 6: Blind modify_hand (The Flint) ===
    new_mult, new_chips, modified = blind.modify_hand(mult, int(hand_chips))
    if modified:
        mult = float(new_mult)
        hand_chips = float(new_chips)
        breakdown.append(f"Blind modify: {int(hand_chips)} chips, {int(mult)} mult")

    # === Phase 7: Per scored card (with retriggers) ===
    for card in scoring_cards:
        if card.debuff:
            continue

        # Collect retriggers
        reps = [1]  # base evaluation
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card,
                {"cardarea": "play"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )

            # Apply effects in source order (state_events.lua:702-776)
            if "chips" in ev:
                hand_chips += ev["chips"]
            if "mult" in ev:
                mult += ev["mult"]
            if "p_dollars" in ev:
                dollars += ev["p_dollars"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]
            if "edition" in ev:
                ed = ev["edition"]
                hand_chips += ed.get("chip_mod", 0)
                mult += ed.get("mult_mod", 0)
                mult *= ed.get("x_mult_mod", 1)

    # === Phase 8: Per held card (with retriggers) ===
    for card in held_cards:
        if card.debuff:
            continue

        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card,
                {"cardarea": "hand"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )

            # Apply in source order (state_events.lua:845-862)
            if "h_mult" in ev:
                mult += ev["h_mult"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]

    # === Phase 12: Final score ===
    total = math.floor(hand_chips * mult)
    breakdown.append(f"Final: {int(hand_chips)} x {mult:.1f} = {total}")

    return ScoreResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        chips=hand_chips,
        mult=mult,
        total=total,
        dollars_earned=dollars,
        debuffed=False,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Full scoring pipeline WITH joker effects (Phases 1-9, 12)
# ---------------------------------------------------------------------------


def _apply_individual_joker_effects(
    effects: list[dict[str, Any]],
    hand_chips: float,
    mult: float,
    dollars: int,
) -> tuple[float, float, int]:
    """Apply a list of individual-context joker results to running totals.

    Matches the effect application order from state_events.lua:704-776:
    chips → mult → p_dollars → dollars → extra → x_mult → edition.
    """
    for eff in effects:
        if "chips" in eff:
            hand_chips += eff["chips"]
        if "mult" in eff:
            mult += eff["mult"]
        if "p_dollars" in eff:
            dollars += eff["p_dollars"]
        if "dollars" in eff:
            dollars += eff["dollars"]
        if "x_mult" in eff:
            mult *= eff["x_mult"]
        if "edition" in eff:
            ed = eff["edition"]
            hand_chips += ed.get("chip_mod", 0)
            mult += ed.get("mult_mod", 0)
            mult *= ed.get("x_mult_mod", 1)
    return hand_chips, mult, dollars


def _apply_held_joker_effects(
    effects: list[dict[str, Any]],
    mult: float,
    dollars: int,
) -> tuple[float, int]:
    """Apply held-card joker effects. Order: dollars → h_mult → x_mult."""
    for eff in effects:
        if "dollars" in eff:
            dollars += eff["dollars"]
        if "h_mult" in eff:
            mult += eff["h_mult"]
        if "x_mult" in eff:
            mult *= eff["x_mult"]
    return mult, dollars


def score_hand(
    played_cards: list[Card],
    held_cards: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind: Blind,
    rng: PseudoRandom,
    *,
    probabilities_normal: float = 1.0,
    game_state: dict[str, Any] | None = None,
    back_key: str | None = None,
    blind_chips: int = 0,
) -> ScoreResult:
    """Full scoring pipeline with joker effects (Phases 1-14).

    Args:
        played_cards: Cards played from hand.
        held_cards: Cards remaining in hand.
        jokers: Joker cards in order (left to right).
        hand_levels: HandLevels instance for base chips/mult.
        blind: Current Blind.
        rng: PseudoRandom instance.
        probabilities_normal: G.GAME.probabilities.normal (default 1).
        game_state: Pre-computed game state dict.
        back_key: Deck back key (e.g. ``'b_plasma'`` for Plasma Deck).
        blind_chips: Blind chip target (for Mr. Bones save check).
    """
    from jackdaw.engine.hand_eval import evaluate_hand
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    gs = game_state or {}
    dollars = 0
    breakdown: list[str] = []

    # Pre-compute derived values
    joker_count = sum(1 for j in jokers if j.ability.get("set") == "Joker")
    if joker_count == 0:
        joker_count = len(jokers)  # fallback: count all

    # Check for meta-jokers
    smeared = any(j.ability.get("name") == "Smeared Joker" and not j.debuff for j in jokers)
    pareidolia = any(j.ability.get("name") == "Pareidolia" and not j.debuff for j in jokers)

    # Build GameSnapshot once — shared across all JokerContext instances
    snapshot = GameSnapshot(
        joker_count=joker_count,
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("money", 0),
        deck_cards_remaining=gs.get("deck_cards_remaining", 0),
        starting_deck_size=gs.get("starting_deck_size", 52),
        playing_cards_count=gs.get("playing_cards_count", 52),
        stone_tally=gs.get("stone_tally", 0),
        steel_tally=gs.get("steel_tally", 0),
        enhanced_card_count=gs.get("enhanced_card_count", 0),
        hands_left=gs.get("hands_left", 0),
        hands_played=gs.get("current_round_hands_played", 0),
        discards_left=gs.get("discards_left", 0),
        discards_used=gs.get("discards_used", 0),
        probabilities_normal=probabilities_normal,
        consumable_usage_tarot=gs.get("consumable_usage_tarot", 0),
        mail_card_id=gs.get("mail_card_id"),
        idol_card=gs.get("idol_card"),
        ancient_suit=gs.get("ancient_suit"),
    )

    # === Phase 1-2: Hand detection ===
    eval_result = evaluate_hand(played_cards, jokers=None)
    hand_type = eval_result.detected_hand
    scoring_cards = eval_result.scoring_cards
    poker_hands = eval_result.poker_hands

    if hand_type == "NULL":
        return ScoreResult(
            hand_type="NULL",
            scoring_cards=[],
            chips=0,
            mult=0,
            total=0,
            dollars_earned=0,
            debuffed=False,
            breakdown=["No hand"],
        )

    # === Phase 3: Boss blind debuff check ===
    # Lua passes G.play.cards (all played cards, not just scoring subset)
    # to debuff_hand (state_events.lua:614).  Matters for The Psychic
    # which checks #cards >= h_size_ge against the full played hand.
    debuffed = blind.debuff_hand(played_cards, poker_hands, hand_type)
    if debuffed:
        # Phase 3a: Matador check on debuffed hands
        for joker in jokers:
            if joker.debuff:
                continue
            ctx = JokerContext(
                debuffed_hand=True,
                blind=blind,
                jokers=jokers,
                full_hand=played_cards,
                scoring_hand=scoring_cards,
                scoring_name=hand_type,
                poker_hands=poker_hands,
            )
            result = calculate_joker(joker, ctx)
            if result and result.dollars:
                dollars += result.dollars

        return ScoreResult(
            hand_type=hand_type,
            scoring_cards=scoring_cards,
            chips=0,
            mult=0,
            total=0,
            dollars_earned=dollars,
            debuffed=True,
            breakdown=[f"Hand blocked by {blind.name}"],
        )

    # === Phase 3b: The Arm — demote played hand type by 1 (min L1) ===
    if (
        getattr(blind, "name", "") == "The Arm"
        and not getattr(blind, "disabled", False)
        and hand_levels[hand_type].level > 1
    ):
        hand_levels.level_up(hand_type, amount=-1)

    # === Phase 3c: Splash — all played cards score ===
    splash_active = any(
        getattr(j, "center_key", None) == "j_splash" and not getattr(j, "debuff", False)
        for j in jokers
    )
    if splash_active:
        scoring_cards = list(played_cards)

    # === Phase 4: Base chips/mult from hand level ===
    base_chips, base_mult = hand_levels.get(hand_type)
    hand_chips = float(base_chips)
    mult = float(base_mult)
    breakdown.append(
        f"Base: {hand_type} L{hand_levels[hand_type].level}"
        f" -> {int(hand_chips)} chips, {int(mult)} mult"
    )

    # Record play
    hand_levels.record_play(hand_type)

    # Shared context fields (lightweight — references snapshot, not copies)
    _shared = dict(
        full_hand=played_cards,
        scoring_hand=scoring_cards,
        scoring_name=hand_type,
        poker_hands=poker_hands,
        jokers=jokers,
        rng=rng,
        smeared=smeared,
        pareidolia=pareidolia,
        hand_levels=hand_levels,
        blind=blind,
        held_cards=held_cards,
        game=snapshot,
    )

    # === Phase 5: "before" joker pass ===
    for joker in jokers:
        if joker.debuff:
            continue
        ctx = JokerContext(before=True, **_shared)
        result = calculate_joker(joker, ctx)
        if result and result.level_up:
            hand_levels.level_up(hand_type)
            base_chips, base_mult = hand_levels.get(hand_type)
            hand_chips = float(base_chips)
            mult = float(base_mult)

    # === Phase 6: Blind modify_hand (The Flint) ===
    new_mult, new_chips, modified = blind.modify_hand(mult, int(hand_chips))
    if modified:
        mult = float(new_mult)
        hand_chips = float(new_chips)
        breakdown.append(f"Blind modify: {int(hand_chips)} chips, {int(mult)} mult")

    # === Phase 7: Per scored card (with retriggers) ===
    for card in scoring_cards:
        if card.debuff:
            continue

        # 7a: Collect retriggers (seal + joker)
        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for joker in jokers:
            if joker.debuff:
                continue
            rep_ctx = JokerContext(
                repetition=True,
                cardarea="play",
                other_card=card,
                **_shared,
            )
            rep_result = calculate_joker(joker, rep_ctx)
            if rep_result and rep_result.repetitions > 0:
                for _ in range(rep_result.repetitions):
                    reps.append(rep_result)

        # 7b-d: Each repetition
        for _rep in reps:
            # Card's own effects
            ev = eval_card(
                card,
                {"cardarea": "play"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )
            effects: list[dict[str, Any]] = [ev]

            # Joker individual effects on this card
            for joker in jokers:
                if joker.debuff:
                    continue
                ind_ctx = JokerContext(
                    individual=True,
                    cardarea="play",
                    other_card=card,
                    **_shared,
                )
                ind_result = calculate_joker(joker, ind_ctx)
                if ind_result:
                    eff: dict[str, Any] = {}
                    if ind_result.chips:
                        eff["chips"] = ind_result.chips
                    if ind_result.mult:
                        eff["mult"] = ind_result.mult
                    if ind_result.x_mult:
                        eff["x_mult"] = ind_result.x_mult
                    if ind_result.dollars:
                        eff["dollars"] = ind_result.dollars
                    if eff:
                        eff["card"] = joker
                        effects.append(eff)

            hand_chips, mult, dollars = _apply_individual_joker_effects(
                effects,
                hand_chips,
                mult,
                dollars,
            )

    # === Phase 8: Per held card (with retriggers) ===
    for card in held_cards:
        if card.debuff:
            continue

        # 8a: Collect retriggers
        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for joker in jokers:
            if joker.debuff:
                continue
            rep_ctx = JokerContext(
                repetition=True,
                cardarea="hand",
                other_card=card,
                **_shared,
            )
            rep_result = calculate_joker(joker, rep_ctx)
            if rep_result and rep_result.repetitions > 0:
                for _ in range(rep_result.repetitions):
                    reps.append(rep_result)

        # 8b-c: Each repetition
        for _rep in reps:
            ev = eval_card(
                card,
                {"cardarea": "hand"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )
            effects_h: list[dict[str, Any]] = [ev]

            # Joker individual effects on held card
            for joker in jokers:
                if joker.debuff:
                    continue
                ind_ctx = JokerContext(
                    individual=True,
                    cardarea="hand",
                    other_card=card,
                    **_shared,
                )
                ind_result = calculate_joker(joker, ind_ctx)
                if ind_result:
                    eff_h: dict[str, Any] = {}
                    if ind_result.h_mult:
                        eff_h["h_mult"] = ind_result.h_mult
                    if ind_result.x_mult:
                        eff_h["x_mult"] = ind_result.x_mult
                    if ind_result.dollars:
                        eff_h["dollars"] = ind_result.dollars
                    if eff_h:
                        effects_h.append(eff_h)

            mult, dollars = _apply_held_joker_effects(
                effects_h,
                mult,
                dollars,
            )

    # === Phase 8d: individual_hand_end (Vampire strip, Obelisk check) ===
    for joker in jokers:
        if joker.debuff:
            continue
        ihe_ctx = JokerContext(individual_hand_end=True, **_shared)
        ihe_result = calculate_joker(joker, ihe_ctx)
        if ihe_result:
            if ihe_result.Xmult_mod:
                mult *= ihe_result.Xmult_mod

    # === Phase 9: Joker main effects (left to right) ===
    joker_creates: list[dict] = []
    for joker in jokers:
        if joker.debuff:
            continue

        # 9a: Edition additive (chip_mod, mult_mod) BEFORE joker effect
        edition = joker.get_edition()
        if edition:
            hand_chips += edition.get("chip_mod", 0)
            mult += edition.get("mult_mod", 0)

        # 9b: Main joker effect
        main_ctx = JokerContext(joker_main=True, **_shared)
        result = calculate_joker(joker, main_ctx)
        if result:
            if result.mult_mod:
                mult += result.mult_mod
            if result.chip_mod:
                hand_chips += result.chip_mod
            if result.Xmult_mod:
                mult *= result.Xmult_mod
            if result.dollars:
                dollars += result.dollars
            if result.extra and "create" in result.extra:
                joker_creates.append(result.extra["create"])

        # 9c: Joker-on-joker (other_joker context)
        for other in jokers:
            if other is joker or other.debuff:
                continue
            j2j_ctx = JokerContext(other_joker=joker, **_shared)
            j2j = calculate_joker(other, j2j_ctx)
            if j2j:
                if j2j.mult_mod:
                    mult += j2j.mult_mod
                if j2j.chip_mod:
                    hand_chips += j2j.chip_mod
                if j2j.Xmult_mod:
                    mult *= j2j.Xmult_mod

        # 9d: Edition multiplicative (x_mult_mod) AFTER joker effect
        if edition and "x_mult_mod" in edition:
            mult *= edition["x_mult_mod"]

    # === Phase 10: Back trigger (final_scoring_step) ===
    if back_key:
        from jackdaw.engine.back import Back as _Back

        _back_effect = _Back(back_key).trigger_effect(
            "final_scoring_step", chips=hand_chips, mult=mult
        )
        if _back_effect:
            prev_chips, prev_mult = hand_chips, mult
            hand_chips = _back_effect["chips"]
            mult = _back_effect["mult"]
            breakdown.append(
                f"Plasma: ({int(prev_chips + prev_mult)}) / 2"
                f" -> {int(hand_chips)} chips, {int(mult)} mult"
            )

    # === Phase 11: Card destruction ===
    cards_destroyed: list[Card] = []
    for sc in scoring_cards:
        if sc.debuff:
            continue
        destroyed = False
        # Joker destroying_card checks (Sixth Sense, etc.)
        for joker in jokers:
            if joker.debuff:
                continue
            dest_ctx = JokerContext(
                destroying_card=sc,
                **_shared,
            )
            dest_result = calculate_joker(joker, dest_ctx)
            if dest_result and dest_result.remove:
                destroyed = True
                break
        # Glass Card self-shatter: 1 in (1/probabilities_normal * 4) chance
        if not destroyed and sc.ability.get("name") == "Glass Card":
            if rng.random("glass") < probabilities_normal / 4:
                destroyed = True
        if destroyed:
            cards_destroyed.append(sc)

    # Notify jokers of destruction (Caino, Glass Joker xMult growth)
    if cards_destroyed:
        for joker in jokers:
            if joker.debuff:
                continue
            dest_notify_ctx = JokerContext(
                cards_destroyed=cards_destroyed,
                **_shared,
            )
            calculate_joker(joker, dest_notify_ctx)

    # === Phase 12: Final score ===
    total = math.floor(hand_chips * mult)
    breakdown.append(f"Final: {int(hand_chips)} x {mult:.1f} = {total}")

    # === Phase 13: "after" joker pass (scaling mutations) ===
    jokers_removed: list[Card] = []
    for joker in jokers:
        if joker.debuff:
            continue
        after_ctx = JokerContext(after=True, **_shared)
        after_result = calculate_joker(joker, after_ctx)
        if after_result and after_result.remove:
            jokers_removed.append(joker)

    # Mr. Bones save check: if score < blind target and last hand
    saved = False
    if blind_chips > 0 and total < blind_chips and gs.get("hands_left", 0) == 0:
        for joker in jokers:
            if joker.debuff:
                continue
            bones_ctx = JokerContext(**_shared)
            bones_ctx.game_over = True  # type: ignore[attr-defined]
            bones_result = calculate_joker(joker, bones_ctx)
            if bones_result and bones_result.saved:
                saved = True
                if bones_result.remove:
                    jokers_removed.append(joker)
                break

    # === Phase 14: Post-play modifiers (debuff played cards) ===
    # Challenge mode: debuff all played cards after scoring.
    # Implemented as a flag check — no joker interaction.

    return ScoreResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        chips=hand_chips,
        mult=mult,
        total=total,
        dollars_earned=dollars,
        debuffed=False,
        breakdown=breakdown,
        jokers_removed=jokers_removed,
        cards_destroyed=cards_destroyed,
        saved=saved,
        joker_creates=joker_creates,
    )
