"""Game step function — the heart of the simulator.

Applies a player :data:`~jackdaw.engine.actions.Action` to the game state
and advances the phase.  Each handler validates legality, executes the
action, and transitions to the next phase as needed.

Usage::

    from jackdaw.engine.game import step
    from jackdaw.engine.actions import SelectBlind

    game_state = initialize_run("b_red", 1, "SEED")
    game_state["phase"] = GamePhase.BLIND_SELECT
    game_state["blind_on_deck"] = "Small"

    game_state = step(game_state, SelectBlind())
    assert game_state["phase"] == GamePhase.SELECTING_HAND
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import (
    Action,
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
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)


class IllegalActionError(Exception):
    """Raised when an action is not valid in the current game state."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def step(game_state: dict[str, Any], action: Action) -> dict[str, Any]:
    """Apply *action* to *game_state* in-place and return it.

    Dispatches based on action type.  Raises :class:`IllegalActionError`
    if the action is not valid in the current phase.
    """
    match action:
        case SelectBlind():
            return _handle_select_blind(game_state)
        case SkipBlind():
            return _handle_skip_blind(game_state)
        case PlayHand(card_indices=indices):
            return _handle_play_hand(game_state, indices)
        case Discard(card_indices=indices):
            return _handle_discard(game_state, indices)
        case CashOut():
            return _handle_cash_out(game_state)
        case BuyCard(shop_index=idx):
            return _handle_buy_card(game_state, idx)
        case SellCard(area=area, card_index=idx):
            return _handle_sell_card(game_state, area, idx)
        case UseConsumable(card_index=idx, target_indices=targets):
            return _handle_use_consumable(game_state, idx, targets)
        case RedeemVoucher(card_index=idx):
            return _handle_redeem_voucher(game_state, idx)
        case OpenBooster(card_index=idx):
            return _handle_open_booster(game_state, idx)
        case PickPackCard(card_index=idx, target_indices=targets):
            return _handle_pick_pack_card(game_state, idx, targets)
        case SkipPack():
            return _handle_skip_pack(game_state)
        case Reroll():
            return _handle_reroll(game_state)
        case NextRound():
            return _handle_next_round(game_state)
        case SortHand(mode=mode):
            return _handle_sort_hand(game_state, mode)
        case SwapHandLeft(idx=idx):
            return _handle_swap_hand(game_state, idx, -1)
        case SwapHandRight(idx=idx):
            return _handle_swap_hand(game_state, idx, +1)
        case SwapJokersLeft(idx=idx):
            return _handle_swap_jokers(game_state, idx, -1)
        case SwapJokersRight(idx=idx):
            return _handle_swap_jokers(game_state, idx, +1)
        case _:
            raise IllegalActionError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Phase validation helper
# ---------------------------------------------------------------------------


def _require_phase(gs: dict[str, Any], *phases: GamePhase) -> GamePhase:
    """Assert the current phase is one of *phases* and return it."""
    raw = gs.get("phase")
    phase = GamePhase(raw) if isinstance(raw, str) else raw
    if phase not in phases:
        raise IllegalActionError(f"Action not valid in phase {phase!r} (expected {phases})")
    return phase


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_select_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept the current blind and start the round.

    Full sequence matching ``game.lua`` select_blind → ``blind.lua``
    ``set_blind`` → ``state_events.lua`` ``new_round``:

    1. Create Blind from blind_choices
    2. Fire joker ``setting_blind`` context (Chicot, Madness, Burglar,
       Marble Joker, Riff-raff, Cartomancer)
    3. Process setting_blind side-effects
    4. Apply boss blind set-time effects (Water, Needle, Manacle,
       Amber Acorn) + debuff playing cards
    5. Call ``start_round`` (reset counters, targeting cards)
    6. Draw hand from deck
    7. Set phase → SELECTING_HAND
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    from jackdaw.engine.blind import Blind
    from jackdaw.engine.run_init import start_round

    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr = gs["round_resets"]
    blind_key = rr["blind_choices"].get(blind_on_deck, "bl_small")

    # ------------------------------------------------------------------
    # 1. Create the active Blind
    # ------------------------------------------------------------------
    ante = rr["ante"]
    scaling = gs.get("modifiers", {}).get("scaling", 1)
    ante_scaling = gs["starting_params"].get("ante_scaling", 1.0)
    no_reward = gs.get("modifiers", {}).get("no_blind_reward", {})
    blind = Blind.create(
        blind_key,
        ante,
        scaling=scaling,
        ante_scaling=ante_scaling,
        no_blind_reward=bool(no_reward.get(blind_on_deck)),
    )
    gs["blind"] = blind
    gs["chips"] = 0
    rr["blind_states"][blind_on_deck] = "Current"
    rr["blind"] = blind

    # ------------------------------------------------------------------
    # 2. Fire joker setting_blind context
    # ------------------------------------------------------------------
    jokers: list = gs.get("jokers", [])
    setting_mutations = _fire_setting_blind(gs, jokers, blind)

    # ------------------------------------------------------------------
    # 3. Process setting_blind side-effects
    # ------------------------------------------------------------------
    _apply_setting_blind_mutations(gs, setting_mutations, jokers)

    # ------------------------------------------------------------------
    # 4. Start round (reset counters, targeting cards)
    #    Must run BEFORE boss effects so that The Water/Needle/etc.
    #    can decrement from the freshly-set values.
    # ------------------------------------------------------------------
    start_round(gs)

    # ------------------------------------------------------------------
    # 5. Boss blind set-time effects (blind.lua:157-209)
    #    In Lua, set_blind fires inside new_round BEFORE the shuffle.
    #    Order: set_blind → joker setting_blind → shuffle → draw.
    # ------------------------------------------------------------------
    if blind.boss and not blind.disabled:
        _apply_boss_blind_effects(gs, blind)

    # Debuff playing cards based on boss blind
    deck: list = gs.get("deck", [])
    pareidolia = any(
        getattr(j, "center_key", None) == "j_pareidolia" and not getattr(j, "debuff", False)
        for j in jokers
    )
    for card in deck:
        blind.debuff_card(card, pareidolia=pareidolia)

    # ------------------------------------------------------------------
    # 6. Per-round deck shuffle (state_events.lua:344)
    #    Fires AFTER set_blind and joker setting_blind context,
    #    BEFORE draw_to_hand.  Key: 'nr' + str(ante).
    # ------------------------------------------------------------------
    rng = gs.get("rng")
    if rng:
        deck_list: list = gs.get("deck", [])
        nr_seed = rng.seed("nr" + str(ante))
        rng.shuffle(deck_list, nr_seed)

    # ------------------------------------------------------------------
    # 7. Draw hand from deck
    # ------------------------------------------------------------------
    _draw_hand(gs)
    # Debuff hand cards too (they were drawn from the deck)
    for card in gs.get("hand", []):
        blind.debuff_card(card, pareidolia=pareidolia)

    # ------------------------------------------------------------------
    # 7b. Boss drawn_to_hand effects (Cerulean Bell, Crimson Heart)
    # ------------------------------------------------------------------
    if blind.boss and not blind.disabled:
        dth = blind.drawn_to_hand(
            hand_cards=gs.get("hand", []),
            joker_cards=gs.get("jokers"),
            rng=rng,
        )
        if dth.get("forced_card_index") is not None:
            hand = gs.get("hand", [])
            idx = dth["forced_card_index"]
            if 0 <= idx < len(hand):
                hand[idx].ability["forced_selection"] = True

    # ------------------------------------------------------------------
    # 8. Phase → SELECTING_HAND
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.SELECTING_HAND
    return gs


def _fire_new_blind_choice_tags(gs: dict[str, Any]) -> None:
    """Fire new_blind_choice context for any awarded tags that need it.

    In Lua, new_blind_choice tags fire when entering the blind select screen
    after a skip.  Pack-creating tags (buffoon, charm, ethereal, meteor,
    standard) open a pack for the player to pick from.  The boss tag rerolls
    the boss blind.

    Since the engine has no interactive player, pack-creating tags populate
    ``gs["pack_cards"]`` and set the phase to PACK_OPENING so the caller
    (or agent) can pick or skip.
    """
    from jackdaw.engine.tags import Tag

    awarded: list[dict] = gs.get("awarded_tags", [])
    rng = gs.get("rng")
    rr = gs.get("round_resets", {})

    for entry in awarded:
        tag_key = entry.get("key", "")
        # Only fire tags that haven't been processed for new_blind_choice yet
        if entry.get("nbc_fired"):
            continue

        tag = Tag(tag_key)
        result = tag.apply("new_blind_choice", gs, rng=rng)
        entry["nbc_fired"] = True

        if result is None:
            continue

        if result.reroll_boss:
            from jackdaw.engine.blind import get_new_boss

            bosses_used = gs.setdefault("bosses_used", {})
            ante = rr.get("ante", 1)
            new_boss = get_new_boss(ante, bosses_used, rng)
            rr.setdefault("blind_choices", {})["Boss"] = new_boss

        if result.create_pack:
            _open_tag_pack(gs, result.create_pack)


def _open_tag_pack(gs: dict[str, Any], pack_key: str) -> None:
    """Open a pack from a tag reward, populating pack_cards.

    The caller (or agent) must then pick from the pack or skip it.
    For the engine-only path (no interactive player), we store the pack
    state so that get_legal_actions returns PickPackCard/SkipPack options.
    """
    from jackdaw.engine.data.prototypes import BOOSTERS
    from jackdaw.engine.packs import generate_pack_cards

    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)

    pack_cards, choices = generate_pack_cards(pack_key, rng, ante, gs)
    gs["pack_cards"] = pack_cards
    gs["pack_choices_remaining"] = choices

    # Determine pack kind from prototype
    proto = BOOSTERS.get(pack_key)
    pack_kind = proto.kind if proto else ""
    gs["pack_type"] = pack_kind

    # Save current phase and switch to pack opening
    gs["shop_return_phase"] = gs.get("phase", GamePhase.BLIND_SELECT)

    # For Arcana/Spectral packs: deal hand from deck for targeting
    # Matches _handle_open_booster behavior
    if pack_kind in ("Arcana", "Spectral"):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        hand_size = gs.get("hand_size", 8)
        to_deal = min(len(deck), hand_size - len(hand))
        pack_hand: list = []
        for _ in range(to_deal):
            if deck:
                card = deck.pop()
                pack_hand.append(card)
        gs["pack_hand"] = pack_hand
        combined_hand = hand + pack_hand
        _sort_hand_desc(combined_hand)
        gs["hand"] = combined_hand

    gs["phase"] = GamePhase.PACK_OPENING


def _apply_tag_result(gs: dict[str, Any], result: Any) -> None:
    """Apply a TagResult's effects to the game state.

    Handles all TagResult fields that produce immediate side-effects.
    """
    if result.dollars:
        gs["dollars"] = gs.get("dollars", 0) + result.dollars

    if result.create_jokers:
        from jackdaw.engine.card_factory import create_card

        jokers = gs.get("jokers", [])
        joker_slots = gs.get("joker_slots", 5)
        rng = gs.get("rng")
        ante = gs.get("round_resets", {}).get("ante", 1)
        for _ in range(result.create_jokers):
            if len(jokers) >= joker_slots:
                break
            # Top-up Tag: create_card('Joker', G.jokers, nil, 0, nil, nil,
            # nil, 'top') — forced Common, append 'top' (tag.lua:138)
            card = create_card(
                "Joker",
                rng,
                ante,
                area="",
                soulable=False,
                forced_rarity=1,
                append="top",
                game_state=gs,
            )
            jokers.append(card)

    if result.level_up is not None:
        hand_type, levels = result.level_up
        hand_levels = gs.get("hand_levels")
        if hand_levels is not None:
            for _ in range(levels):
                hand_levels.level_up(hand_type)


def _handle_skip_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip the current blind (Small or Big) and advance.

    Full sequence matching ``button_callbacks.lua:2740-2775``:

    1. Validate: not Boss
    2. Increment skips
    3. Award skip tag from ``blind_tags``
    4. Fire tag ``apply('immediate')`` for immediate tags
    5. Fire joker ``skip_blind`` context (Throwback tracking)
    6. Advance ``blind_on_deck``: Small→Big, Big→Boss
    7. Check Double Tag: if active, duplicate the just-awarded tag
    8. Phase stays BLIND_SELECT
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    blind_on_deck = gs.get("blind_on_deck", "Small")
    if blind_on_deck not in ("Small", "Big"):
        raise IllegalActionError("Cannot skip Boss blind")

    rr = gs["round_resets"]

    # ------------------------------------------------------------------
    # 1-2. Increment skips
    # ------------------------------------------------------------------
    gs["skips"] = gs.get("skips", 0) + 1
    rr["blind_states"][blind_on_deck] = "Skipped"

    # ------------------------------------------------------------------
    # 3. Award skip tag
    # ------------------------------------------------------------------
    blind_tags = rr.get("blind_tags", {})
    tag_key = blind_tags.get(blind_on_deck)
    awarded_tags: list[dict[str, Any]] = gs.setdefault("awarded_tags", [])

    if tag_key:
        from jackdaw.engine.tags import Tag

        tag = Tag(tag_key)
        tag_result = tag.apply("immediate", gs, rng=gs.get("rng"))

        awarded_tags.append(
            {
                "key": tag_key,
                "result": tag_result,
                "blind": blind_on_deck,
            }
        )

        # Apply immediate tag effects
        if tag_result is not None:
            _apply_tag_result(gs, tag_result)

    # ------------------------------------------------------------------
    # 4. Fire joker skip_blind context
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    game_snap = GameSnapshot(
        joker_count=len(jokers),
        money=gs.get("dollars", 0),
        skips=gs.get("skips", 0),
    )
    for joker in jokers:
        if not getattr(joker, "debuff", False):
            ctx = JokerContext(
                skip_blind=True,
                jokers=jokers,
                game=game_snap,
            )
            calculate_joker(joker, ctx)

    # ------------------------------------------------------------------
    # 5. Advance blind_on_deck
    # ------------------------------------------------------------------
    if blind_on_deck == "Small":
        gs["blind_on_deck"] = "Big"
        rr["blind_states"]["Big"] = "Select"
    else:
        gs["blind_on_deck"] = "Boss"
        rr["blind_states"]["Boss"] = "Select"

    # ------------------------------------------------------------------
    # 6. Double Tag check
    # ------------------------------------------------------------------
    if tag_key:
        _check_double_tag(gs, tag_key)

    # ------------------------------------------------------------------
    # 7. Fire new_blind_choice tags from awarded (deferred) tags
    # ------------------------------------------------------------------
    # In Lua, new_blind_choice tags fire when the blind select screen
    # appears after a skip.  Tags like tag_buffoon, tag_charm, etc.
    # open packs; tag_boss rerolls the boss blind.
    _fire_new_blind_choice_tags(gs)

    # Only return to BLIND_SELECT if a tag didn't open a pack
    if gs.get("phase") != GamePhase.PACK_OPENING:
        gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_play_hand(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Play cards from the hand, score them, and check if blind is beaten.

    Full sequence matching ``state_events.lua`` play_cards_from_highlighted
    → evaluate_play:

    1. Validate indices
    2. Move cards from hand to play area (preserve index order)
    3. Decrement hands_left, increment hands_played
    4. Update per-card stats (times_played, played_this_ante)
    5. Fire ``Blind:press_play`` (The Hook, The Tooth)
    6. Call ``score_hand`` (full 14-phase pipeline)
    7. Process scoring side-effects (dollars, card destruction,
       joker removal)
    8. Move surviving played cards to discard pile
    9. Record hand type in hand_levels
    10. Determine next phase: won / continue / game over
    11. If continuing: draw cards, re-debuff for boss
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["hands_left"] <= 0:
        raise IllegalActionError("No hands remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot play more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")

    # ------------------------------------------------------------------
    # 2. Move cards from hand to play area
    # ------------------------------------------------------------------
    # Preserve SELECTION ORDER (not hand position order).
    # In Balatro, cards are placed left-to-right in click order.
    # The first index in card_indices is the leftmost scored card.
    idx_set = set(indices)
    played = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in idx_set]
    gs["hand"] = held

    # ------------------------------------------------------------------
    # 3. Decrement hands_left, increment hands_played
    # ------------------------------------------------------------------
    cr["hands_left"] -= 1
    cr["hands_played"] += 1
    gs["hands_played"] = gs.get("hands_played", 0) + 1

    # ------------------------------------------------------------------
    # 4. Per-card stats
    # ------------------------------------------------------------------
    for card in played:
        base = getattr(card, "base", None)
        if base is not None:
            base.times_played = getattr(base, "times_played", 0) + 1
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict):
            ability["played_this_ante"] = True

    # ------------------------------------------------------------------
    # 5. Blind:press_play (blind.lua:464)
    # ------------------------------------------------------------------
    blind = gs["blind"]
    rng = gs.get("rng")
    _press_play(gs, blind, played, rng)

    # ------------------------------------------------------------------
    # 6. Score the hand (full 14-phase pipeline)
    # ------------------------------------------------------------------
    from jackdaw.engine.scoring import score_hand

    jokers = gs.get("jokers", [])
    hand_levels = gs.get("hand_levels")

    # Populate snapshot values that score_hand reads from game_state.
    # These are stored in nested structures but score_hand expects them
    # at the top level.
    gs["hands_left"] = cr.get("hands_left", 0)
    gs["current_round_hands_played"] = cr.get("hands_played", 0)
    gs["discards_left"] = cr.get("discards_left", 0)
    gs["discards_used"] = cr.get("discards_used", 0)
    gs["money"] = gs.get("dollars", 0)
    gs["deck_cards_remaining"] = len(gs.get("deck", []))

    # Card tallies for jokers that reference full-deck counts
    all_cards = gs.get("deck", []) + gs.get("hand", []) + gs.get("discard_pile", []) + played
    gs["playing_cards_count"] = len(all_cards)
    gs["stone_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_stone")
    gs["steel_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_steel")
    gs["enhanced_card_count"] = sum(
        1 for c in all_cards if getattr(c, "center_key", "") not in ("", "c_base")
    )

    # Targeting card values (from current_round)
    gs["mail_card_id"] = cr.get("mail_card", {}).get("id")
    gs["idol_card"] = cr.get("idol_card")
    gs["ancient_suit"] = cr.get("ancient_card", {}).get("suit")

    # Consumable usage tally
    usage = gs.get("consumable_usage_total", {})
    gs["consumable_usage_tarot"] = usage.get("tarot", 0)

    result = score_hand(
        played_cards=played,
        held_cards=held,
        jokers=jokers,
        hand_levels=hand_levels,
        blind=blind,
        rng=rng,
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        game_state=gs,
        back_key=gs.get("selected_back_key"),
        blind_chips=blind.chips,
    )

    # ------------------------------------------------------------------
    # 7. Process scoring side-effects
    # ------------------------------------------------------------------
    # Accumulate chips
    gs["chips"] = gs.get("chips", 0) + result.total
    gs["last_score_result"] = result

    # Dollars from scoring (Gold Seal, Lucky Card, joker economy)
    if result.dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + result.dollars_earned

    # Joker self-destruction (Ice Cream, Popcorn, etc.)
    for removed in result.jokers_removed:
        if removed in jokers:
            jokers.remove(removed)

    # Playing card destruction (Glass shatter, etc.)
    destroyed_set = set(id(c) for c in result.cards_destroyed)
    played = [c for c in played if id(c) not in destroyed_set]

    # Joker card creation (Vagabond, 8-Ball, Superposition, etc.)
    if result.joker_creates:
        _resolve_create_descriptors(
            gs, [{"type": c.get("type", "Tarot"), **c} for c in result.joker_creates]
        )

    # The Ox: set money to $0 if most-played hand type is played
    # (blind.lua:debuff_hand fires during scoring in Lua)
    if (
        getattr(blind, "name", "") == "The Ox"
        and not getattr(blind, "disabled", False)
        and hand_levels is not None
        and result.hand_type != "NULL"
    ):
        from jackdaw.engine.data.hands import HandType as _HT

        try:
            played_ht = _HT(result.hand_type)
            if played_ht == hand_levels.most_played():
                blind.triggered = True
                gs["dollars"] = 0
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # 8. Move surviving played cards to discard pile
    #
    # In Lua, draw_from_play_to_discard (state_events.lua:522, 1088-1096)
    # moves played cards to the discard area after scoring.
    # ------------------------------------------------------------------
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(played)

    # ------------------------------------------------------------------
    # 9. Record hand type
    # ------------------------------------------------------------------
    if hand_levels is not None and result.hand_type != "NULL":
        hand_levels.record_play(result.hand_type)

    # ------------------------------------------------------------------
    # 10. Determine next phase
    # ------------------------------------------------------------------
    if gs["chips"] >= blind.chips:
        _round_won(gs)
    elif cr["hands_left"] <= 0:
        if not result.saved:
            gs["phase"] = GamePhase.GAME_OVER
            gs["won"] = False
        else:
            _round_won(gs)
    else:
        # ------------------------------------------------------------------
        # 11. More hands — draw cards and stay in SELECTING_HAND
        # ------------------------------------------------------------------
        # The Serpent: draw only 3 cards instead of filling to hand_size
        serpent_play = getattr(blind, "name", "") == "The Serpent" and not getattr(
            blind, "disabled", False
        )
        if serpent_play:
            deck: list = gs.get("deck", [])
            hand_out: list = gs.get("hand", [])
            for _ in range(min(3, len(deck))):
                if deck:
                    hand_out.append(deck.pop())
            _sort_hand_desc(hand_out)
        else:
            _draw_hand(gs)

        # Re-debuff hand cards for boss blind (new cards from deck)
        if blind.boss and not blind.disabled:
            pareidolia = any(
                getattr(j, "center_key", None) == "j_pareidolia" and not getattr(j, "debuff", False)
                for j in jokers
            )
            for card in gs.get("hand", []):
                blind.debuff_card(card, pareidolia=pareidolia)

        # The Fish: flip newly drawn cards face-down
        if getattr(blind, "name", "") == "The Fish" and getattr(blind, "prepped", False):
            for card in gs.get("hand", []):
                card.facing = "back"

        # Boss drawn_to_hand effects on redraw (Cerulean Bell, Crimson Heart)
        if blind.boss and not blind.disabled:
            dth = blind.drawn_to_hand(
                hand_cards=gs.get("hand", []),
                joker_cards=jokers,
                rng=rng,
            )
            if dth.get("forced_card_index") is not None:
                hand = gs.get("hand", [])
                idx = dth["forced_card_index"]
                if 0 <= idx < len(hand):
                    hand[idx].ability["forced_selection"] = True

    return gs


def _handle_discard(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Discard highlighted cards, fire joker contexts, draw replacements.

    Full sequence matching ``state_events.lua:379-448``:

    1. Validate
    2. Sort discarded cards (left-to-right by index)
    3. Fire joker ``pre_discard`` context (Burnt Joker)
    4. Per-card: fire seal effects (Purple Seal → Tarot) + joker ``discard``
       context with ``other_card`` + ``full_hand``
    5. Process side-effects (dollars, card destruction, joker mutations)
    6. Discard cost (Golden Needle challenge)
    7. Decrement discards_left, increment discards_used
    8. Move surviving cards to discard pile
    9. Draw replacements from deck
    10. The Serpent: draw only 3 if not first action
    11. Re-debuff drawn cards for boss blind
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["discards_left"] <= 0:
        raise IllegalActionError("No discards remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot discard more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")

    # ------------------------------------------------------------------
    # 2. Extract discarded cards in sorted order
    # ------------------------------------------------------------------
    idx_set = set(indices)
    discarded = [hand[i] for i in sorted(indices)]
    gs["hand"] = [c for i, c in enumerate(hand) if i not in idx_set]

    # ------------------------------------------------------------------
    # 3. Fire joker pre_discard context (Burnt Joker: level up hand)
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    rng = gs.get("rng")
    game_snap = _build_discard_snapshot(gs, jokers)

    pre_discard_effects: list = []
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            pre_discard=True,
            full_hand=discarded,
            jokers=jokers,
            rng=rng,
            game=game_snap,
        )
        result = calculate_joker(joker, ctx)
        if result:
            pre_discard_effects.append(result)

    # Burnt Joker: level up the hand type of discarded cards
    for eff in pre_discard_effects:
        if eff.level_up:
            hand_levels = gs.get("hand_levels")
            if hand_levels is not None:
                from jackdaw.engine.hand_eval import evaluate_hand

                det = evaluate_hand(discarded)
                if det.detected_hand and det.detected_hand != "NULL":
                    hand_levels.level_up(det.detected_hand)

    # ------------------------------------------------------------------
    # 4. Per-card: seal effects + joker discard context
    # ------------------------------------------------------------------
    dollars_earned = 0
    destroyed: list = []
    jokers_to_remove: list = []

    for card in discarded:
        # Seal: Purple Seal → create random Tarot with append '8ba'
        # (card.lua:2254-2260; slot check gates the roll so no RNG is
        # consumed when consumable slots are full)
        if getattr(card, "seal", None) == "Purple":
            consumables: list = gs.setdefault("consumables", [])
            consumable_limit = gs.get("consumable_slots", 2)
            if len(consumables) < consumable_limit:
                _resolve_create_descriptors(gs, [{"type": "Tarot", "count": 1, "seed": "8ba"}])

        # Fire joker discard context per card
        card_destroyed = False
        for joker in jokers:
            if getattr(joker, "debuff", False):
                continue
            ctx = JokerContext(
                discard=True,
                other_card=card,
                full_hand=discarded,
                jokers=jokers,
                rng=rng,
                game=game_snap,
            )
            result = calculate_joker(joker, ctx)
            if result:
                dollars_earned += result.dollars
                if result.level_up:
                    # Burnt Joker: level up the discard hand type
                    hl = gs.get("hand_levels")
                    if hl is not None:
                        from jackdaw.engine.hand_eval import evaluate_hand as _eval

                        det = _eval(discarded)
                        if det.detected_hand and det.detected_hand != "NULL":
                            hl.level_up(det.detected_hand)
                if result.remove:
                    # Trading Card: destroy the discarded card
                    if result.extra and result.extra.get("destroy"):
                        card_destroyed = True
                    else:
                        # Ramen: self-destruct
                        if joker not in jokers_to_remove:
                            jokers_to_remove.append(joker)

        if card_destroyed:
            destroyed.append(card)

    # ------------------------------------------------------------------
    # 5. Process side-effects
    # ------------------------------------------------------------------
    if dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + dollars_earned

    for joker in jokers_to_remove:
        if joker in jokers:
            jokers.remove(joker)

    # ------------------------------------------------------------------
    # 6. Discard cost (Golden Needle challenge)
    # ------------------------------------------------------------------
    discard_cost = gs.get("modifiers", {}).get("discard_cost", 0)
    if discard_cost > 0:
        gs["dollars"] = gs.get("dollars", 0) - discard_cost

    # ------------------------------------------------------------------
    # 7. Decrement discards_left, increment discards_used
    # ------------------------------------------------------------------
    cr["discards_left"] -= 1
    cr["discards_used"] += 1

    # ------------------------------------------------------------------
    # 8. Move surviving cards to discard pile
    # ------------------------------------------------------------------
    surviving = [c for c in discarded if c not in destroyed]
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(surviving)

    # Track stat
    gs["round_scores"] = gs.get("round_scores", {})
    gs["round_scores"]["cards_discarded"] = gs["round_scores"].get("cards_discarded", 0) + len(
        discarded
    )

    # ------------------------------------------------------------------
    # 9-10. Draw replacements from deck
    # ------------------------------------------------------------------
    blind = gs.get("blind")
    serpent = (
        blind is not None
        and getattr(blind, "name", "") == "The Serpent"
        and not getattr(blind, "disabled", False)
        and (cr.get("hands_played", 0) > 0 or cr.get("discards_used", 0) > 0)
    )
    if serpent:
        # The Serpent: draw only 3 after first action
        # Lua's draw_card(G.deck, G.hand) pops LAST card from deck
        deck: list = gs.get("deck", [])
        hand_out: list = gs.get("hand", [])
        for _ in range(min(3, len(deck))):
            if deck:
                hand_out.append(deck.pop())
        _sort_hand_desc(hand_out)
    else:
        _draw_hand(gs)

    # ------------------------------------------------------------------
    # 11. Re-debuff drawn cards for boss blind
    # ------------------------------------------------------------------
    if blind and getattr(blind, "boss", False) and not getattr(blind, "disabled", False):
        pareidolia = any(
            getattr(j, "center_key", None) == "j_pareidolia" and not getattr(j, "debuff", False)
            for j in jokers
        )
        for card in gs.get("hand", []):
            blind.debuff_card(card, pareidolia=pareidolia)

        # Boss drawn_to_hand effects on discard redraw
        rng = gs.get("rng")
        dth = blind.drawn_to_hand(
            hand_cards=gs.get("hand", []),
            joker_cards=jokers,
            rng=rng,
        )
        if dth.get("forced_card_index") is not None:
            hand = gs.get("hand", [])
            idx = dth["forced_card_index"]
            if 0 <= idx < len(hand):
                hand[idx].ability["forced_selection"] = True

    return gs


def _build_discard_snapshot(gs: dict[str, Any], jokers: list) -> Any:
    """Build a GameSnapshot for discard context."""
    from jackdaw.engine.jokers import GameSnapshot

    cr = gs.get("current_round", {})
    return GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        discards_used=cr.get("discards_used", 0),
        mail_card_id=cr.get("mail_card", {}).get("id"),
        castle_card_suit=cr.get("castle_card", {}).get("suit"),
        skips=gs.get("skips", 0),
    )


def _handle_cash_out(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept round earnings and proceed to the shop.

    1. Shuffle deck (button_callbacks.lua:2918)
    2. Apply round earnings to dollars
    3. Track previous_round.dollars
    4. Populate shop (jokers, voucher, boosters)
    5. Phase → SHOP
    """
    _require_phase(gs, GamePhase.ROUND_EVAL)

    # End-of-round targeting-card re-roll (state_events.lua:273-276):
    # idol / mail / ancient / castle streams advance once per round END
    # (they are NOT re-rolled at round start; see start_round).
    rng = gs.get("rng")
    if rng:
        from jackdaw.engine.round_lifecycle import reset_round_targets

        ante = gs.get("round_resets", {}).get("ante", 1)
        reset_round_targets(rng, ante, gs)

    # Shuffle deck at cash-out (button_callbacks.lua:2918)
    # G.deck:shuffle('cashout'..G.GAME.round_resets.ante)
    if rng:
        deck: list = gs.get("deck", [])
        ante = gs.get("round_resets", {}).get("ante", 1)
        cashout_seed = rng.seed("cashout" + str(ante))
        rng.shuffle(deck, cashout_seed)

    earnings = gs.get("round_earnings")
    if earnings:
        gs["dollars"] = gs.get("dollars", 0) + earnings.total

    # eval-context tags (Investment: $25 once per tag, after a boss kill).
    if gs.get("last_blind_was_boss"):
        from jackdaw.engine.tags import Tag

        for entry in gs.get("awarded_tags", []):
            if entry.get("eval_fired"):
                continue
            result = Tag(entry.get("key", "")).apply(
                "eval", gs, rng=rng, last_blind_is_boss=True
            )
            if result is not None and result.dollars:
                gs["dollars"] = gs.get("dollars", 0) + result.dollars
                entry["eval_fired"] = True

    gs["previous_round"] = {"dollars": gs.get("dollars", 0)}

    # Populate shop
    _populate_shop(gs)

    gs["phase"] = GamePhase.SHOP
    return gs


def _handle_buy_card(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase a card from the shop.

    After buying:
    - Joker: add to jokers area, mark in used_jokers
    - Consumable: add to consumables area
    - Playing card: add to deck, fire ``playing_card_added`` joker context
    - Fire ``buying_card`` on all jokers
    """
    _require_phase(gs, GamePhase.SHOP)

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")

    card = shop_cards[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford card")

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)
    gs["current_round"]["jokers_purchased"] = (
        gs.get("current_round", {}).get("jokers_purchased", 0) + 1
    )

    # Place card in appropriate area
    card_set = _get_card_set(card)
    added_playing_card = False
    if card_set == "Joker":
        gs.setdefault("jokers", []).append(card)
        gs.setdefault("used_jokers", {})[card.center_key] = True
    elif card_set in ("Tarot", "Planet", "Spectral"):
        gs.setdefault("consumables", []).append(card)
    else:
        gs.setdefault("deck", []).append(card)
        added_playing_card = True

    # Fire buying_card joker context
    _fire_shop_joker_context(gs, buying_card=True)

    # Fire playing_card_added if a playing card was bought
    if added_playing_card:
        _fire_shop_joker_context(gs, playing_card_added=True, cards=[card])

    return gs


def _handle_sell_card(gs: dict[str, Any], area: str, idx: int) -> dict[str, Any]:
    """Sell a card for its sell value.

    After selling:
    - Fire ``selling_card`` on all jokers (Campfire +xMult)
    - If joker sold itself: fire ``selling_self``
    """
    _require_phase(gs, GamePhase.SHOP)

    cards: list = gs.get(area, [])
    if idx < 0 or idx >= len(cards):
        raise IllegalActionError(f"Invalid {area} index {idx}")

    card = cards[idx]
    if getattr(card, "eternal", False):
        raise IllegalActionError("Cannot sell eternal card")

    gs["dollars"] = gs.get("dollars", 0) + card.sell_cost
    cards.pop(idx)

    # Fire selling_card joker context (Campfire +xMult per card sold)
    _fire_shop_joker_context(gs, selling_card=True)

    return gs


def _handle_use_consumable(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Use a consumable from the player's consumable slots.

    Consumables can be used in BLIND_SELECT, SELECTING_HAND,
    ROUND_EVAL, and SHOP phases.  The phase does NOT change after use.

    Sequence:
    1. Validate phase and index
    2. Pop card from consumables
    3. Use via ``_use_consumable_card`` (builds ConsumableContext,
       calls handler, applies ConsumableResult mutations)
    4. Fire ``using_consumeable`` joker context if the result
       requests it (Constellation +xMult when Planet used)
    5. Track usage stats (last_tarot_planet)
    """
    _require_phase(
        gs, GamePhase.BLIND_SELECT, GamePhase.SELECTING_HAND, GamePhase.ROUND_EVAL, GamePhase.SHOP
    )

    consumables: list = gs.get("consumables", [])
    if idx < 0 or idx >= len(consumables):
        raise IllegalActionError(f"Invalid consumable index {idx}")

    card = consumables[idx]

    # Validate can_use before consuming (matches Balatro's can_use_consumeable)
    from jackdaw.engine.consumables import can_use_consumable

    hand: list = gs.get("hand", [])
    highlighted: list = []
    if targets:
        highlighted = [hand[i] for i in targets if i < len(hand)]

    if not can_use_consumable(
        card,
        highlighted=highlighted,
        hand_cards=hand,
        jokers=gs.get("jokers", []),
        consumables=consumables,
        joker_limit=gs.get("joker_slots", 5),
        consumable_limit=gs.get("consumable_slots", 2),
        game_state=gs,
    ):
        consumable_name = card.ability.get("name", card.center_key)
        raise IllegalActionError(f"Consumable {consumable_name!r} cannot be used at this time")

    consumables.pop(idx)
    _use_consumable_card(gs, card, targets)

    # Phase does NOT change — returns to whatever it was
    return gs


def _handle_redeem_voucher(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase and activate a voucher."""
    _require_phase(gs, GamePhase.SHOP)

    vouchers: list = gs.get("shop_vouchers", [])
    if idx < 0 or idx >= len(vouchers):
        raise IllegalActionError(f"Invalid voucher index {idx}")

    card = vouchers[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford voucher")

    gs["dollars"] -= card.cost
    vouchers.pop(idx)

    from jackdaw.engine.vouchers import apply_voucher

    gs["used_vouchers"][card.center_key] = True
    apply_voucher(card.center_key, gs)

    # Clear the voucher slot so the next shop doesn't re-offer it.
    # Matches card.lua:1850: G.GAME.current_round.voucher = nil
    gs.get("current_round", {})["voucher"] = None

    return gs


def _handle_open_booster(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Open a booster pack — generate cards and transition to PACK_OPENING.

    1. Deduct cost, remove pack from shop
    2. Generate pack cards via :func:`generate_pack_cards`
    3. Set ``pack_cards``, ``pack_choices_remaining``, ``pack_type``
    4. For Arcana/Spectral: deal hand from deck for targeting
    5. Fire ``open_booster`` joker context (Hallucination)
    6. Phase → PACK_OPENING
    """
    _require_phase(gs, GamePhase.SHOP)

    boosters: list = gs.get("shop_boosters", [])
    if idx < 0 or idx >= len(boosters):
        raise IllegalActionError(f"Invalid booster index {idx}")

    pack = boosters[idx]
    if pack.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford booster")

    gs["dollars"] -= pack.cost
    boosters.pop(idx)

    # Generate pack cards
    from jackdaw.engine.data.prototypes import BOOSTERS
    from jackdaw.engine.packs import generate_pack_cards

    pack_key = pack.center_key
    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)

    if rng and pack_key in BOOSTERS:
        cards, choose = generate_pack_cards(pack_key, rng, ante, gs)
        gs["pack_cards"] = cards
        gs["pack_choices_remaining"] = choose
        gs["pack_type"] = BOOSTERS[pack_key].kind
    else:
        gs["pack_cards"] = []
        gs["pack_choices_remaining"] = 1
        gs["pack_type"] = "Unknown"

    gs["shop_return_phase"] = GamePhase.SHOP

    # For Arcana/Spectral packs: deal hand from deck for targeting
    # Cards are drawn from the END of deck (top of visual stack),
    # matching Lua's draw_card(G.deck, G.hand) which pops last card.
    pack_kind = gs.get("pack_type", "")
    if pack_kind in ("Arcana", "Spectral"):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        hand_size = gs.get("hand_size", 8)
        to_deal = min(len(deck), hand_size - len(hand))
        pack_hand: list = []
        for _ in range(to_deal):
            if deck:
                card = deck.pop()
                pack_hand.append(card)
        gs["pack_hand"] = pack_hand
        # These cards serve as targets for Tarot/Spectral use
        combined_hand = hand + pack_hand
        _sort_hand_desc(combined_hand)
        gs["hand"] = combined_hand

    # Fire open_booster joker context (Hallucination creates Tarot)
    _fire_shop_joker_context(gs, open_booster=True)

    gs["phase"] = GamePhase.PACK_OPENING
    return gs


def _handle_pick_pack_card(
    gs: dict[str, Any],
    idx: int,
    targets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Pick a card from an opened booster pack.

    Matching ``button_callbacks.lua:2155-2247`` use_card:

    - **Consumable** (Arcana/Spectral/Celestial): use immediately via
      ``use_consumeable``.  For Arcana/Spectral, ``targets`` specifies
      which dealt hand cards the consumable should target.  Planets
      are used without targets (level up hand type).
    - **Playing card** (Standard pack): added to deck.  Fires
      ``playing_card_added`` joker context (Hologram).
    - **Joker** (Buffoon pack): added to joker slots.  Marks in
      ``used_jokers``.

    When ``pack_choices_remaining`` hits 0 or pack is empty, the pack
    closes: remaining cards are removed, dealt hand cards (if any)
    return to deck, and phase restores to SHOP.
    """
    _require_phase(gs, GamePhase.PACK_OPENING)

    pack_cards: list = gs.get("pack_cards", [])
    remaining = gs.get("pack_choices_remaining", 0)
    if remaining <= 0:
        raise IllegalActionError("No pack choices remaining")
    if idx < 0 or idx >= len(pack_cards):
        raise IllegalActionError(f"Invalid pack card index {idx}")

    card = pack_cards.pop(idx)
    gs["pack_choices_remaining"] = remaining - 1

    # Determine card type and handle accordingly
    card_set = _get_card_set(card)

    if card_set in ("Tarot", "Planet", "Spectral"):
        # Consumable: use immediately (Arcana/Spectral/Celestial pack)
        _use_consumable_card(gs, card, targets)

        # Fire using_consumeable joker context
        _fire_shop_joker_context(gs, using_consumeable=True)

    elif card_set == "Joker":
        # Buffoon pack: add to joker slots
        gs.setdefault("jokers", []).append(card)
        gs.setdefault("used_jokers", {})[card.center_key] = True

    else:
        # Standard pack: playing card → add to deck
        gs.setdefault("deck", []).append(card)
        # Fire playing_card_added joker context (Hologram)
        _fire_shop_joker_context(gs, playing_card_added=True, cards=[card])

    # Check if pack should close
    if gs["pack_choices_remaining"] <= 0 or not pack_cards:
        _close_pack(gs)

    return gs


def _handle_skip_pack(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip remaining pack cards.

    Fires ``skipping_booster`` on all jokers (Red Card +mult per skip),
    then closes the pack.
    """
    _require_phase(gs, GamePhase.PACK_OPENING)

    # Fire skipping_booster joker context (Red Card +mult)
    _fire_shop_joker_context(gs, skipping_booster=True)

    _close_pack(gs)
    return gs


def _handle_reroll(gs: dict[str, Any]) -> dict[str, Any]:
    """Reroll the shop.

    After rerolling:
    - Fire ``reroll_shop`` on all jokers (Flash Card +mult)
    - Track times_rerolled stat
    """
    _require_phase(gs, GamePhase.SHOP)

    cr = gs.get("current_round", {})
    free = cr.get("free_rerolls", 0)
    cost = cr.get("reroll_cost", 5)

    if free > 0:
        cr["free_rerolls"] = free - 1
    elif gs.get("dollars", 0) >= cost:
        gs["dollars"] -= cost
    else:
        raise IllegalActionError("Cannot afford reroll")

    # Increment reroll cost for next reroll
    cr["reroll_cost_increase"] = cr.get("reroll_cost_increase", 0) + 1
    cr["reroll_cost"] = gs.get("base_reroll_cost", 5) + cr["reroll_cost_increase"]

    # Track stat
    gs.setdefault("round_scores", {})
    gs["round_scores"]["times_rerolled"] = gs["round_scores"].get("times_rerolled", 0) + 1

    # Regenerate shop joker cards
    _reroll_shop_cards(gs)

    # Fire reroll_shop joker context (Flash Card +mult)
    _fire_shop_joker_context(gs, reroll_shop=True)

    return gs


def _handle_next_round(gs: dict[str, Any]) -> dict[str, Any]:
    """Leave the shop and proceed to the next blind.

    Before leaving:
    - Fire ``ending_shop`` on all jokers (Perkeo copies consumable)
    - Process Perkeo side-effects
    """
    _require_phase(gs, GamePhase.SHOP)

    # Fire ending_shop joker context (Perkeo)
    mutations = _fire_shop_joker_context(gs, ending_shop=True)
    _apply_shop_mutations(gs, mutations)

    # Clear shop areas
    gs["shop_cards"] = []
    gs["shop_vouchers"] = []
    gs["shop_boosters"] = []

    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")

    # Set next blind to Select
    if blind_on_deck == "Small":
        rr["blind_states"]["Small"] = "Select"
    elif blind_on_deck == "Big":
        rr["blind_states"]["Big"] = "Select"
    else:
        rr["blind_states"]["Boss"] = "Select"

    gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_sort_hand(gs: dict[str, Any], mode: str) -> dict[str, Any]:
    """Sort the hand by rank or suit."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    if mode == "rank":
        hand.sort(
            key=lambda c: (
                getattr(c.base, "id", 0) if c.base else 0,
                getattr(c.base, "suit_nominal", 0) if c.base else 0,
            )
        )
    elif mode == "suit":
        hand.sort(
            key=lambda c: (
                getattr(c.base, "suit_nominal", 0) if c.base else 0,
                getattr(c.base, "id", 0) if c.base else 0,
            )
        )
    return gs


def _handle_swap_hand(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a hand card with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    Free action — no cost, doesn't consume hands or discards.
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    other = idx + direction
    if not (0 <= idx < len(hand) and 0 <= other < len(hand)):
        raise IllegalActionError("Swap index out of range")

    hand[idx], hand[other] = hand[other], hand[idx]
    return gs


def _handle_swap_jokers(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a joker with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    """
    _require_phase(gs, GamePhase.SELECTING_HAND, GamePhase.SHOP)

    jokers: list = gs.get("jokers", [])
    other = idx + direction
    if not (0 <= idx < len(jokers) and 0 <= other < len(jokers)):
        raise IllegalActionError("Swap index out of range")

    jokers[idx], jokers[other] = jokers[other], jokers[idx]
    return gs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_hand_desc(hand: list) -> None:
    """Sort hand in place, descending by nominal value.

    Matches Lua ``CardArea:sort()`` with default config ``sort='desc'``
    (cardarea.lua:577-580).  Uses ``Card.get_nominal()`` as the sort key,
    which combines rank, suit tiebreaker, face nominal, and a unique
    micro-value so every card gets a distinct position.

    Only sorts cards that have a ``get_nominal`` method (playing cards);
    non-playing-card entries are left at the end.
    """
    hand.sort(key=lambda c: c.get_nominal() if hasattr(c, "get_nominal") else -1e9, reverse=True)


def _draw_hand(gs: dict[str, Any]) -> None:
    """Draw cards from deck to fill the hand up to hand_size.

    Cards are drawn from the END of the deck list (top of the visual
    stack), matching Lua's ``draw_card(G.deck, G.hand, ...)`` which
    pops from the last position.

    After drawing, the hand is sorted descending by nominal value
    (matching Lua's ``draw_from_deck_to_hand`` which passes ``sort=true``
    to ``draw_card``, triggering ``CardArea:sort()`` with default 'desc').
    """
    deck: list = gs.get("deck", [])
    hand: list = gs.setdefault("hand", [])
    hand_size: int = gs.get("hand_size", 8)
    to_draw = min(len(deck), hand_size - len(hand))
    for _ in range(to_draw):
        if deck:
            hand.append(deck.pop())
    # Sort hand descending by nominal (matches Lua CardArea:sort 'desc')
    _sort_hand_desc(hand)


def _round_won(gs: dict[str, Any]) -> None:
    """Handle winning a round — transition to ROUND_EVAL.

    Full sequence matching ``state_events.lua:87-120``:

    1. Fire joker ``end_of_round`` context (economy + scaling)
    2. Process perishable/rental (round_lifecycle)
    3. Gold Seal: +$3 per held card with Gold Seal
    4. Return all cards to deck (hand + played + discard)
    5. Un-debuff all playing cards (blind debuffs don't persist)
    6. Track unused discards (for Garbage Tag)
    7. Mark blind as Defeated
    8. Advance blind progression (Small→Big, Big→Boss)
    9. Boss beaten: check win condition, advance ante
    10. Calculate round earnings
    11. Phase → ROUND_EVAL
    """
    from jackdaw.engine.economy import calculate_round_earnings
    from jackdaw.engine.round_lifecycle import process_round_end_cards

    cr = gs["current_round"]
    blind = gs["blind"]
    jokers = gs.get("jokers", [])
    rng = gs.get("rng")

    # ------------------------------------------------------------------
    # 1. Fire joker end_of_round context
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import GameSnapshot, on_end_of_round

    game_snap = GameSnapshot(
        money=gs.get("dollars", 0),
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        joker_count=len(jokers),
    )
    eor = on_end_of_round(jokers, game_snap, rng)
    # Apply joker end-of-round dollars
    gs["dollars"] = gs.get("dollars", 0) + eor.get("dollars_earned", 0)
    # Remove self-destructed jokers (Popcorn, Turtle Bean, etc.)
    for removed_joker in eor.get("jokers_removed", []):
        if removed_joker in jokers:
            jokers.remove(removed_joker)

    # ------------------------------------------------------------------
    # 2. Process perishable/rental
    # ------------------------------------------------------------------
    process_round_end_cards(jokers, gs)

    # ------------------------------------------------------------------
    # 3. Gold Seal: +$3 per held card with Gold Seal in hand
    # ------------------------------------------------------------------
    hand: list = gs.get("hand", [])
    gold_seal_dollars = sum(
        3 for c in hand if getattr(c, "seal", None) == "Gold" and not getattr(c, "debuff", False)
    )
    if gold_seal_dollars:
        gs["dollars"] = gs.get("dollars", 0) + gold_seal_dollars

    # ------------------------------------------------------------------
    # 3b. Blue Seal: create Planet for most-played hand type
    # ------------------------------------------------------------------
    hand_levels = gs.get("hand_levels")
    consumables: list = gs.get("consumables", [])
    consumable_limit = gs.get("consumable_slots", 2)
    for c in hand:
        if getattr(c, "seal", None) == "Blue" and not getattr(c, "debuff", False):
            if len(consumables) < consumable_limit and hand_levels is not None:
                most_played = hand_levels.most_played()
                # Find the planet key for this hand type
                from jackdaw.engine.consumables import _PLANET_HAND

                planet_key = None
                for pk, ht in _PLANET_HAND.items():
                    if ht == most_played.value:
                        planet_key = pk
                        break
                if planet_key:
                    from jackdaw.engine.card import Card as _BSCard

                    planet = _BSCard(center_key=planet_key)
                    planet.ability = {"set": "Planet", "effect": ""}
                    consumables.append(planet)

    # ------------------------------------------------------------------
    # 4. Return all cards to deck
    #
    # Lua sequence (state_events.lua:237-250):
    #   a) draw_from_hand_to_discard — hand cards removed first-first,
    #      appended at end of discard
    #   b) draw_from_discard_to_deck — discard cards popped LAST-first
    #      (remove_card on discard type takes #cards), then INSERTED AT
    #      FRONT of deck (emplace on deck type does table.insert(1))
    #
    # Net effect: [old_discard, hand] is prepended to deck front in
    # original order (pop-last + insert-at-front cancel out).
    # ------------------------------------------------------------------
    deck: list = gs.setdefault("deck", [])
    played: list = gs.get("played_cards_area", [])
    discarded: list = gs.get("discard_pile", [])

    # Step a: hand → discard end (forward order)
    discarded.extend(hand)
    # Any leftover played cards (shouldn't normally exist post-scoring)
    discarded.extend(played)
    # Step b: discard → deck FRONT (pop-last + insert-at-front = original order at front)
    deck[:0] = discarded

    gs["hand"] = []
    gs["played_cards_area"] = []
    gs["discard_pile"] = []

    # ------------------------------------------------------------------
    # 5. Un-debuff all playing cards (blind debuffs don't persist)
    # ------------------------------------------------------------------
    for card in deck:
        # Only clear blind-applied debuffs; perishable debuffs are permanent
        if getattr(card, "debuff", False):
            if not (getattr(card, "perishable", False) and getattr(card, "perish_tally", 1) <= 0):
                card.debuff = False

    # ------------------------------------------------------------------
    # 6. Track unused discards / hands played (for Garbage/Handy Tags)
    #    These are run-level cumulative totals matching Lua:
    #    - G.GAME.unused_discards += current_round.discards_left  (state_events.lua:124)
    #    - G.GAME.hands_played += 1 per hand  (state_events.lua:523, tracked at line 522)
    # ------------------------------------------------------------------
    gs["unused_discards"] = gs.get("unused_discards", 0) + cr.get("discards_left", 0)

    # ------------------------------------------------------------------
    # 7. Mark blind as Defeated
    # ------------------------------------------------------------------
    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr["blind_states"][blind_on_deck] = "Defeated"
    gs["round"] = gs.get("round", 0) + 1
    # Remembered for cash-out tag hooks (Investment Tag pays after a boss).
    gs["last_blind_was_boss"] = blind_on_deck == "Boss"

    # ------------------------------------------------------------------
    # 8-9. Advance blind progression
    # ------------------------------------------------------------------
    if blind_on_deck == "Small":
        gs["blind_on_deck"] = "Big"
    elif blind_on_deck == "Big":
        gs["blind_on_deck"] = "Boss"
    elif blind_on_deck == "Boss":
        # Boss beaten — check win, advance ante
        if rr["ante"] >= gs.get("win_ante", 8):
            gs["won"] = True
        _advance_ante(gs)
        gs["blind_on_deck"] = "Small"

    # The Manacle: restore hand size after boss defeat
    if blind_on_deck == "Boss" and getattr(blind, "name", "") == "The Manacle":
        if not getattr(blind, "disabled", False):
            gs["hand_size"] = gs.get("hand_size", 7) + 1

    # ------------------------------------------------------------------
    # 10. Calculate round earnings (for cash-out screen)
    # ------------------------------------------------------------------
    earnings = calculate_round_earnings(
        blind=blind,
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        money=gs.get("dollars", 0),
        jokers=jokers,
        game_state=gs,
        rng=rng,
        joker_dollars=eor.get("dollars_earned", 0),
    )
    gs["round_earnings"] = earnings

    # ------------------------------------------------------------------
    # 11. Phase → ROUND_EVAL
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.ROUND_EVAL


def _advance_ante(gs: dict[str, Any]) -> None:
    """Advance to the next ante after boss is defeated."""
    rr = gs["round_resets"]
    rr["ante"] += 1
    rr["blind_ante"] = rr["ante"]
    rr["blind_states"] = {"Small": "Select", "Big": "Upcoming", "Boss": "Upcoming"}
    rr["boss_rerolled"] = False

    # Generate new boss, tags, voucher for next ante
    from jackdaw.engine.tags import assign_ante_blinds

    rng = gs.get("rng")
    if rng:
        ante_result = assign_ante_blinds(rr["ante"], rng, gs)
        rr["blind_choices"]["Boss"] = ante_result["blind_choices"]["Boss"]
        gs["current_round"]["voucher"] = ante_result["voucher"]


# ---------------------------------------------------------------------------
# setting_blind joker context
# ---------------------------------------------------------------------------


def _fire_setting_blind(
    gs: dict[str, Any],
    jokers: list,
    blind: Any,
) -> list[dict[str, Any]]:
    """Fire ``setting_blind`` on all jokers and collect mutations.

    Returns a list of side-effect dicts from JokerResult.extra.
    """
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    game_snap = GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
    )

    mutations: list[dict[str, Any]] = []
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            setting_blind=True,
            blind=blind,
            jokers=jokers,
            game=game_snap,
        )
        result = calculate_joker(joker, ctx)
        if result and result.extra:
            mutations.append(result.extra)

    return mutations


def _apply_setting_blind_mutations(
    gs: dict[str, Any],
    mutations: list[dict[str, Any]],
    jokers: list,
) -> None:
    """Process side-effects from setting_blind jokers."""
    rng = gs.get("rng")

    for mut in mutations:
        # Chicot / Luchador: disable blind
        if mut.get("disable_blind"):
            blind = gs.get("blind")
            if blind:
                blind.disabled = True
                # Un-debuff all playing cards
                for card in gs.get("deck", []):
                    card.debuff = False

        # Madness: destroy random joker (not self)
        if mut.get("destroy_random_joker") and len(jokers) > 1:
            if rng:
                # Pick a random non-self joker to destroy

                candidates = [j for j in jokers if j is not jokers[0]]
                if candidates:
                    seed_val = rng.seed("madness")
                    target, _ = rng.element(candidates, seed_val)
                    jokers.remove(target)

        # Burglar: set hands / remove discards
        if "set_hands" in mut:
            cr = gs.get("current_round", {})
            cr["hands_left"] = cr.get("hands_left", 0) + mut["set_hands"]
        if "set_discards" in mut:
            cr = gs.get("current_round", {})
            cr["discards_left"] = mut["set_discards"]

        # Marble Joker / Certificate / Riff-raff / Cartomancer: create cards
        if "create" in mut:
            create = mut["create"]
            ctype = create.get("type", "")
            if ctype == "playing_card":
                # Marble Joker (card.lua:2580): front via 'marb_fr', Stone
                # center, joins the playing cards. Certificate (card.lua:2462):
                # front via 'cert_fr' emplaced into the HAND, seal via
                # 'certsl' (>0.75 Red, >0.5 Blue, >0.25 Gold, else Purple).
                # The old code appended a Card with NO front (base=None),
                # which corrupted the playing-card pool and crashed the
                # round-end targeting rolls (mail.base.id).
                from jackdaw.engine.card_factory import (
                    RANK_LETTER,
                    SUIT_LETTER,
                    create_playing_card,
                )

                ckey = create.get("key", "")
                stream = {"cert": "cert_fr", "marble": "marb_fr"}.get(ckey)
                if rng is not None and stream is not None:
                    p_cards = {
                        f"{sl}_{rl}": (suit, rank)
                        for sl, suit in SUIT_LETTER.items()
                        for rl, rank in RANK_LETTER.items()
                    }
                    (c_suit, c_rank), _ = rng.element(p_cards, rng.seed(stream))
                    seal = None
                    if create.get("seal"):
                        roll = rng.random(rng.seed("certsl"))
                        if roll > 0.75:
                            seal = "Red"
                        elif roll > 0.5:
                            seal = "Blue"
                        elif roll > 0.25:
                            seal = "Gold"
                        else:
                            seal = "Purple"
                    c = create_playing_card(
                        c_suit,
                        c_rank,
                        create.get("enhancement", "c_base"),
                        seal=seal,
                        hands_played=gs.get("hands_played", 0),
                    )
                    if ckey == "cert" and gs.get("phase") == GamePhase.SELECTING_HAND:
                        gs.setdefault("hand", []).append(c)
                        _sort_hand_desc(gs.get("hand", []))
                    else:
                        gs.setdefault("deck", []).append(c)
            elif ctype in ("Joker", "Tarot", "Planet", "Spectral"):
                # Riff-raff ('rif', Common), Cartomancer ('car'), 8 Ball
                # ('8ba'), etc. — roll the real pool with the descriptor's
                # append key instead of hardcoding a center.
                _resolve_create_descriptors(gs, [create])


# ---------------------------------------------------------------------------
# Boss blind set-time effects
# ---------------------------------------------------------------------------


def _apply_boss_blind_effects(gs: dict[str, Any], blind: Any) -> None:
    """Apply boss blind effects at set-time (blind.lua:157-209).

    These are one-time mutations that happen when the blind is set,
    before the round starts.
    """
    cr = gs.get("current_round", {})
    name = getattr(blind, "name", "")

    # The Water: remove all discards
    if name == "The Water":
        current_discards = cr.get("discards_left", 0)
        blind.discards_sub = current_discards
        cr["discards_left"] = 0

    # The Needle: reduce to 1 hand
    elif name == "The Needle":
        rr = gs.get("round_resets", {})
        current_hands = rr.get("hands", 4)
        blind.hands_sub = current_hands - 1
        cr["hands_left"] = max(1, cr.get("hands_left", current_hands) - blind.hands_sub)

    # The Manacle: -1 hand size
    elif name == "The Manacle":
        gs["hand_size"] = gs.get("hand_size", 8) - 1

    # Amber Acorn: shuffle jokers (flip + randomize order)
    elif name == "Amber Acorn":
        jokers: list = gs.get("jokers", [])
        if jokers:
            for j in jokers:
                j.facing = "back"
            rng = gs.get("rng")
            if rng and len(jokers) > 1:
                seed_val = rng.seed("aajk")
                rng.shuffle(jokers, seed_val)

    # The Eye: reset hand tracking
    elif name == "The Eye":
        blind.hands = {
            ht: False
            for ht in [
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
        }

    # The Mouth: reset only_hand
    elif name == "The Mouth":
        blind.only_hand = None

    # The House / The Mark: flip cards face-down (blind.lua:200-203)
    # Cards are flipped at draw_to_hand time, not set_blind time.
    # We handle this in _draw_hand context by checking blind.name.


# ---------------------------------------------------------------------------
# Shop joker context helpers
# ---------------------------------------------------------------------------


def _use_consumable_card(
    gs: dict[str, Any],
    card: Any,
    targets: tuple[int, ...] | None = None,
) -> None:
    """Build a ConsumableContext and use a consumable card.

    Bridges the game_state dict with the ``use_consumable(card, ctx)`` API.

    1. Build ConsumableContext from game_state + target_indices
    2. Call handler → ConsumableResult
    3. Apply all result mutations
    4. Fire ``using_consumeable`` joker context if result requests it
    5. Track usage (last_tarot_planet)
    """
    from jackdaw.engine.consumables import ConsumableContext, use_consumable

    hand: list = gs.get("hand", [])
    highlighted: list = []
    if targets:
        highlighted = [hand[i] for i in targets if i < len(hand)]

    ctx = ConsumableContext(
        card=card,
        highlighted=highlighted or None,
        hand_cards=hand or None,
        jokers=gs.get("jokers") or None,
        consumables=gs.get("consumables") or None,
        playing_cards=gs.get("deck") or None,
        rng=gs.get("rng"),
        game_state=gs,
    )
    result = use_consumable(card, ctx)

    if result is None:
        return

    # Apply all mutations
    _apply_consumable_result(gs, result, card)

    # Fire using_consumeable joker context (Constellation, etc.)
    if getattr(result, "notify_jokers_consumeable", False):
        from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

        jokers: list = gs.get("jokers", [])
        game_snap = GameSnapshot(
            joker_count=len(jokers),
            money=gs.get("dollars", 0),
        )
        for joker in list(jokers):
            if getattr(joker, "debuff", False):
                continue
            jctx = JokerContext(
                using_consumeable=True,
                consumeable=card,
                jokers=jokers,
                game=game_snap,
            )
            calculate_joker(joker, jctx)


def _apply_consumable_result(
    gs: dict[str, Any],
    result: Any,
    card: Any = None,
) -> None:
    """Apply a ConsumableResult's mutations to game_state.

    Handles all 14+ mutation types from ConsumableResult.
    """
    # Track last_tarot_planet for The Fool
    card_key = getattr(card, "center_key", None)
    if card_key:
        card_set = _get_card_set(card) if card else ""
        if card_set in ("Tarot", "Planet"):
            gs["last_tarot_planet"] = card_key

    # ---- Card modifications ----

    # a. Enhancement
    if getattr(result, "enhance", None):
        for target, enh_key in result.enhance:
            if hasattr(target, "set_ability"):
                target.set_ability(enh_key)

    # b. Suit changes
    if getattr(result, "change_suit", None):
        for target, suit in result.change_suit:
            if hasattr(target, "change_suit"):
                target.change_suit(suit)

    # c. Rank changes
    if getattr(result, "change_rank", None):
        for target, delta in result.change_rank:
            if hasattr(target, "change_rank"):
                target.change_rank(delta)

    # d. Copy card (Death)
    if getattr(result, "copy_card", None):
        source, target = result.copy_card
        if hasattr(target, "copy_from"):
            target.copy_from(source)
        else:
            # Manual copy: base, enhancement, edition, seal
            if source.base and target.base:
                target.set_base(
                    source.card_key or "",
                    source.base.suit.value,
                    source.base.rank.value,
                )
            target.edition = source.edition
            target.seal = source.seal
            if hasattr(source, "center_key") and hasattr(target, "set_ability"):
                target.set_ability(source.center_key)

    # e. Destroy playing cards
    if getattr(result, "destroy", None):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        for destroyed in result.destroy:
            if destroyed in hand:
                hand.remove(destroyed)
            if destroyed in deck:
                deck.remove(destroyed)

    # f. Add seal
    if getattr(result, "add_seal", None):
        for target, seal_type in result.add_seal:
            target.seal = seal_type

    # g. Create cards (High Priestess, Emperor, Judgement, etc.)
    if getattr(result, "create", None):
        _resolve_create_descriptors(gs, result.create)

    # ---- Economy ----

    # h. Dollars
    if getattr(result, "dollars", 0):
        gs["dollars"] = gs.get("dollars", 0) + result.dollars

    # i. Money set (Wraith → set to 0)
    if getattr(result, "money_set", None) is not None:
        gs["dollars"] = result.money_set

    # ---- Hand levels ----

    # j. Level up (Planet cards)
    if getattr(result, "level_up", None):
        hand_levels = gs.get("hand_levels")
        if hand_levels:
            for ht, amount in result.level_up:
                hand_levels.level_up(ht, amount)

    # ---- Deck mutation ----

    # k. Add playing cards to deck
    if getattr(result, "add_to_deck", None):
        import copy as _copy

        from jackdaw.engine.card import Card as _Card
        from jackdaw.engine.card import _next_sort_id

        deck_list: list = gs.setdefault("deck", [])
        for card_spec in result.add_to_deck:
            # Cryptid: copy an existing card — copies are emplaced into the
            # HAND, not the draw pile (card.lua:1206-1213: copy_card +
            # G.hand:emplace), with a fresh sort_id like any new Card.
            copy_source = card_spec.get("copy_of")
            if copy_source is not None:
                new_card = _copy.deepcopy(copy_source)
                new_card.sort_id = _next_sort_id()
                if gs.get("phase") == GamePhase.SELECTING_HAND:
                    gs.setdefault("hand", []).append(new_card)
                    _sort_hand_desc(gs.get("hand", []))
                else:
                    deck_list.append(new_card)
                continue
            new_card = _Card()
            if "suit" in card_spec and "rank" in card_spec:
                new_card.set_base(
                    card_spec.get("key", ""),
                    card_spec["suit"],
                    card_spec["rank"],
                )
            if "enhancement" in card_spec:
                new_card.set_ability(card_spec["enhancement"])
            deck_list.append(new_card)

    # ---- Joker effects ----

    # l. Add edition (Wheel of Fortune, Aura)
    if getattr(result, "add_edition", None):
        ae = result.add_edition
        target = ae.get("target")
        edition = ae.get("edition")
        if target and edition:
            target.edition = edition

    # m. Destroy jokers (Ankh: destroy all except one)
    if getattr(result, "destroy_jokers", None):
        jokers: list = gs.get("jokers", [])
        for j in result.destroy_jokers:
            if j in jokers:
                jokers.remove(j)

    # ---- Game state ----

    # n. Hand size modification (Ectoplasm -1, Ouija -1)
    if getattr(result, "hand_size_mod", 0):
        gs["hand_size"] = gs.get("hand_size", 8) + result.hand_size_mod


def _resolve_create_descriptors(gs: dict[str, Any], descriptors: list[dict[str, Any]]) -> None:
    """Resolve card creation descriptors from ConsumableResult.create.

    Each descriptor is ``{'type': ..., 'count': ..., 'seed': ...,
    'forced_key': ...}``.  Creates the actual Card objects and adds
    them to the appropriate area.

    Delegates to :func:`~jackdaw.engine.card_factory.resolve_create_descriptor`
    for actual card creation.
    """
    from jackdaw.engine.card_factory import resolve_create_descriptor

    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)
    # Planet pool softlock filtering needs current played-hand counts
    # (High Priestess / Blue Seal create Planets mid-round).
    _sync_played_hand_types(gs)
    consumables: list = gs.setdefault("consumables", [])
    consumable_limit = gs.get("consumable_slots", 2)
    jokers: list = gs.setdefault("jokers", [])
    joker_slots = gs.get("joker_slots", 5)
    deck: list = gs.setdefault("deck", [])

    for desc in descriptors:
        count = desc.get("count", 1)

        for _ in range(count):
            card = resolve_create_descriptor(desc, rng, ante, gs)
            if card is None:
                continue

            card_set = card.ability.get("set", "")
            if card_set == "Joker":
                negative = card.edition and card.edition.get("negative")
                if len(jokers) < joker_slots + (1 if negative else 0):
                    jokers.append(card)
            elif card_set in ("Tarot", "Planet", "Spectral"):
                if len(consumables) < consumable_limit:
                    consumables.append(card)
            elif card_set in ("Default", "Enhanced", ""):
                # Playing card — add to hand if mid-round, otherwise deck.
                # Matches Balatro which routes spectral-created cards to
                # the hand area during SELECTING_HAND.
                if gs.get("phase") == GamePhase.SELECTING_HAND:
                    gs.setdefault("hand", []).append(card)
                else:
                    deck.append(card)

    # Re-sort hand if any cards were added during SELECTING_HAND
    if gs.get("phase") == GamePhase.SELECTING_HAND:
        _sort_hand_desc(gs.get("hand", []))


# ---------------------------------------------------------------------------
# Shop population helpers
# ---------------------------------------------------------------------------


def _sync_played_hand_types(gs: dict[str, Any]) -> None:
    """Populate ``gs["played_hand_types"]`` from per-run play counts.

    The Planet pool softlock gate is ``G.GAME.hands[hand_type].played > 0``
    (common_events.lua:2009) — a hand type counts only once it has been
    PLAYED this run.  Leveling a secret hand (e.g. via a Ceres from a pack)
    makes it *visible* but does not unlock its planet in pools.
    """
    from jackdaw.engine.hand_levels import HandLevels

    hand_levels: HandLevels | None = gs.get("hand_levels")
    if hand_levels is None:
        return
    played: set[str] = set()
    for ht, state in hand_levels._hands.items():
        if state.played > 0:
            played.add(ht.value)
    gs["played_hand_types"] = played


def _populate_shop(gs: dict[str, Any]) -> None:
    """Generate shop cards using populate_shop and store in game_state.

    Places results in ``gs["shop_cards"]``, ``gs["shop_vouchers"]``,
    ``gs["shop_boosters"]``.
    """
    from jackdaw.engine.shop import populate_shop

    rng = gs.get("rng")
    if rng is None:
        return

    # Sync played hand types for Planet pool softlock filtering
    # (G.GAME.hands[ht].played > 0, common_events.lua:2009).
    _sync_played_hand_types(gs)

    ante = gs.get("round_resets", {}).get("ante", 1)
    result = populate_shop(rng, ante, gs)

    gs["shop_cards"] = result.get("jokers", [])
    voucher = result.get("voucher")
    gs["shop_vouchers"] = [voucher] if voucher else []
    gs["shop_boosters"] = result.get("boosters", [])

    _fire_shop_tags(gs, rerolled=False)


def _fire_shop_tags(gs: dict[str, Any], rerolled: bool) -> None:
    """Fire pending shop-context tag hooks on freshly stocked shop cards.

    Wires the hooks :func:`~jackdaw.engine.shop.populate_shop` defers (its
    "M11 tag system" note), consuming awarded tags FIFO:

    * ``store_joker_modify`` — Foil/Holo/Polychrome/Negative Tag: the next
      base-edition shop Joker gains the edition and becomes free
      (tags.lua ``Tag:apply_to_run``). Applies to rerolled jokers too.
    * ``shop_final_pass`` — Coupon Tag: initial shop cards and boosters
      become free (shop entry only).
    * ``shop_start`` — D6 Tag: rerolls start free (shop entry only).

    ``store_joker_create`` (Rare/Uncommon Tag) and ``voucher_add``
    (Voucher Tag) remain unwired: both replace RNG-consuming rolls at
    creation time, so a faithful implementation needs live validation to
    pin stream behavior first. Those tags stay pending and inert rather
    than risking a new stream divergence.
    """
    from jackdaw.engine.tags import Tag

    awarded: list[dict[str, Any]] = gs.get("awarded_tags", [])
    rng = gs.get("rng")

    for entry in awarded:
        if entry.get("shop_fired"):
            continue
        tag = Tag(entry.get("key", ""))

        result = tag.apply("store_joker_modify", gs, rng=rng)
        if result is not None and result.force_edition:
            target = None
            for c in gs.get("shop_cards", []):
                ability = getattr(c, "ability", None) or {}
                if ability.get("set") == "Joker" and not getattr(c, "edition", None):
                    target = c
                    break
            if target is None:
                continue  # no eligible joker; tag stays pending for later
            edition = {str(result.force_edition): True}
            if hasattr(target, "set_edition"):
                target.set_edition(edition)
            else:
                target.edition = edition
            target.cost = 0  # vanilla: tag-edition shop jokers are free
            entry["shop_fired"] = True
            continue

        if rerolled:
            continue  # remaining hooks fire on shop entry only

        result = tag.apply("shop_final_pass", gs, rng=rng)
        if result is not None and result.coupon:
            for c in gs.get("shop_cards", []) + gs.get("shop_boosters", []):
                c.cost = 0
            entry["shop_fired"] = True
            continue

        result = tag.apply("shop_start", gs, rng=rng)
        if result is not None and result.free_rerolls:
            cr = gs.setdefault("current_round", {})
            cr["free_rerolls"] = cr.get("free_rerolls", 0) + result.free_rerolls
            entry["shop_fired"] = True


def _reroll_shop_cards(gs: dict[str, Any]) -> None:
    """Regenerate the shop joker/consumable cards (not voucher or boosters).

    Matches the repopulate step of ``reroll_shop``
    (``button_callbacks.lua:2855``).
    """
    from jackdaw.engine.card_factory import create_card
    from jackdaw.engine.shop import select_shop_card_type

    rng = gs.get("rng")
    if rng is None:
        return

    _sync_played_hand_types(gs)

    ante = gs.get("round_resets", {}).get("ante", 1)
    shop_joker_max: int = gs.get("shop", {}).get("joker_max", 2)

    new_cards = []
    for _ in range(shop_joker_max):
        card_type = select_shop_card_type(
            rng,
            ante,
            joker_rate=gs.get("joker_rate", 20.0),
            tarot_rate=gs.get("tarot_rate", 4.0),
            planet_rate=gs.get("planet_rate", 4.0),
            spectral_rate=gs.get("spectral_rate", 0.0),
            playing_card_rate=gs.get("playing_card_rate", 0.0),
        )
        card = create_card(
            card_type,
            rng,
            ante,
            area="shop",
            append="sho",
            game_state=gs,
        )
        new_cards.append(card)

    gs["shop_cards"] = new_cards
    # Pending edition tags also claim rerolled jokers (vanilla behavior).
    _fire_shop_tags(gs, rerolled=True)


def _get_card_set(card: Any) -> str:
    """Get the set name from a Card's ability dict."""
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("set", "")
    return ""


def _fire_shop_joker_context(gs: dict[str, Any], **context_flags: Any) -> list[dict[str, Any]]:
    """Fire a joker context during shop phase and return mutations.

    Accepts keyword arguments matching :class:`JokerContext` flags
    (e.g. ``buying_card=True``, ``reroll_shop=True``).
    """
    from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker

    jokers: list = gs.get("jokers", [])
    if not jokers:
        return []

    game_snap = GameSnapshot(
        joker_count=len(jokers),
        joker_slots=gs.get("joker_slots", 5),
        money=gs.get("dollars", 0),
    )

    # Extract 'cards' from flags if present (for playing_card_added)
    cards_arg = context_flags.pop("cards", None)

    mutations: list[dict[str, Any]] = []
    for joker in list(jokers):  # copy to allow mutation during iteration
        if getattr(joker, "debuff", False):
            continue
        ctx = JokerContext(
            jokers=jokers,
            game=game_snap,
            rng=gs.get("rng"),
            **context_flags,
        )
        if cards_arg is not None:
            ctx.cards = cards_arg
        result = calculate_joker(joker, ctx)
        if result and result.extra:
            mutations.append(result.extra)

    return mutations


def _apply_shop_mutations(
    gs: dict[str, Any],
    mutations: list[dict[str, Any]],
) -> None:
    """Process side-effect dicts from shop joker contexts.

    Handles Perkeo's consumable_copy creation.
    """
    for mut in mutations:
        if "create" in mut:
            create = mut["create"]
            ctype = create.get("type", "")

            if ctype == "consumable_copy":
                # Perkeo: copy a random consumable with Negative edition
                consumables: list = gs.get("consumables", [])
                if consumables:
                    rng = gs.get("rng")
                    if rng:
                        import copy

                        seed_val = rng.seed("perkeo")
                        original, _ = rng.element(consumables, seed_val)
                        duplicate = copy.copy(original)
                        duplicate.edition = {"negative": True}
                        consumables.append(duplicate)


# ---------------------------------------------------------------------------
# Pack close helper
# ---------------------------------------------------------------------------


def _close_pack(gs: dict[str, Any]) -> None:
    """Close the current booster pack and return to the previous phase.

    Matches ``end_consumeable`` in ``button_callbacks.lua:2565``:
    - Remove remaining pack cards
    - Return dealt hand cards to deck (Arcana/Spectral packs deal a hand)
    - Fire ``new_blind_choice`` tags (deferred from skip)
    - Restore phase from ``shop_return_phase``
    """
    # Clear pack state
    gs["pack_cards"] = []
    gs["pack_choices_remaining"] = 0

    # Return dealt hand cards to deck (Arcana/Spectral packs deal from deck)
    # Lua's draw_from_hand_to_deck (state_events.lua:1121-1126) removes
    # first card from hand and inserts at position 1 (front) of deck,
    # repeated for all cards.  Net effect: hand cards end up REVERSED
    # at the FRONT of the deck.
    pack_hand: list = gs.get("pack_hand", [])
    if pack_hand:
        deck: list = gs.setdefault("deck", [])
        hand: list = gs.get("hand", [])
        hand_set = set(id(c) for c in hand)
        # Only return cards that are still in the hand (not destroyed)
        surviving = [c for c in pack_hand if id(c) in hand_set]
        deck[:0] = list(reversed(surviving))
        # Remove only the pack_hand cards from hand, preserving any
        # cards that were there before the pack opened.
        pack_ids = set(id(c) for c in pack_hand)
        gs["hand"] = [c for c in hand if id(c) not in pack_ids]
        gs["pack_hand"] = []

    # Restore phase
    gs["phase"] = gs.get("shop_return_phase", GamePhase.SHOP)


# ---------------------------------------------------------------------------
# Blind:press_play — blind.lua:464
# ---------------------------------------------------------------------------


def _press_play(
    gs: dict[str, Any],
    blind: Any,
    played: list,
    rng: Any,
) -> None:
    """Fire boss blind press_play effects before scoring.

    Mirrors ``Blind:press_play`` (blind.lua:464-502).
    """
    if getattr(blind, "disabled", False):
        return

    name = getattr(blind, "name", "")

    if name == "The Hook":
        # Discard 2 random cards from hand
        hand: list = gs.get("hand", [])
        discard_pile: list = gs.setdefault("discard_pile", [])
        for _ in range(min(2, len(hand))):
            if hand and rng:
                seed_val = rng.seed("hook")
                target, _ = rng.element(hand, seed_val)
                hand.remove(target)
                discard_pile.append(target)

    elif name == "The Tooth":
        # Lose $1 per card played
        gs["dollars"] = gs.get("dollars", 0) - len(played)

    elif name == "The Fish":
        # Flip all hand cards face-down after play (blind.lua:494-496)
        blind.prepped = True

    elif name == "Crimson Heart":
        # Debuff a random joker each hand (blind.lua:488-493)
        jokers: list = gs.get("jokers", [])
        if jokers and rng:
            blind.triggered = True
            blind.prepped = True


# ---------------------------------------------------------------------------
# Double Tag check
# ---------------------------------------------------------------------------


def _check_double_tag(gs: dict[str, Any], awarded_tag_key: str) -> None:
    """If player has a Double Tag active, duplicate the just-awarded tag."""
    tags: list = gs.get("tags", [])
    if not tags:
        return

    # Check if any active tag is tag_double
    from jackdaw.engine.tags import Tag

    for i, tag_entry in enumerate(tags):
        tag_key = tag_entry if isinstance(tag_entry, str) else getattr(tag_entry, "key", "")
        if tag_key == "tag_double" and awarded_tag_key != "tag_double":
            # Fire the duplicate
            dup_tag = Tag(awarded_tag_key)
            dup_result = dup_tag.apply("immediate", gs, rng=gs.get("rng"))

            awarded_tags: list = gs.setdefault("awarded_tags", [])
            awarded_tags.append(
                {
                    "key": awarded_tag_key,
                    "result": dup_result,
                    "blind": "double",
                }
            )

            if dup_result and dup_result.dollars:
                gs["dollars"] = gs.get("dollars", 0) + dup_result.dollars

            # Remove the Double Tag (consumed)
            tags.pop(i)
            break
