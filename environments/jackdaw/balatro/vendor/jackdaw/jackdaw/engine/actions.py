"""Player action types and game phase enum.

Every decision point in a Balatro run is represented as a frozen dataclass
action.  The simulator's step function accepts an :data:`Action` and
advances the game state accordingly.

Game phases
-----------
:class:`GamePhase` enumerates the phases where player input is needed.
Each phase permits a specific subset of actions:

+---------------------+------------------------------------------------------+
| Phase               | Valid actions                                         |
+=====================+======================================================+
| BLIND_SELECT        | SelectBlind, SkipBlind                               |
+---------------------+------------------------------------------------------+
| SELECTING_HAND      | PlayHand, Discard, SortHand, UseConsumable,          |
|                     | SwapHandLeft/Right, SwapJokersLeft/Right             |
+---------------------+------------------------------------------------------+
| SHOP                | BuyCard, SellCard, UseConsumable, RedeemVoucher,     |
|                     | OpenBooster, Reroll, SwapJokersLeft/Right, NextRound |
+---------------------+------------------------------------------------------+
| PACK_OPENING        | PickPackCard, SkipPack                               |
+---------------------+------------------------------------------------------+
| ROUND_EVAL          | CashOut                                              |
+---------------------+------------------------------------------------------+
| GAME_OVER           | *(terminal — no valid actions)*                      |
+---------------------+------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------


class GamePhase(StrEnum):
    """Phases where player input is needed.

    Maps to ``G.STATES`` in the Lua source, filtered to decision points.
    """

    BLIND_SELECT = "blind_select"
    SELECTING_HAND = "selecting_hand"
    ROUND_EVAL = "round_eval"
    SHOP = "shop"
    PACK_OPENING = "pack_opening"
    GAME_OVER = "game_over"


# ---------------------------------------------------------------------------
# Blind selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectBlind:
    """Accept the current blind and start the round."""


@dataclass(frozen=True)
class SkipBlind:
    """Skip the current blind (Small or Big) and collect the tag reward."""


# ---------------------------------------------------------------------------
# Hand play / discard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayHand:
    """Play selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    """

    card_indices: tuple[int, ...]


@dataclass(frozen=True)
class Discard:
    """Discard selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    """

    card_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Shop actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyCard:
    """Purchase a card from the shop.

    ``shop_index`` is the 0-based index into the shop card area
    (jokers, tarots, planets, or playing cards depending on slot).
    """

    shop_index: int


@dataclass(frozen=True)
class SellCard:
    """Sell a card from jokers or consumables for its sell value.

    ``area`` is ``'jokers'`` or ``'consumables'``.
    """

    area: str
    card_index: int


@dataclass(frozen=True)
class UseConsumable:
    """Use a consumable from the consumable area.

    ``target_indices`` are 0-based indices into ``hand.cards`` for
    targeted consumables (tarots that modify specific cards).
    ``None`` for untargeted consumables (planets, most spectrals).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RedeemVoucher:
    """Purchase and activate a voucher from the shop."""

    card_index: int


@dataclass(frozen=True)
class OpenBooster:
    """Open a booster pack from the shop."""

    card_index: int


@dataclass(frozen=True)
class Reroll:
    """Reroll the shop (costs reroll_cost dollars)."""


@dataclass(frozen=True)
class NextRound:
    """Leave the shop and proceed to the next round."""


# ---------------------------------------------------------------------------
# Pack opening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickPackCard:
    """Select a card from an opened booster pack.

    For Arcana/Spectral packs, ``target_indices`` specifies which dealt
    hand cards the consumable should target (e.g. for The Magician
    enhancing up to 2 cards).  ``None`` for non-targeted picks
    (Celestial, Standard, Buffoon packs).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SkipPack:
    """Skip remaining picks in a booster pack."""


# ---------------------------------------------------------------------------
# Round evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashOut:
    """Accept end-of-round earnings and proceed to the shop."""


# ---------------------------------------------------------------------------
# Utility actions (available in multiple phases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortHand:
    """Sort the hand by rank or suit.

    ``mode`` is ``'rank'`` or ``'suit'``.
    """

    mode: str


@dataclass(frozen=True)
class SwapHandLeft:
    """Swap a hand card one position to the left.

    Swaps ``hand[idx]`` with ``hand[idx - 1]``.  Free action — no cost,
    doesn't consume hands or discards.  Left-to-right order affects which
    card is "first scored" for jokers like Photograph and Hanging Chad.
    """

    idx: int


@dataclass(frozen=True)
class SwapHandRight:
    """Swap a hand card one position to the right.

    Swaps ``hand[idx]`` with ``hand[idx + 1]``.  Free action.
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersLeft:
    """Swap a joker one position to the left.

    Swaps ``jokers[idx]`` with ``jokers[idx - 1]``.
    Left-to-right order affects Phase 9 scoring (additive before
    multiplicative), Blueprint (copies right neighbor), Brainstorm
    (copies leftmost), Ceremonial Dagger (destroys right neighbor).
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersRight:
    """Swap a joker one position to the right.

    Swaps ``jokers[idx]`` with ``jokers[idx + 1]``.
    """

    idx: int


# ---------------------------------------------------------------------------
# Union type
# ---------------------------------------------------------------------------

Action = (
    PlayHand
    | Discard
    | SelectBlind
    | SkipBlind
    | BuyCard
    | SellCard
    | UseConsumable
    | RedeemVoucher
    | OpenBooster
    | PickPackCard
    | SkipPack
    | Reroll
    | NextRound
    | CashOut
    | SortHand
    | SwapHandLeft
    | SwapHandRight
    | SwapJokersLeft
    | SwapJokersRight
)


# ---------------------------------------------------------------------------
# Legal action generation
# ---------------------------------------------------------------------------


def get_legal_actions(game_state: dict[str, Any]) -> list[Action]:
    """Return every valid action from the current game state.

    The SELECTING_HAND phase has a combinatorial action space —
    ``PlayHand`` and ``Discard`` each admit up to C(n, 5) card
    subsets.  This function does **not** enumerate every possible
    subset.  Instead it returns *marker* actions with empty
    ``card_indices``, signalling that play/discard is legal.
    The Gym env (M13) will handle subset enumeration or
    parameterised action spaces.

    Parameters
    ----------
    game_state:
        The full runtime state dict.  Must include:

        * ``phase`` (:class:`GamePhase`) — current game phase.
        * ``current_round`` (dict) — ``hands_left``, ``discards_left``,
          ``reroll_cost``, ``free_rerolls``.
        * ``dollars`` (int) — current bank balance.
        * ``round_resets`` (dict) — ``blind_states`` dict.
        * ``blind_on_deck`` (str | None) — ``"Small"``/``"Big"``/``"Boss"``.
        * ``hand`` (list[Card]) — cards in the player's hand.
        * ``jokers`` (list[Card]) — active joker cards.
        * ``consumables`` (list[Card]) — owned consumable cards.
        * ``joker_slots`` (int) — max joker capacity.
        * ``consumable_slots`` (int) — max consumable capacity.
        * ``shop_cards`` (list[Card]) — buyable cards in shop.
        * ``shop_vouchers`` (list[Card]) — buyable vouchers.
        * ``shop_boosters`` (list[Card]) — buyable booster packs.
        * ``pack_cards`` (list[Card]) — cards in an opened pack.
        * ``pack_choices_remaining`` (int) — picks left in pack.

    Returns
    -------
    list[Action]
        All legal actions.  Empty for ``GAME_OVER``.
    """
    phase = game_state.get("phase")
    if phase is None:
        return []

    if isinstance(phase, str):
        phase = GamePhase(phase)

    if phase == GamePhase.GAME_OVER:
        return []
    if phase == GamePhase.BLIND_SELECT:
        return _legal_blind_select(game_state)
    if phase == GamePhase.SELECTING_HAND:
        return _legal_selecting_hand(game_state)
    if phase == GamePhase.ROUND_EVAL:
        return _legal_round_eval(game_state)
    if phase == GamePhase.SHOP:
        return _legal_shop(game_state)
    if phase == GamePhase.PACK_OPENING:
        return _legal_pack_opening(game_state)
    return []


# ---------------------------------------------------------------------------
# Per-phase helpers
# ---------------------------------------------------------------------------


def _legal_blind_select(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    blind_on_deck = gs.get("blind_on_deck", "Small")

    actions.append(SelectBlind())

    # Can skip Small and Big, but not Boss
    if blind_on_deck in ("Small", "Big"):
        actions.append(SkipBlind())

    # Consumables usable during blind select
    actions.extend(_usable_consumables(gs))
    return actions


def _legal_selecting_hand(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    cr = gs.get("current_round", {})
    hand: list[Card] = gs.get("hand", [])

    # PlayHand: marker action — at least 1 card in hand + hands remaining
    if hand and cr.get("hands_left", 0) > 0:
        actions.append(PlayHand(card_indices=()))

    # Discard: marker action — at least 1 card in hand + discards remaining
    if hand and cr.get("discards_left", 0) > 0:
        actions.append(Discard(card_indices=()))

    # Sort and reorder hand
    if len(hand) > 1:
        actions.append(SortHand(mode="rank"))
        actions.append(SortHand(mode="suit"))
        for i in range(1, len(hand)):
            actions.append(SwapHandLeft(idx=i))
        for i in range(len(hand) - 1):
            actions.append(SwapHandRight(idx=i))

    # Consumables usable during hand selection
    actions.extend(_usable_consumables(gs))

    # Swap jokers
    jokers: list[Card] = gs.get("jokers", [])
    if len(jokers) > 1:
        for i in range(1, len(jokers)):
            actions.append(SwapJokersLeft(idx=i))
        for i in range(len(jokers) - 1):
            actions.append(SwapJokersRight(idx=i))

    return actions


def _legal_round_eval(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = [CashOut()]
    actions.extend(_usable_consumables(gs))
    return actions


def _legal_shop(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    dollars: int = gs.get("dollars", 0)
    jokers: list[Card] = gs.get("jokers", [])
    joker_slots: int = gs.get("joker_slots", 5)
    consumables: list[Card] = gs.get("consumables", [])
    consumable_slots: int = gs.get("consumable_slots", 2)

    # Buy cards from shop
    shop_cards: list[Card] = gs.get("shop_cards", [])
    for i, card in enumerate(shop_cards):
        if card.cost > dollars:
            continue
        # Check slot availability
        card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
        if card_set == "Joker":
            is_negative = isinstance(card.edition, dict) and card.edition.get("negative")
            if len(jokers) >= joker_slots and not is_negative:
                continue
        elif card_set in ("Tarot", "Planet", "Spectral"):
            if len(consumables) >= consumable_slots:
                continue  # No room — agent must use/sell a consumable first
        actions.append(BuyCard(shop_index=i))

    # Sell jokers (non-eternal)
    for i, joker in enumerate(jokers):
        if not joker.eternal:
            actions.append(SellCard(area="jokers", card_index=i))

    # Sell consumables
    for i in range(len(consumables)):
        actions.append(SellCard(area="consumables", card_index=i))

    # Use owned consumables
    actions.extend(_usable_consumables(gs))

    # Redeem vouchers
    shop_vouchers: list[Card] = gs.get("shop_vouchers", [])
    for i, voucher in enumerate(shop_vouchers):
        if voucher.cost <= dollars:
            actions.append(RedeemVoucher(card_index=i))

    # Open boosters
    shop_boosters: list[Card] = gs.get("shop_boosters", [])
    for i, booster in enumerate(shop_boosters):
        if booster.cost <= dollars:
            actions.append(OpenBooster(card_index=i))

    # Reroll
    cr = gs.get("current_round", {})
    reroll_cost = cr.get("reroll_cost", 5)
    free_rerolls = cr.get("free_rerolls", 0)
    if free_rerolls > 0 or dollars >= reroll_cost:
        actions.append(Reroll())

    # Next round (always available in shop)
    actions.append(NextRound())

    # Swap jokers
    if len(jokers) > 1:
        for i in range(1, len(jokers)):
            actions.append(SwapJokersLeft(idx=i))
        for i in range(len(jokers) - 1):
            actions.append(SwapJokersRight(idx=i))

    return actions


def _legal_pack_opening(gs: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    pack_cards: list[Card] = gs.get("pack_cards", [])
    remaining: int = gs.get("pack_choices_remaining", 0)
    pack_type: str = gs.get("pack_type", "")

    # Spectral packs: balatrobot cannot handle Spectral card highlighting
    # via RPC, so only SkipPack is offered.  RNG stays in sync because
    # the pack is generated normally — we just force the agent to skip.
    if remaining > 0 and pack_type != "Spectral":
        # Vanilla gate (button_callbacks.lua:2112-2113): a Joker in a pack
        # is selectable only if joker slots have room or it is negative.
        jokers: list[Card] = gs.get("jokers", [])
        joker_limit: int = gs.get("joker_slots", 5)
        board_full = len(jokers) >= joker_limit
        for i, card in enumerate(pack_cards):
            if board_full:
                ability = getattr(card, "ability", None) or {}
                edition = getattr(card, "edition", None) or {}
                is_joker = ability.get("set", "") == "Joker"
                negative = bool(edition.get("negative")) if isinstance(edition, dict) else False
                if is_joker and not negative:
                    continue
            actions.append(PickPackCard(card_index=i))

    actions.append(SkipPack())
    return actions


# ---------------------------------------------------------------------------
# Consumable usability helper
# ---------------------------------------------------------------------------


def _usable_consumables(gs: dict[str, Any]) -> list[Action]:
    """Return UseConsumable actions for each usable owned consumable."""
    consumables: list[Card] = gs.get("consumables", [])
    if not consumables:
        return []

    from jackdaw.engine.consumables import can_use_consumable

    actions: list[Action] = []
    hand: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    joker_limit: int = gs.get("joker_slots", 5)
    consumable_limit: int = gs.get("consumable_slots", 2)

    for i, card in enumerate(consumables):
        if can_use_consumable(
            card,
            hand_cards=hand,
            jokers=jokers,
            consumables=consumables,
            consumable_limit=consumable_limit,
            joker_limit=joker_limit,
        ):
            actions.append(UseConsumable(card_index=i))
    return actions
