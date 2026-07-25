"""Per-round card maintenance and targeting card reset.

Round-end card maintenance
--------------------------
Fires at end of round after joker ``calculate_joker({end_of_round=true})``
but before ``evaluate_round`` (the cash-out calculation in :mod:`economy`).

Source: ``state_events.lua:87-110`` — the ``end_round`` function calls
``calculate_joker``, then ``calculate_rental``, then ``calculate_perishable``
for each joker in order.

* **Perishable countdown**: ``perish_tally`` decrements each round;
  when it hits 0 the card is permanently debuffed.
* **Rental charges**: each rental card costs ``rental_rate`` ($3) per round,
  deducted directly from ``game_state["dollars"]``.

Glass Card behaviour: only shatters during scoring (Phase 11), not at
round end.

Targeting card reset
--------------------
Called at the start of each round.  Four jokers read per-round targeting
cards from ``game_state["current_round"]``:

- **The Idol** (``j_idol``): ``idol_card`` — suit + rank
- **Mail-In Rebate** (``j_mail``): ``mail_card`` — rank only
- **Ancient Joker** (``j_ancient``): ``ancient_card`` — suit only
- **Castle** (``j_castle``): ``castle_card`` — suit only

Source: ``common_events.lua:2271-2324``.

Selection mechanism: Idol, Mail, Castle pick a random non-Stone playing
card from the deck via ``pseudorandom_element``.  Ancient picks from the
3 suits that differ from the current ``ancient_card.suit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom


@dataclass
class RoundEndResult:
    """Descriptor returned by :func:`process_round_end_cards`.

    All lists contain references to the actual Card objects that were
    affected — callers can inspect or remove them as needed.
    """

    perished: list[Card] = field(default_factory=list)
    """Jokers whose ``perish_tally`` hit 0 this round and became debuffed."""

    rental_cost: int = 0
    """Total dollars deducted for rental cards this round."""

    rental_cards: list[Card] = field(default_factory=list)
    """Cards that incurred a rental charge."""


def process_round_end_cards(
    jokers: list[Card],
    game_state: dict[str, Any],
) -> RoundEndResult:
    """Process card maintenance at end of round.

    Called after :func:`~jackdaw.engine.jokers.on_end_of_round` and before
    :func:`~jackdaw.engine.economy.calculate_round_earnings`.

    Mirrors the per-joker loop in ``state_events.lua:99-109``::

        for i = 1, #G.jokers.cards do
            G.jokers.cards[i]:calculate_joker({end_of_round = true, ...})
            G.jokers.cards[i]:calculate_rental()
            G.jokers.cards[i]:calculate_perishable()
        end

    The ``calculate_joker`` call is handled by
    :func:`~jackdaw.engine.jokers.on_end_of_round`;
    this function handles the remaining two steps.

    Parameters
    ----------
    jokers:
        Active joker cards.  Mutated in place (``perish_tally``,
        ``debuff`` fields).
    game_state:
        Mutable run-state dict.  ``game_state["dollars"]`` is decremented
        by rental charges.  Reads ``game_state.get("rental_rate", 3)``.

    Returns
    -------
    RoundEndResult
        Summary of what happened.
    """
    rental_rate: int = game_state.get("rental_rate", 3)
    result = RoundEndResult()

    for joker in jokers:
        # ---------------------------------------------------------------
        # calculate_rental — card.lua:2271-2276
        # ---------------------------------------------------------------
        if _is_rental(joker):
            game_state["dollars"] = game_state.get("dollars", 0) - rental_rate
            result.rental_cost += rental_rate
            result.rental_cards.append(joker)

        # ---------------------------------------------------------------
        # calculate_perishable — card.lua:2278-2289
        # ---------------------------------------------------------------
        if _is_perishable(joker):
            tally = _get_perish_tally(joker)
            if tally > 0:
                if tally == 1:
                    # Hits zero — permanently debuff
                    _set_perish_tally(joker, 0)
                    joker.debuff = True
                    result.perished.append(joker)
                else:
                    _set_perish_tally(joker, tally - 1)

    return result


# ---------------------------------------------------------------------------
# Card field accessors — handle both attribute and dict-based ability
# ---------------------------------------------------------------------------
# The Card dataclass stores perishable/rental as top-level fields AND
# historically some code stores them in card.ability dict (matching Lua's
# self.ability.perishable).  We check both for robustness.


def _is_rental(card: Card) -> bool:
    if getattr(card, "rental", False):
        return True
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return bool(ability.get("rental"))
    return False


def _is_perishable(card: Card) -> bool:
    if getattr(card, "perishable", False):
        return True
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return bool(ability.get("perishable"))
    return False


def _get_perish_tally(card: Card) -> int:
    # Prefer card.ability.perish_tally (matches Lua) then card.perish_tally
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict) and "perish_tally" in ability:
        return ability["perish_tally"]
    return getattr(card, "perish_tally", 0)


def _set_perish_tally(card: Card, value: int) -> None:
    # Update both locations for consistency
    if isinstance(getattr(card, "ability", None), dict):
        card.ability["perish_tally"] = value
    card.perish_tally = value


# ---------------------------------------------------------------------------
# Targeting card reset — common_events.lua:2271-2324
# ---------------------------------------------------------------------------

_ALL_SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]


def reset_round_targets(
    rng: PseudoRandom,
    ante: int,
    game_state: dict[str, Any],
) -> None:
    """Reset per-round targeting cards used by specific jokers.

    Called at the start of each round (and during run init).  Mutates
    ``game_state["current_round"]`` in place.

    Mirrors the four Lua functions in ``common_events.lua:2271-2324``:

    * ``reset_idol_card()``   — picks a random non-Stone card → rank + suit
    * ``reset_mail_rank()``   — picks a random non-Stone card → rank only
    * ``reset_ancient_card()``— picks from 3 suits ≠ current → suit only
    * ``reset_castle_card()`` — picks a random non-Stone card → suit only

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
        Each call advances the named RNG stream for that target type.
    ante:
        Current ante number, appended to the seed key for per-ante
        stream independence (e.g. ``'idol1'``, ``'mail2'``).
    game_state:
        Mutable run-state dict.  Must have ``current_round`` sub-dict
        and ``deck`` list of Card objects.
    """
    cr = game_state["current_round"]
    # Vanilla iterates G.playing_cards — every playing card in the run
    # regardless of zone (deck / hand / discard).  rng.element re-sorts by
    # sort_id, so zone concatenation order is irrelevant.
    all_cards: list[Card] = [
        *game_state.get("deck", []),
        *game_state.get("hand", []),
        *game_state.get("discard_pile", []),
    ]

    # Filter out Stone cards — only non-Stone playing cards are eligible.
    # Also skip any card without a base (cannot occur in vanilla; defends
    # the mail.base.id read below against sim-side creation bugs).
    valid_cards = [
        c for c in all_cards if _card_effect(c) != "Stone Card" and c.base is not None
    ]

    # ------------------------------------------------------------------
    # reset_idol_card — common_events.lua:2271-2286
    # One pseudorandom_element call → card's rank AND suit
    # ------------------------------------------------------------------
    cr["idol_card"] = {"suit": "Spades", "rank": "Ace"}
    if valid_cards:
        seed_val = rng.seed("idol" + str(ante))
        idol, _ = rng.element(valid_cards, seed_val)
        cr["idol_card"]["rank"] = _card_rank_str(idol)
        cr["idol_card"]["suit"] = _card_suit_str(idol)

    # ------------------------------------------------------------------
    # reset_mail_rank — common_events.lua:2288-2301
    # One pseudorandom_element call → card's rank only
    # ------------------------------------------------------------------
    cr["mail_card"] = {"rank": "Ace", "id": 14}
    if valid_cards:
        seed_val = rng.seed("mail" + str(ante))
        mail, _ = rng.element(valid_cards, seed_val)
        cr["mail_card"]["rank"] = _card_rank_str(mail)
        cr["mail_card"]["id"] = mail.base.id

    # ------------------------------------------------------------------
    # reset_ancient_card — common_events.lua:2303-2310
    # Picks from 3 suits ≠ current suit.  Does NOT use the deck.
    # ------------------------------------------------------------------
    old_suit = cr.get("ancient_card", {}).get("suit")
    cr["ancient_card"] = {"suit": "Spades"}
    if old_suit is None:
        # First call in a run: game.lua:2387 sets suit to nil before reset
        ancient_suits = list(_ALL_SUITS)
    else:
        ancient_suits = [s for s in _ALL_SUITS if s != old_suit]
    seed_val = rng.seed("anc" + str(ante))
    chosen_suit, _ = rng.element(ancient_suits, seed_val)
    cr["ancient_card"]["suit"] = chosen_suit

    # ------------------------------------------------------------------
    # reset_castle_card — common_events.lua:2312-2324
    # One pseudorandom_element call → card's suit only
    # ------------------------------------------------------------------
    cr["castle_card"] = {"suit": "Spades"}
    if valid_cards:
        seed_val = rng.seed("cas" + str(ante))
        castle, _ = rng.element(valid_cards, seed_val)
        cr["castle_card"]["suit"] = _card_suit_str(castle)


# ---------------------------------------------------------------------------
# Card field helpers for targeting card reset
# ---------------------------------------------------------------------------


def _card_effect(card: Card) -> str:
    """Get the enhancement effect string from a Card object."""
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("effect", "") or ""
    return ""


def _card_rank_str(card: Card) -> str:
    """Get the rank display string (e.g. ``'Ace'``, ``'10'``) from a Card."""
    base = getattr(card, "base", None)
    if base is not None:
        rank = base.rank
        return rank.value if hasattr(rank, "value") else str(rank)
    return "Ace"


def _card_suit_str(card: Card) -> str:
    """Get the suit display string (e.g. ``'Spades'``) from a Card."""
    base = getattr(card, "base", None)
    if base is not None:
        suit = base.suit
        return suit.value if hasattr(suit, "value") else str(suit)
    return "Spades"
