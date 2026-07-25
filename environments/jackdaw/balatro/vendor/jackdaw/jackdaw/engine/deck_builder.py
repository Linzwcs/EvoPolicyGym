"""Deck builder — constructs the starting deck for a run.

Matches the deck building logic in ``game.lua:2328-2375`` and the
post-creation suit changes from ``Back:apply_to_run`` (back.lua:239).

Source references:
  - game.lua:2328-2375 (card_protos loop)
  - game.lua:2367-2369 (deterministic sort)
  - misc_functions.lua:1625 (card_from_control)
  - back.lua:239-253 (Checkered Deck suit swap)
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import card_from_control
from jackdaw.engine.data.enums import Suit
from jackdaw.engine.data.prototypes import BACKS, PLAYING_CARDS
from jackdaw.engine.rng import PseudoRandom


def build_deck(
    back_key: str,
    rng: PseudoRandom,
    challenge: dict[str, Any] | None = None,
    starting_params: dict[str, Any] | None = None,
) -> list[Card]:
    """Build the starting deck for a run.

    Args:
        back_key: P_CENTERS back key (e.g. ``"b_red"``, ``"b_abandoned"``).
        rng: The run's PseudoRandom instance.
        challenge: Challenge definition dict (with ``deck`` sub-dict) or None.
        starting_params: Effective starting parameters (needs ``no_faces``,
            ``erratic_suits_and_ranks``).  If None, derived from the back's config.

    Returns:
        List of Card objects forming the starting deck.
    """
    back = BACKS.get(back_key)
    back_config = back.config if back else {}

    # Resolve starting params
    params = starting_params or {}
    no_faces = params.get("no_faces", back_config.get("remove_faces", False))
    erratic = params.get(
        "erratic_suits_and_ranks",
        back_config.get("randomize_rank_suit", False),
    )

    # Challenge deck override
    challenge_deck: dict[str, Any] | None = None
    if challenge and "deck" in challenge:
        challenge_deck = challenge["deck"]

    # -------------------------------------------------------------------
    # Step 1: Build card_protos
    # -------------------------------------------------------------------
    card_protos: list[dict[str, str | None]]

    if challenge_deck and "cards" in challenge_deck:
        # Explicit card list from challenge
        card_protos = list(challenge_deck["cards"])
    else:
        card_protos = []
        # Iterate all 52 standard P_CARDS entries
        for card_key in PLAYING_CARDS:
            key = card_key

            # Erratic Deck: replace with random P_CARDS key
            if erratic:
                _, key = rng.element(
                    {k: k for k in PLAYING_CARDS},
                    rng.seed("erratic"),
                )

            # Parse suit letter and rank letter from key like "S_A"
            _s = key[0]  # suit letter
            _r = key[2]  # rank letter

            keep = True
            _e: str | None = None
            _d: str | None = None
            _g: str | None = None

            # Challenge filtering
            if challenge_deck:
                yes_ranks = challenge_deck.get("yes_ranks")
                if yes_ranks and _r not in yes_ranks:
                    keep = False
                no_ranks = challenge_deck.get("no_ranks")
                if no_ranks and _r in no_ranks:
                    keep = False
                yes_suits = challenge_deck.get("yes_suits")
                if yes_suits and _s not in yes_suits:
                    keep = False
                no_suits = challenge_deck.get("no_suits")
                if no_suits and _s in no_suits:
                    keep = False
                if "enhancement" in challenge_deck:
                    _e = challenge_deck["enhancement"]
                if "edition" in challenge_deck:
                    _d = challenge_deck["edition"]
                if "gold_seal" in challenge_deck:
                    _g = challenge_deck["gold_seal"]

            # Abandoned Deck: skip face cards
            if no_faces and _r in ("K", "Q", "J"):
                keep = False

            if keep:
                card_protos.append({"s": _s, "r": _r, "e": _e, "d": _d, "g": _g})

    # -------------------------------------------------------------------
    # Step 2: Deterministic sort (game.lua:2367)
    # -------------------------------------------------------------------
    def _sort_key(proto: dict) -> str:
        return (
            (proto.get("s") or "")
            + (proto.get("r") or "")
            + (proto.get("e") or "")
            + (proto.get("d") or "")
            + (proto.get("g") or "")
        )

    card_protos.sort(key=_sort_key)

    # -------------------------------------------------------------------
    # Step 3: Create Card objects
    # -------------------------------------------------------------------
    cards: list[Card] = []
    for i, proto in enumerate(card_protos):
        card = card_from_control(proto, playing_card_index=i + 1)
        cards.append(card)

    # -------------------------------------------------------------------
    # Step 4: Post-creation deck mutations
    # -------------------------------------------------------------------

    # Checkered Deck: Clubs→Spades, Diamonds→Hearts (back.lua:239-253)
    if back_key == "b_checkered":
        _apply_checkered(cards)

    return cards


def _apply_checkered(cards: list[Card]) -> None:
    """Apply Checkered Deck suit swaps: Clubs→Spades, Diamonds→Hearts."""
    for card in cards:
        if card.base is None:
            continue
        if card.base.suit is Suit.CLUBS:
            _change_suit(card, Suit.SPADES)
        elif card.base.suit is Suit.DIAMONDS:
            _change_suit(card, Suit.HEARTS)


def _change_suit(card: Card, new_suit: Suit) -> None:
    """Change a card's suit, updating all base fields.

    Matches ``Card:change_suit`` (card.lua:547).
    """
    if card.base is None:
        return

    # Suit nominal values
    suit_nominals = {
        Suit.DIAMONDS: (0.01, 0.001),
        Suit.CLUBS: (0.02, 0.002),
        Suit.HEARTS: (0.03, 0.003),
        Suit.SPADES: (0.04, 0.004),
    }

    card.base.suit = new_suit
    nom, nom_orig = suit_nominals[new_suit]
    card.base.suit_nominal = nom
    card.base.suit_nominal_original = nom_orig
