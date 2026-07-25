"""CardArea — container for collections of cards.

Replaces the Lua ``CardArea`` class (cardarea.lua) with all rendering
and layout logic stripped.  Keeps only the data management operations
the simulator needs: add, remove, shuffle, sort, draw, highlight.

Source: cardarea.lua lines 5-668.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom


class CardArea:
    """A container managing a list of :class:`Card` objects.

    Corresponds to ``G.hand``, ``G.deck``, ``G.jokers``, ``G.consumeables``,
    ``G.play``, ``G.discard``, ``G.shop_jokers``, etc.

    Args:
        card_limit: Maximum number of cards this area can hold.
        area_type: Identifies the area's role.  One of ``'hand'``, ``'deck'``,
            ``'play'``, ``'discard'``, ``'joker'``, ``'consumeable'``,
            ``'shop'``, ``'voucher'``, ``'title'``, ``'title_2'``.
    """

    __slots__ = ("cards", "highlighted", "config", "type")

    def __init__(self, card_limit: int = 52, area_type: str = "deck") -> None:
        self.cards: list[Card] = []
        self.highlighted: list[Card] = []
        self.config: dict = {"card_limit": card_limit}
        self.type: str = area_type

    # -- card_limit property ------------------------------------------------

    @property
    def card_limit(self) -> int:
        return self.config["card_limit"]

    @card_limit.setter
    def card_limit(self, value: int) -> None:
        self.config["card_limit"] = max(0, value)

    # -- basic operations ---------------------------------------------------

    def add(self, card: Card, front: bool = False) -> None:
        """Add *card* to the area.

        Args:
            front: If True, insert at position 0 (used by deck emplace).
                   Default appends to the end.
        """
        if front:
            self.cards.insert(0, card)
        else:
            self.cards.append(card)

    def remove(self, card: Card) -> Card:
        """Remove *card* from this area and return it.

        Also removes the card from the highlighted list if present.
        """
        self.cards.remove(card)
        if card in self.highlighted:
            self.highlighted.remove(card)
        return card

    def remove_top(self) -> Card:
        """Remove and return the last card (top of a stack/deck)."""
        card = self.cards.pop()
        if card in self.highlighted:
            self.highlighted.remove(card)
        return card

    def has_space(self, negative_bonus: int = 0) -> bool:
        """Check if there's room for another card.

        Args:
            negative_bonus: Extra slots from Negative edition cards (+1 each).
        """
        return len(self.cards) < self.card_limit + negative_bonus

    # -- highlighting (for hand card selection) -----------------------------

    def add_to_highlighted(self, card: Card) -> None:
        """Mark a card as highlighted (selected for play/discard)."""
        if card not in self.highlighted:
            self.highlighted.append(card)

    def remove_from_highlighted(self, card: Card) -> None:
        if card in self.highlighted:
            self.highlighted.remove(card)

    def unhighlight_all(self) -> None:
        self.highlighted.clear()

    # -- shuffle / sort -----------------------------------------------------

    def shuffle(self, rng: PseudoRandom, seed_key: str) -> None:
        """Shuffle cards using the game's deterministic PRNG.

        Mirrors ``CardArea:shuffle`` (cardarea.lua:572).
        """
        sv = rng.seed(seed_key)
        rng.shuffle(self.cards, sv)

    def sort_by_value(self, descending: bool = True) -> None:
        """Sort cards by rank nominal value, matching ``CardArea:sort('desc')``."""
        self.cards.sort(
            key=lambda c: _card_nominal(c),
            reverse=descending,
        )

    def sort_by_suit(self, descending: bool = True) -> None:
        """Sort cards by suit then rank, matching ``CardArea:sort('suit desc')``."""
        self.cards.sort(
            key=lambda c: _card_nominal_suit(c),
            reverse=descending,
        )

    # -- info ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"CardArea(type={self.type!r}, cards={len(self.cards)}, limit={self.card_limit})"


# ---------------------------------------------------------------------------
# Sorting helpers (match Card:get_nominal in card.lua:950)
# ---------------------------------------------------------------------------


def _card_nominal(card: Card) -> float:
    """Sort key for rank-based sorting (``Card:get_nominal()``)."""
    if card.base is None:
        return 0.0
    b = card.base
    effect = card.ability.get("effect", "")
    stone_penalty = -1000.0 if effect == "Stone Card" else 0.0
    return (
        b.nominal
        + b.suit_nominal
        + (b.suit_nominal_original or 0) * 0.0001
        + b.face_nominal
        + 0.000001 * card.sort_id
        + stone_penalty
    )


def _card_nominal_suit(card: Card) -> float:
    """Sort key for suit-based sorting (``Card:get_nominal('suit')``)."""
    if card.base is None:
        return 0.0
    b = card.base
    effect = card.ability.get("effect", "")
    stone_penalty = -1000.0 if effect == "Stone Card" else 0.0
    return (
        b.nominal
        + b.suit_nominal * 1000
        + (b.suit_nominal_original or 0) * 0.0001 * 1000
        + b.face_nominal
        + 0.000001 * card.sort_id
        + stone_penalty
    )


# ---------------------------------------------------------------------------
# Standalone draw function
# ---------------------------------------------------------------------------


def draw_card(
    from_area: CardArea,
    to_area: CardArea,
    count: int = 1,
) -> list[Card]:
    """Move up to *count* cards from *from_area* to *to_area*.

    Draws from the end of ``from_area.cards`` (top of deck).
    Respects ``to_area.card_limit`` — stops if the target is full.
    For deck→hand draws, this matches the game's ``draw_card`` function
    (common_events.lua:386).

    Returns the list of cards drawn.
    """
    drawn: list[Card] = []
    for _ in range(count):
        if not from_area.cards:
            break
        if len(to_area.cards) >= to_area.card_limit:
            break
        card = from_area.cards.pop()
        to_area.cards.append(card)
        drawn.append(card)
    return drawn
