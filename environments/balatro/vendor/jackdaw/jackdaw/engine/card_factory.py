"""Card creation functions matching the Balatro source's card factory patterns.

Provides typed constructors for playing cards, jokers, consumables, and
vouchers, plus:

* ``card_from_control`` — deck-builder helper (misc_functions.lua:1625)
* ``create_card`` — the unified shop/pack factory (common_events.lua:2082)
* ``resolve_create_descriptor`` — side-effect descriptor → Card bridge (M10)
* ``resolve_destroy_descriptor`` — destruction descriptor → Card bridge (M10)

Source references:
  - card_from_control: misc_functions.lua:1625
  - create_card: common_events.lua:2082
  - Card:init, Card:set_base, Card:set_ability: card.lua:5-342
  - Madness destroy seed: card.lua:2509 (pseudoseed('madness'))
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from jackdaw.engine.card import Card
from jackdaw.engine.data.enums import Rank, Suit

if TYPE_CHECKING:
    from jackdaw.engine.rng import PseudoRandom

# ---------------------------------------------------------------------------
# Suit / rank letter mappings (used by card_from_control and deck building)
# ---------------------------------------------------------------------------

SUIT_LETTER: dict[str, Suit] = {
    "H": Suit.HEARTS,
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "S": Suit.SPADES,
}

RANK_LETTER: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

# Reverse mappings
SUIT_TO_LETTER: dict[Suit, str] = {v: k for k, v in SUIT_LETTER.items()}
RANK_TO_LETTER: dict[Rank, str] = {v: k for k, v in RANK_LETTER.items()}


def _card_key(suit: Suit, rank: Rank) -> str:
    """Build a P_CARDS key like ``'S_A'`` or ``'H_K'``."""
    return f"{SUIT_TO_LETTER[suit]}_{RANK_TO_LETTER[rank]}"


# ---------------------------------------------------------------------------
# Playing card creation
# ---------------------------------------------------------------------------


def create_playing_card(
    suit: Suit,
    rank: Rank,
    enhancement: str = "c_base",
    edition: dict[str, bool] | None = None,
    seal: str | None = None,
    *,
    playing_card_index: int | None = None,
    hands_played: int = 0,
) -> Card:
    """Create a playing card (goes into the deck).

    Mirrors the card creation in ``card_from_control`` (misc_functions.lua:1625)
    and ``Card:init`` (card.lua:5).

    Args:
        suit: Card suit.
        rank: Card rank.
        enhancement: P_CENTERS key for the enhancement center.
            ``"c_base"`` for a normal card, ``"m_glass"`` for Glass, etc.
        edition: Edition dict (``{"foil": True}``) or None.
        seal: Seal string (``"Gold"``, ``"Red"``, etc.) or None.
        playing_card_index: Index in ``G.playing_cards`` list.
        hands_played: Current ``G.GAME.hands_played`` for post-init fields.

    Returns:
        A fully initialised playing card.
    """
    card = Card()
    key = _card_key(suit, rank)
    card.set_base(key, suit.value, rank.value)
    card.set_ability(enhancement, hands_played=hands_played)
    card.playing_card = playing_card_index
    if edition:
        card.set_edition(edition)
    if seal:
        card.set_seal(seal)
    return card


# ---------------------------------------------------------------------------
# Joker creation
# ---------------------------------------------------------------------------


def create_joker(
    key: str,
    edition: dict[str, bool] | None = None,
    *,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
    hands_played: int = 0,
) -> Card:
    """Create a joker card from a P_CENTERS key (e.g. ``"j_joker"``).

    Args:
        key: P_CENTERS joker key.
        edition: Edition dict or None.
        eternal: Whether the joker has the Eternal sticker.
        perishable: Whether the joker has the Perishable sticker.
        rental: Whether the joker has the Rental sticker.
        hands_played: Current hands_played for post-init fields.
    """
    card = Card()
    card.set_ability(key, hands_played=hands_played)
    if edition:
        card.set_edition(edition)
    if eternal:
        card.set_eternal(True)
    if perishable:
        card.set_perishable(True)
    if rental:
        card.set_rental(True)
    return card


# ---------------------------------------------------------------------------
# Consumable creation (tarots, planets, spectrals)
# ---------------------------------------------------------------------------


def create_consumable(key: str, *, hands_played: int = 0) -> Card:
    """Create a tarot, planet, or spectral card from a P_CENTERS key.

    Args:
        key: P_CENTERS consumable key (e.g. ``"c_magician"``, ``"c_pluto"``).
        hands_played: Current hands_played for post-init fields.
    """
    card = Card()
    card.set_ability(key, hands_played=hands_played)
    return card


# ---------------------------------------------------------------------------
# Voucher creation
# ---------------------------------------------------------------------------


def create_voucher(key: str) -> Card:
    """Create a voucher card from a P_CENTERS key (e.g. ``"v_overstock_norm"``)."""
    card = Card()
    card.set_ability(key)
    return card


# ---------------------------------------------------------------------------
# card_from_control — deck building helper
# ---------------------------------------------------------------------------


def card_from_control(
    control: dict,
    *,
    playing_card_index: int | None = None,
    hands_played: int = 0,
) -> Card:
    """Create a playing card from a control dict.

    Matches ``card_from_control`` in ``misc_functions.lua:1625``.

    Control dict fields:
        - ``s``: suit letter (``'H'``/``'C'``/``'D'``/``'S'``)
        - ``r``: rank letter (``'2'``-``'9'``/``'T'``/``'J'``/``'Q'``/``'K'``/``'A'``)
        - ``e``: enhancement center key (e.g. ``'m_glass'``), defaults to ``'c_base'``
        - ``d``: edition key (e.g. ``'foil'``/``'holo'``/``'polychrome'``), or None
        - ``g``: seal (e.g. ``'Gold'``/``'Red'``), or None

    Args:
        control: The control dict.
        playing_card_index: Index in the playing_cards list.
        hands_played: Current hands_played for post-init fields.
    """
    suit = SUIT_LETTER[control["s"]]
    rank = RANK_LETTER[control["r"]]
    enhancement = control.get("e") or "c_base"
    edition_key = control.get("d")
    seal = control.get("g")

    edition = {edition_key: True} if edition_key else None

    return create_playing_card(
        suit=suit,
        rank=rank,
        enhancement=enhancement,
        edition=edition,
        seal=seal,
        playing_card_index=playing_card_index,
        hands_played=hands_played,
    )


# ---------------------------------------------------------------------------
# create_card — common_events.lua:2082
# ---------------------------------------------------------------------------

# Eternal/Perishable roll threshold (30% eternal, 30% perishable, 40% neither)
_EP_ETERNAL_THRESHOLD = 0.7
_EP_PERISHABLE_THRESHOLD = 0.4
_RENTAL_THRESHOLD = 0.7

# Area-specific RNG stream key prefixes
_EP_KEY: dict[str, str] = {"shop": "etperpoll", "pack": "packetper"}
_RENTAL_KEY: dict[str, str] = {"shop": "ssjr", "pack": "packssjr"}


def create_card(
    card_type: str,
    rng: PseudoRandom,
    ante: int,
    *,
    area: str = "shop",
    soulable: bool = True,
    forced_key: str | None = None,
    forced_rarity: int | None = None,
    append: str = "",
    game_state: dict | None = None,
) -> Card:
    """Create a card using the full Balatro creation pipeline.

    Mirrors ``create_card`` at ``common_events.lua:2082``.  The pipeline is:

    1. **Key determination** — ``forced_key`` → soul/Black-Hole chance →
       pool pick.
    2. **Card construction** — :meth:`Card.set_ability` from the resolved key.
    3. **Joker modifiers** (only when ``card.ability["set"] == "Joker"`` and
       *area* is ``"shop"`` or ``"pack"``):

       a. Eternal / Perishable roll (shared; mutually exclusive).
       b. Rental roll (independent).
       c. Edition roll via :func:`~jackdaw.engine.card_utils.poll_edition`.

    4. **Cost** — :meth:`Card.set_cost` with values extracted from
       *game_state*.

    All three modifier RNG streams are always advanced for Jokers in
    shop/pack context (even when the corresponding stake option is disabled)
    so that the stream positions remain deterministic regardless of stake.

    Parameters
    ----------
    card_type:
        ``"Joker"``, ``"Tarot"``, ``"Planet"``, ``"Spectral"``, or
        ``"PlayingCard"``.  Controls pool type and soul-chance pool type.
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number (used in all RNG stream keys).
    area:
        Context for modifier seed keys.  ``"shop"`` uses ``etperpoll``/
        ``ssjr``; ``"pack"`` uses ``packetper``/``packssjr``.
    soulable:
        When ``True``, a 0.3% chance exists for the card to be forced to
        Soul (Joker) or Black Hole (Planet/Spectral).  Pass ``False`` for
        playing-card draws that cannot become Soul.
    forced_key:
        If provided, skip soul-chance and pool selection entirely and create
        this center key directly.
    forced_rarity:
        Force a specific joker rarity (1–4) for pool selection.  Ignored for
        non-Joker types.
    append:
        Seed-key suffix forwarded to pool selection and the edition roll
        (``"edi" + append + str(ante)``).
    game_state:
        Dict of game-state values.  All keys are optional and default
        conservatively (no stickers, rate=1, no inflation, etc.).

        Pool-filtering keys (forwarded to :func:`~jackdaw.engine.pools.get_current_pool`):
        ``used_jokers``, ``used_vouchers``, ``banned_keys``, ``pool_flags``,
        ``has_showman``, ``deck_enhancements``, ``playing_card_count``,
        ``played_hand_types``, ``shop_vouchers``.

        Modifier-enable keys:
        ``enable_eternals_in_shop`` (bool), ``enable_perishables_in_shop``
        (bool), ``enable_rentals_in_shop`` (bool).

        Edition key:
        ``edition_rate`` (float, default 1.0).

        Cost keys forwarded to :meth:`Card.set_cost`:
        ``inflation`` (int), ``discount_percent`` (int),
        ``booster_ante_scaling`` (bool), ``has_astronomer`` (bool).

    Returns
    -------
    Card
        Fully initialised card with ability, optional stickers, optional
        edition, and cost set.
    """
    from jackdaw.engine.card_utils import poll_edition
    from jackdaw.engine.pools import check_soul_chance, pick_card_from_pool

    gs = game_state or {}

    # ------------------------------------------------------------------
    # 1. Determine center key
    # ------------------------------------------------------------------
    key = forced_key

    # Lua: if _type == 'Base' then forced_key = 'c_base' end
    # PlayingCard / Base type bypasses pool selection entirely.
    if key is None and card_type in ("Base", "PlayingCard"):
        key = "c_base"

    if key is None and soulable:
        key = check_soul_chance(card_type, rng, ante)

    if key is None:
        key = pick_card_from_pool(
            card_type,
            rng,
            ante,
            append=append,
            rarity=forced_rarity,
            legendary=(card_type == "Joker" and forced_rarity == 4),
            used_jokers=gs.get("used_jokers"),
            used_vouchers=gs.get("used_vouchers"),
            banned_keys=gs.get("banned_keys"),
            pool_flags=gs.get("pool_flags"),
            has_showman=gs.get("has_showman", False),
            deck_enhancements=gs.get("deck_enhancements"),
            playing_card_count=gs.get("playing_card_count", 52),
            played_hand_types=gs.get("played_hand_types"),
            shop_vouchers=gs.get("shop_vouchers"),
        )

    # ------------------------------------------------------------------
    # 2. Construct the card
    # ------------------------------------------------------------------
    card = Card()
    card.set_ability(key)

    # Card:set_ability registers EVERY created center key in
    # G.GAME.used_jokers (card.lua:349-354) — shop displays, pack contents,
    # and consumable-created cards alike.  This drives run-wide duplicate
    # exclusion in all pools (a key never repeats without Showman).
    if game_state is not None:
        game_state.setdefault("used_jokers", {})[key] = True

    # ------------------------------------------------------------------
    # 3. Joker modifiers (shop / pack context only)
    # ------------------------------------------------------------------
    if card.ability.get("set") == "Joker" and area in ("shop", "pack"):
        enable_eternals = gs.get("enable_eternals_in_shop", False)
        enable_perishables = gs.get("enable_perishables_in_shop", False)
        enable_rentals = gs.get("enable_rentals_in_shop", False)

        # -- Eternal / Perishable (shared roll) --
        ep_roll = rng.random(_EP_KEY[area] + str(ante))
        if ep_roll > _EP_ETERNAL_THRESHOLD and enable_eternals:
            card.set_eternal(True)
        elif ep_roll > _EP_PERISHABLE_THRESHOLD and enable_perishables:
            card.set_perishable(True)

        # -- Rental (independent roll) --
        r_roll = rng.random(_RENTAL_KEY[area] + str(ante))
        if r_roll > _RENTAL_THRESHOLD and enable_rentals:
            card.set_rental(True)

        # -- Edition --
        edition = poll_edition(
            "edi" + append + str(ante),
            rng,
            rate=gs.get("edition_rate", 1.0),
        )
        card.set_edition(edition)

    # ------------------------------------------------------------------
    # 4. Cost
    # ------------------------------------------------------------------
    card.set_cost(
        inflation=gs.get("inflation", 0),
        discount_percent=gs.get("discount_percent", 0),
        ante=ante,
        booster_ante_scaling=gs.get("booster_ante_scaling", False),
        has_astronomer=gs.get("has_astronomer", False),
    )

    return card


# ---------------------------------------------------------------------------
# Descriptor resolvers — M10 bridge between side-effects and card creation
# ---------------------------------------------------------------------------

# String or int rarity → forced_rarity int (1–4)
_RARITY_MAP: dict[str | int, int] = {
    "Common": 1,
    "Uncommon": 2,
    "Rare": 3,
    "Legendary": 4,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}


def resolve_create_descriptor(
    descriptor: dict[str, Any],
    rng: PseudoRandom,
    ante: int,
    game_state: dict[str, Any],
) -> Card | None:
    """Resolve a create-descriptor dict into an actual Card.

    This is the bridge between the side-effect descriptors returned by joker
    handlers (:mod:`jackdaw.engine.jokers`) and consumable handlers
    (:mod:`jackdaw.engine.consumables`) and the actual card-creation
    machinery.

    Descriptor shapes
    -----------------
    **Consumable / playing-card types** (no ``type`` key, or
    ``type='PlayingCard'``):

    .. code-block:: python

        {'rank': 'King', 'suit': 'Spades', 'enhancement': 'm_lucky'}
        {'type': 'PlayingCard', 'rank': 'Ace', 'suit': 'Hearts'}

    Produces a playing card via :func:`create_playing_card`.

    **Copy** (Ankh):

    .. code-block:: python

        {'type': 'Joker', 'copy_of': <Card>}

    Returns a deep copy of the referenced card.

    **Pool-drawn cards** (all other types):

    .. code-block:: python

        {'type': 'Tarot'}                           # random Tarot
        {'type': 'Tarot', 'key': 'car'}             # random Tarot, append='car'
        {'type': 'Planet', 'seed': 'pri'}           # random Planet, append='pri'
        {'type': 'Tarot_Planet', 'forced_key': 'c_fool', 'seed': 'fool'}
        {'type': 'Joker', 'rarity': 3, 'seed': 'wra'}   # Rare Joker
        {'type': 'Joker', 'rarity': 4, 'seed': 'soul'}  # Legendary Joker
        {'type': 'Joker', 'rarity': 'Common', 'key': 'rif'}
        {'type': 'Spectral', 'key': 'sea'}

    ``'key'`` (joker descriptors) and ``'seed'`` (consumable descriptors)
    are both accepted as the RNG-append value.  ``'forced_key'`` provides an
    explicit center key, bypassing pool selection entirely.

    Area is left blank (``""``), so Joker sticker rolls (eternal/perishable/
    rental) and edition rolls are **not** applied — consumable/joker-triggered
    creation never produces stickered cards.

    Parameters
    ----------
    descriptor:
        Side-effect descriptor dict.
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    game_state:
        Forwarded to :func:`create_card` / :func:`create_playing_card`.

    Returns
    -------
    Card | None
        The created card, or ``None`` if the descriptor is unrecognised.
    """
    card_type: str | None = descriptor.get("type")

    # ------------------------------------------------------------------
    # Playing card (explicit or inferred from rank/suit keys)
    # ------------------------------------------------------------------
    if card_type == "PlayingCard" or (card_type is None and "rank" in descriptor):
        rank_str: str | None = descriptor.get("rank")
        suit_str: str | None = descriptor.get("suit")
        if not rank_str or not suit_str:
            return None
        enhancement = descriptor.get("enhancement", "c_base")
        return create_playing_card(
            Suit(suit_str),
            Rank(rank_str),
            enhancement=enhancement,
        )

    # ------------------------------------------------------------------
    # Copy of an existing card (Ankh)
    # ------------------------------------------------------------------
    copy_source: Card | None = descriptor.get("copy_of")
    if copy_source is not None:
        return copy.deepcopy(copy_source)

    # ------------------------------------------------------------------
    # Pool-drawn consumables and jokers
    # ------------------------------------------------------------------
    if card_type not in ("Tarot", "Planet", "Spectral", "Tarot_Planet", "Joker"):
        return None

    forced_key: str | None = descriptor.get("forced_key")
    # Both 'key' (joker descriptors) and 'seed' (consumable descriptors)
    # serve as the RNG-stream append.
    append: str = descriptor.get("key") or descriptor.get("seed") or ""

    raw_rarity = descriptor.get("rarity")
    forced_rarity: int | None = _RARITY_MAP.get(raw_rarity) if raw_rarity is not None else None

    return create_card(
        card_type,
        rng,
        ante,
        area="",  # no shop/pack stickers for consumable-triggered creation
        # Vanilla passes soulable=nil for every consumable/joker-triggered
        # create_card call (only pack opening passes soulable=true), so
        # descriptor-created cards never roll the Soul/Black Hole chance.
        soulable=False,
        forced_key=forced_key,
        forced_rarity=forced_rarity,
        append=append,
        game_state=game_state,
    )


def resolve_destroy_descriptor(
    descriptor: dict[str, Any],
    jokers: list[Card],
    rng: PseudoRandom,
) -> Card | None:
    """Resolve a destruction descriptor, returning the card to destroy (if any).

    Descriptor shapes
    -----------------
    ``{'destroy_random_joker': True}``
        Select a random non-eternal joker from *jokers* via RNG stream
        ``'madness'`` (matching ``pseudoseed('madness')`` in
        ``card.lua:2509``).  Returns the selected card, or ``None`` if there
        are no eligible (non-eternal) jokers.

    ``{'disable_blind': True}``
        No card is destroyed.  Returns ``None``; the caller is responsible
        for calling ``blind.disable()``.

    Parameters
    ----------
    descriptor:
        Destruction descriptor dict.
    jokers:
        The current list of joker cards.
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.

    Returns
    -------
    Card | None
        The card that should be destroyed, or ``None``.
    """
    if descriptor.get("destroy_random_joker"):
        eligible = [j for j in jokers if not j.eternal]
        if not eligible:
            return None
        selected, _ = rng.element(eligible, rng.seed("madness"))
        return selected

    # disable_blind and any other unknown descriptors: no card to destroy
    return None
