"""Consumable use dispatch system.

Ports ``Card:use_consumeable`` from ``card.lua:1091`` and
``Card:can_use_consumeable`` from ``card.lua:1523`` as a dispatch table
keyed by center key.

Each consumable handler is a pure function
``(Card, ConsumableContext) -> ConsumableResult`` registered via the
``@register_consumable`` decorator.  The ``use_consumable`` entry point
validates, dispatches, and returns a side-effect descriptor.

Source: card.lua:1091-1522 (use_consumeable), card.lua:1523-1579
(can_use_consumeable).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jackdaw.engine.card_utils import poll_edition

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom


# ---------------------------------------------------------------------------
# ConsumableContext — data available when using a consumable
# ---------------------------------------------------------------------------


@dataclass
class ConsumableContext:
    """Context for consumable use.  Mirrors the data available to
    ``Card:use_consumeable`` in the source.
    """

    card: Card | None = None
    """The consumable being used."""

    highlighted: list[Card] | None = None
    """Cards highlighted (selected) in hand."""

    hand_cards: list[Card] | None = None
    """All cards currently in hand."""

    jokers: list[Card] | None = None
    """Active joker cards."""

    consumables: list[Card] | None = None
    """Current consumable slots."""

    playing_cards: list[Card] | None = None
    """Full deck master list."""

    rng: PseudoRandom | None = None

    game_state: dict[str, Any] | None = None
    """money, consumable_usage, last_tarot_planet, etc."""


# ---------------------------------------------------------------------------
# ConsumableResult — side-effect descriptor
# ---------------------------------------------------------------------------


@dataclass
class ConsumableResult:
    """Side-effect descriptor returned by consumable handlers.

    The caller (state machine) interprets and applies these mutations.
    """

    # Card modifications (tarots targeting highlighted cards)
    enhance: list[tuple[Card, str]] | None = None
    """[(card, enhancement_key)] — set_ability on each card."""

    change_suit: list[tuple[Card, str]] | None = None
    """[(card, suit)] — change card's base suit."""

    change_rank: list[tuple[Card, int]] | None = None
    """[(card, rank_delta)] — shift rank by delta (Strength=+1)."""

    copy_card: tuple[Card, Card] | None = None
    """(source, target) — copy source onto target (Death)."""

    destroy: list[Card] | None = None
    """Cards to destroy (Hanged Man, etc.)."""

    add_seal: list[tuple[Card, str]] | None = None
    """[(card, seal_type)] — set_seal on each card."""

    # Card creation
    create: list[dict[str, Any]] | None = None
    """[{'type': 'Tarot', 'count': 2, ...}] — cards to create."""

    # Economy
    dollars: int = 0

    # Hand level
    level_up: list[tuple[str, int]] | None = None
    """[(hand_type, amount)] — level up hand types (Planets)."""

    # Deck mutation
    add_to_deck: list[dict[str, Any]] | None = None
    """Playing cards to add to deck."""

    # Joker effects
    add_edition: dict[str, Any] | None = None
    """{'target': card, 'edition': {...}} — Wheel of Fortune, Aura."""

    destroy_jokers: list[Card] | None = None
    """Jokers to destroy (Ankh side-effect: destroy all others)."""

    # Game state
    hand_size_mod: int = 0
    """Ectoplasm (-1), Ouija (-1) reduce hand size."""

    money_set: int | None = None
    """Wraith sets money to 0."""

    notify_jokers_consumeable: bool = False
    """When True, the state machine must call calculate_joker with
    {using_consumeable=True, consumeable=card} on each joker after applying
    this result.  Set for all Planet uses (Constellation, Glass Joker, etc.)."""


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

ConsumableHandler = Callable[["Card", ConsumableContext], ConsumableResult | None]
_CONSUMABLE_REGISTRY: dict[str, ConsumableHandler] = {}


def register_consumable(key: str) -> Callable[[ConsumableHandler], ConsumableHandler]:
    """Decorator to register a consumable handler by center key."""

    def wrapper(fn: ConsumableHandler) -> ConsumableHandler:
        _CONSUMABLE_REGISTRY[key] = fn
        return fn

    return wrapper


def registered_consumables() -> list[str]:
    """Return sorted list of all registered consumable center keys."""
    return sorted(_CONSUMABLE_REGISTRY)


# ---------------------------------------------------------------------------
# Validation — can_use_consumable
# ---------------------------------------------------------------------------


def _resolve_consumable_config(card: Card) -> dict[str, Any]:
    """Get the consumable config from the card's ability."""
    cfg = card.ability.get("consumeable", {})
    if isinstance(cfg, list) and len(cfg) == 0:
        return {}
    return cfg if isinstance(cfg, dict) else {}


# Consumables that need no highlighted cards and no slot checks
_ALWAYS_USABLE = frozenset(
    {
        "c_hermit",
        "c_temperance",
        "c_black_hole",
    }
)

# Consumables that need a consumable slot
_NEED_CONSUMABLE_SLOT = frozenset(
    {
        "c_fool",
        "c_emperor",
        "c_high_priestess",
    }
)

# Consumables that need a joker slot
_NEED_JOKER_SLOT = frozenset(
    {
        "c_judgement",
        "c_soul",
        "c_wraith",
    }
)

# Consumables that need an eligible joker (editionless)
_NEED_ELIGIBLE_JOKER = frozenset(
    {
        "c_wheel_of_fortune",
        "c_ectoplasm",
        "c_hex",
    }
)

# Consumables that need >1 card in hand (no highlight selection)
_NEED_HAND_CARDS = frozenset(
    {
        "c_familiar",
        "c_grim",
        "c_incantation",
        "c_immolate",
        "c_sigil",
        "c_ouija",
    }
)

# Planet cards — always usable (they have hand_type in config)
_PLANET_KEYS = frozenset(
    {
        "c_mercury",
        "c_venus",
        "c_earth",
        "c_mars",
        "c_jupiter",
        "c_saturn",
        "c_uranus",
        "c_neptune",
        "c_pluto",
        "c_planet_x",
        "c_ceres",
        "c_eris",
    }
)


def can_use_consumable(
    card: Card,
    *,
    highlighted: list[Card] | None = None,
    hand_cards: list[Card] | None = None,
    jokers: list[Card] | None = None,
    consumables: list[Card] | None = None,
    consumable_limit: int = 2,
    joker_limit: int = 5,
    cards_in_play: int = 0,
    game_state: dict[str, Any] | None = None,
) -> bool:
    """Check if a consumable can be used.

    Mirrors ``Card:can_use_consumeable`` (card.lua:1523-1579).
    """
    # Global blockers
    if cards_in_play > 0:
        return False

    key = card.center_key
    highlighted = highlighted or []
    hand_cards = hand_cards or []
    jokers = jokers or []
    consumables = consumables or []
    gs = game_state or {}

    # Planets: always usable
    if key in _PLANET_KEYS:
        return True

    # Always usable (no selection needed)
    if key in _ALWAYS_USABLE:
        return True

    # Planet-like: cards with hand_type in consumeable config
    cfg = _resolve_consumable_config(card)
    if cfg.get("hand_type"):
        return True

    # Need consumable slot
    if key in _NEED_CONSUMABLE_SLOT:
        # The card itself frees a slot when used from consumable area
        effective = len(consumables) - (1 if card in consumables else 0)
        if effective < consumable_limit:
            if key == "c_fool":
                ltp = gs.get("last_tarot_planet")
                return bool(ltp) and ltp != "c_fool"
            return True
        return False

    # Need joker slot
    if key in _NEED_JOKER_SLOT:
        effective_jokers = len(jokers)
        if effective_jokers < joker_limit:
            return True
        return False

    # Need eligible joker (editionless)
    if key in _NEED_ELIGIBLE_JOKER:
        return any(not j.edition and not j.debuff for j in jokers)

    # Ankh: need joker + room to duplicate
    if key == "c_ankh":
        return len(jokers) > 0 and joker_limit > 1

    # Aura: need 1 highlighted card with no edition
    if key == "c_aura":
        return len(highlighted) == 1 and not highlighted[0].edition

    # Need >1 card in hand (destroy-based spectrals)
    if key in _NEED_HAND_CARDS:
        return len(hand_cards) > 1

    # Highlighted card selection (tarots with max_highlighted)
    if cfg.get("max_highlighted"):
        max_h = cfg["max_highlighted"]
        min_h = cfg.get("min_highlighted", 1)
        mod_num = cfg.get("mod_num", max_h)
        return min_h <= len(highlighted) <= mod_num

    return False


# ---------------------------------------------------------------------------
# Use dispatch
# ---------------------------------------------------------------------------


def use_consumable(
    card: Card,
    context: ConsumableContext,
) -> ConsumableResult | None:
    """Use a consumable. Returns side-effect descriptor or None.

    Mirrors ``Card:use_consumeable`` (card.lua:1091).
    """
    if card.debuff:
        return None
    handler = _CONSUMABLE_REGISTRY.get(card.center_key)
    if handler is None:
        return None
    return handler(card, context)


# ---------------------------------------------------------------------------
# Enhancement tarots — change card enhancement via Card.enhance()
# Source: card.lua:1091-1150 (mod_conv path)
# ---------------------------------------------------------------------------


def _enhance_handler(enhancement: str) -> ConsumableHandler:
    """Factory for enhancement tarot handlers."""

    def handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(
            enhance=[(c, enhancement) for c in (ctx.highlighted or [])],
        )

    return handler


register_consumable("c_magician")(_enhance_handler("m_lucky"))
register_consumable("c_empress")(_enhance_handler("m_mult"))
register_consumable("c_heirophant")(_enhance_handler("m_bonus"))
register_consumable("c_lovers")(_enhance_handler("m_wild"))
register_consumable("c_chariot")(_enhance_handler("m_steel"))
register_consumable("c_justice")(_enhance_handler("m_glass"))
register_consumable("c_devil")(_enhance_handler("m_gold"))
register_consumable("c_tower")(_enhance_handler("m_stone"))


# ---------------------------------------------------------------------------
# Suit-change tarots (1-3 highlighted → change suit)
# Source: card.lua:1137 (suit_conv path)
# ---------------------------------------------------------------------------


def _suit_handler(suit: str) -> ConsumableHandler:
    """Factory for suit-change tarot handlers."""

    def handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(
            change_suit=[(c, suit) for c in (ctx.highlighted or [])],
        )

    return handler


register_consumable("c_star")(_suit_handler("Diamonds"))
register_consumable("c_moon")(_suit_handler("Clubs"))
register_consumable("c_sun")(_suit_handler("Hearts"))
register_consumable("c_world")(_suit_handler("Spades"))


# ---------------------------------------------------------------------------
# Transformation tarots
# ---------------------------------------------------------------------------

# Rank progression: id-based. Ace(14)→2, else id+1 capped at 14.
_ID_TO_RANK = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}


def _next_rank(current_id: int) -> str:
    """Strength rank progression: Ace→2, else +1 (cap at Ace)."""
    if current_id == 14:
        return "2"
    return _ID_TO_RANK[min(current_id + 1, 14)]


@register_consumable("c_strength")
def _strength(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Strength: +1 rank to highlighted cards. Ace wraps to 2.

    Source: card.lua:1121. Uses id-based increment.
    """
    changes = []
    for c in ctx.highlighted or []:
        if c.base is not None:
            new_rank = _next_rank(c.base.id)
            changes.append((c, new_rank))
    return ConsumableResult(change_rank=changes)


@register_consumable("c_death")
def _death(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Death: copy the rightmost highlighted card onto the others.

    Source: card.lua:1111. Rightmost by sort_id (position proxy).
    The copy transfers: base (suit+rank), enhancement, edition, seal.
    """
    highlighted = ctx.highlighted or []
    if len(highlighted) < 2:
        return ConsumableResult()

    # Find rightmost card (highest sort_id = rightmost position)
    rightmost = max(highlighted, key=lambda c: c.sort_id)
    targets = [c for c in highlighted if c is not rightmost]

    return ConsumableResult(
        copy_card=(rightmost, targets[0]) if targets else None,
        # For >2 cards, extend: but Death is exactly 2 in vanilla
    )


@register_consumable("c_hanged_man")
def _hanged_man(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Hanged Man: destroy highlighted cards. Source: card.lua:1271."""
    return ConsumableResult(
        destroy=list(ctx.highlighted or []),
    )


# ---------------------------------------------------------------------------
# Generation tarots — create card descriptors
# Source: card.lua:1373-1430 (Fool, Emperor, High Priestess, Judgement)
# ---------------------------------------------------------------------------


@register_consumable("c_fool")
def _fool(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """The Fool: create a copy of the last Tarot/Planet used.

    Reads game_state['last_tarot_planet'] for the forced key.
    Source: card.lua:1373.
    """
    gs = ctx.game_state or {}
    forced_key = gs.get("last_tarot_planet")
    if not forced_key:
        return ConsumableResult()
    return ConsumableResult(
        create=[{"type": "Tarot_Planet", "forced_key": forced_key, "seed": "fool"}],
    )


@register_consumable("c_high_priestess")
def _high_priestess(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """The High Priestess: create 2 random Planet cards.

    Source: card.lua:1401 (planets=2, seed='pri').
    """
    count = card.ability.get("consumeable", {}).get("planets", 2)
    return ConsumableResult(
        create=[{"type": "Planet", "count": count, "seed": "pri"}],
    )


@register_consumable("c_emperor")
def _emperor(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """The Emperor: create 2 random Tarot cards.

    Source: card.lua:1401 (tarots=2, seed='emp').
    """
    count = card.ability.get("consumeable", {}).get("tarots", 2)
    return ConsumableResult(
        create=[{"type": "Tarot", "count": count, "seed": "emp"}],
    )


@register_consumable("c_judgement")
def _judgement(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Judgement: create 1 random Joker.

    Source: card.lua:1413 (seed='jud').
    """
    return ConsumableResult(
        create=[{"type": "Joker", "count": 1, "seed": "jud"}],
    )


# ---------------------------------------------------------------------------
# Wheel of Fortune — add random edition to a random editionless joker
# Source: card.lua:1470-1510 (wheel_of_fortune path)
# ---------------------------------------------------------------------------


@register_consumable("c_wheel_of_fortune")
def _wheel_of_fortune(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """The Wheel of Fortune: 1-in-4 chance to add a random edition to a joker.

    Source: card.lua:1470.  Uses 3 sequential RNG advances on stream
    'wheel_of_fortune': probability check, card selection, edition selection.
    Returns ConsumableResult with add_edition on success, empty on failure.
    """
    if ctx.rng is None:
        return ConsumableResult()

    gs = ctx.game_state or {}
    prob = gs.get("probabilities_normal", 1)
    extra = card.ability.get("extra", 4)

    # Step 1: probability check (card.lua:1474)
    roll = ctx.rng.random("wheel_of_fortune")
    if roll >= prob / extra:
        return ConsumableResult()  # failure — "Nope!"

    # Step 2: pick a random editionless joker (card.lua:1477)
    editionless = [j for j in (ctx.jokers or []) if not j.edition and not j.debuff]
    if not editionless:
        return ConsumableResult()
    seed_val = ctx.rng.seed("wheel_of_fortune")
    target, _ = ctx.rng.element(editionless, seed_val)

    # Step 3: poll edition (card.lua:1484, guaranteed=True, no_neg=True)
    edition = poll_edition("wheel_of_fortune", ctx.rng, no_neg=True, guaranteed=True)

    return ConsumableResult(add_edition={"target": target, "edition": edition})


# ---------------------------------------------------------------------------
# Economy tarots
# Source: card.lua:1383-1399 (Hermit, Temperance)
# ---------------------------------------------------------------------------


@register_consumable("c_hermit")
def _hermit(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """The Hermit: gain min(current_dollars, $20).

    Source: card.lua:1383 — ease_dollars(min(G.GAME.dollars, ability.extra)).
    """
    gs = ctx.game_state or {}
    current = gs.get("dollars", 0)
    cap = card.ability.get("extra", 20)
    gain = max(0, min(current, cap))
    return ConsumableResult(dollars=gain)


@register_consumable("c_temperance")
def _temperance(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Temperance: gain $ = total sell value of all jokers (cap $50).

    Source: card.lua:1393 — sums sell_cost of all jokers, capped by ability.extra.
    """
    cap = card.ability.get("extra", 50)
    total = sum(j.sell_cost for j in (ctx.jokers or []))
    gain = min(total, cap)
    return ConsumableResult(dollars=gain)


# ---------------------------------------------------------------------------
# Planet cards — level up a specific hand type
# Source: card.lua:1157-1170 (hand_type path), common_events.lua:464 (level_up_hand)
#
# Usage tracking (set_consumeable_usage equivalent):
#   game_state['consumable_usage_total']['planet'] += 1
#   game_state['consumable_usage_total']['all']    += 1
#   game_state['last_tarot_planet'] = card.center_key
# ---------------------------------------------------------------------------

# Planet key → hand type string (from centers.json config.hand_type)
_PLANET_HAND: dict[str, str] = {
    "c_pluto": "High Card",
    "c_mercury": "Pair",
    "c_uranus": "Two Pair",
    "c_venus": "Three of a Kind",
    "c_saturn": "Straight",
    "c_jupiter": "Flush",
    "c_earth": "Full House",
    "c_mars": "Four of a Kind",
    "c_neptune": "Straight Flush",
    "c_planet_x": "Five of a Kind",
    "c_ceres": "Flush House",
    "c_eris": "Flush Five",
}

_ALL_HAND_TYPES: list[str] = list(_PLANET_HAND.values())


def _track_planet_usage(card: Card, ctx: ConsumableContext) -> None:
    """Mutate ctx.game_state to track planet usage (mirrors set_consumeable_usage).

    Updates consumable_usage_total.planet, .all and last_tarot_planet.
    """
    gs = ctx.game_state
    if gs is None:
        return
    totals = gs.setdefault(
        "consumable_usage_total",
        {
            "tarot": 0,
            "planet": 0,
            "spectral": 0,
            "tarot_planet": 0,
            "all": 0,
        },
    )
    totals["planet"] = totals.get("planet", 0) + 1
    totals["tarot_planet"] = totals.get("tarot_planet", 0) + 1
    totals["all"] = totals.get("all", 0) + 1
    gs["last_tarot_planet"] = card.center_key


def _make_planet_handler(hand_type: str) -> ConsumableHandler:
    """Factory for single-hand-type planet handlers."""

    def handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        _track_planet_usage(card, ctx)
        return ConsumableResult(
            level_up=[(hand_type, 1)],
            notify_jokers_consumeable=True,
        )

    return handler


for _key, _ht in _PLANET_HAND.items():
    register_consumable(_key)(_make_planet_handler(_ht))


@register_consumable("c_black_hole")
def _black_hole(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Black Hole: level up ALL 12 hand types by 1.

    Source: card.lua:1175 — iterates G.GAME.hands and calls level_up_hand.
    """
    gs = ctx.game_state
    if gs is not None:
        totals = gs.setdefault(
            "consumable_usage_total",
            {
                "tarot": 0,
                "planet": 0,
                "spectral": 0,
                "tarot_planet": 0,
                "all": 0,
            },
        )
        totals["planet"] = totals.get("planet", 0) + 1
        totals["tarot_planet"] = totals.get("tarot_planet", 0) + 1
        totals["all"] = totals.get("all", 0) + 1
        gs["last_tarot_planet"] = card.center_key
    return ConsumableResult(
        level_up=[(ht, 1) for ht in _ALL_HAND_TYPES],
        notify_jokers_consumeable=True,
    )


# ---------------------------------------------------------------------------
# Seal spectrals — add a seal to the highlighted card
# Source: card.lua:1178-1192.  conv_card:set_seal(self.ability.extra, …)
# The seal name is stored in ability.extra (from config.extra).
# ---------------------------------------------------------------------------


def _seal_handler(seal: str) -> ConsumableHandler:
    """Factory for seal-adding spectral handlers."""

    def handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        highlighted = ctx.highlighted or []
        return ConsumableResult(
            add_seal=[(c, seal) for c in highlighted],
        )

    return handler


register_consumable("c_talisman")(_seal_handler("Gold"))
register_consumable("c_deja_vu")(_seal_handler("Red"))
register_consumable("c_trance")(_seal_handler("Blue"))
register_consumable("c_medium")(_seal_handler("Purple"))


# ---------------------------------------------------------------------------
# Cryptid — create N exact copies of the highlighted card in the deck
# Source: card.lua:1201-1218.  copy_card(G.hand.highlighted[1], …) × extra
# The copy_of reference lets the state machine clone the source card fully.
# ---------------------------------------------------------------------------


@register_consumable("c_cryptid")
def _cryptid(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Cryptid: create 2 copies of the highlighted card and add to deck.

    Source: card.lua:1201.  ability.extra (=2) controls copy count.
    Returns add_to_deck descriptors each carrying a ``copy_of`` reference
    to the source card so the state machine can clone suit/rank/enhancement/
    edition/seal exactly.
    """
    highlighted = ctx.highlighted or []
    if not highlighted:
        return ConsumableResult()
    source = highlighted[0]
    count = card.ability.get("extra", 2)
    return ConsumableResult(
        add_to_deck=[{"copy_of": source} for _ in range(count)],
    )


# ---------------------------------------------------------------------------
# Shared data for destroy/create spectrals
# ---------------------------------------------------------------------------

# Lua short-code → Python full name
_SUIT_CODE: dict[str, str] = {
    "S": "Spades",
    "H": "Hearts",
    "D": "Diamonds",
    "C": "Clubs",
}
_RANK_CODE: dict[str, str] = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "Jack",
    "Q": "Queen",
    "K": "King",
    "A": "Ace",
}

# Enhanced pool (non-Stone), sorted by order field — built once on first use
_ENHANCED_POOL_CACHE: list[str] | None = None


def _get_enhanced_pool() -> list[str]:
    global _ENHANCED_POOL_CACHE  # noqa: PLW0603
    if _ENHANCED_POOL_CACHE is None:
        from jackdaw.engine.data.prototypes import _load_json

        centers = _load_json("centers.json")
        _ENHANCED_POOL_CACHE = sorted(
            [k for k, v in centers.items() if v.get("set") == "Enhanced" and k != "m_stone"],
            key=lambda k: centers[k].get("order", 0),
        )
    return _ENHANCED_POOL_CACHE


def _roll_card_spec(
    rank_pool: list[str],
    rank_seed: str,
    suit_seed: str,
    rng: PseudoRandom,
) -> dict:
    """Roll one playing-card creation descriptor.

    Picks rank from *rank_pool*, suit from all four, and a random
    non-Stone enhancement — all using the provided RNG seeds.

    Returns ``{'rank': str, 'suit': str, 'enhancement': str}``.
    """
    rank_code, _ = rng.element(rank_pool, rng.seed(rank_seed))
    suit_code, _ = rng.element(list(_SUIT_CODE), rng.seed(suit_seed))
    enhancement, _ = rng.element(_get_enhanced_pool(), rng.seed("spe_card"))
    return {
        "rank": _RANK_CODE[rank_code],
        "suit": _SUIT_CODE[suit_code],
        "enhancement": enhancement,
    }


# ---------------------------------------------------------------------------
# Familiar / Grim / Incantation — destroy 1, create N
# Source: card.lua:1292-1338
# ---------------------------------------------------------------------------


@register_consumable("c_familiar")
def _familiar(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Familiar: destroy 1 random hand card, create 3 enhanced face cards.

    Source: card.lua:1319.  Seeds: 'random_destroy', 'familiar_create', 'spe_card'.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    hand = list(ctx.hand_cards)
    destroyed, _ = ctx.rng.element(hand, ctx.rng.seed("random_destroy"))
    count = card.ability.get("extra", 3)
    created = [
        _roll_card_spec(["J", "Q", "K"], "familiar_create", "familiar_create", ctx.rng)
        for _ in range(count)
    ]
    return ConsumableResult(destroy=[destroyed], create=created)


@register_consumable("c_grim")
def _grim(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Grim: destroy 1 random hand card, create 2 enhanced Aces.

    Source: card.lua:1322.  Seeds: 'random_destroy', 'grim_create', 'spe_card'.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    hand = list(ctx.hand_cards)
    destroyed, _ = ctx.rng.element(hand, ctx.rng.seed("random_destroy"))
    count = card.ability.get("extra", 2)
    created = [_roll_card_spec(["A"], "grim_create", "grim_create", ctx.rng) for _ in range(count)]
    return ConsumableResult(destroy=[destroyed], create=created)


@register_consumable("c_incantation")
def _incantation(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Incantation: destroy 1 random hand card, create 4 enhanced number cards.

    Source: card.lua:1325.  Seeds: 'random_destroy', 'incantation_create', 'spe_card'.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    hand = list(ctx.hand_cards)
    destroyed, _ = ctx.rng.element(hand, ctx.rng.seed("random_destroy"))
    count = card.ability.get("extra", 4)
    number_ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T"]
    created = [
        _roll_card_spec(number_ranks, "incantation_create", "incantation_create", ctx.rng)
        for _ in range(count)
    ]
    return ConsumableResult(destroy=[destroyed], create=created)


# ---------------------------------------------------------------------------
# Immolate — destroy 5, gain $20
# Source: card.lua:1340-1365.  Pseudoshuffle + take first extra.destroy cards.
# ---------------------------------------------------------------------------


@register_consumable("c_immolate")
def _immolate(card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Immolate: destroy 5 random hand cards, gain $20.

    Source: card.lua:1340.  Pseudoshuffles hand (seed 'immolate'), destroys first 5.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    extra = card.ability.get("extra", {"destroy": 5, "dollars": 20})
    destroy_count = extra.get("destroy", 5) if isinstance(extra, dict) else 5
    dollars = extra.get("dollars", 20) if isinstance(extra, dict) else 20

    temp = list(ctx.hand_cards)
    ctx.rng.shuffle(temp, ctx.rng.seed("immolate"))
    destroyed = temp[:destroy_count]
    return ConsumableResult(destroy=destroyed, dollars=dollars)


# ---------------------------------------------------------------------------
# Sigil — change all hand cards to one random suit
# Source: card.lua:1232-1244.  pseudorandom_element({'S','H','D','C'}, pseudoseed('sigil'))
# ---------------------------------------------------------------------------


@register_consumable("c_sigil")
def _sigil(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Sigil: change all hand cards to a single random suit.

    Source: card.lua:1232.  Seed: 'sigil'.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    suit_code, _ = ctx.rng.element(list(_SUIT_CODE), ctx.rng.seed("sigil"))
    suit = _SUIT_CODE[suit_code]
    return ConsumableResult(
        change_suit=[(c, suit) for c in ctx.hand_cards],
    )


# ---------------------------------------------------------------------------
# Ouija — change all hand cards to one random rank, -1 hand size
# Source: card.lua:1246-1260.  pseudorandom_element({'2'…'A'}, pseudoseed('ouija'))
# ---------------------------------------------------------------------------


@register_consumable("c_ouija")
def _ouija(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Ouija: change all hand cards to a single random rank, reduce hand size by 1.

    Source: card.lua:1246.  Seed: 'ouija'.
    """
    if ctx.rng is None or not ctx.hand_cards:
        return ConsumableResult()
    rank_pool = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    rank_code, _ = ctx.rng.element(rank_pool, ctx.rng.seed("ouija"))
    rank = _RANK_CODE[rank_code]
    return ConsumableResult(
        change_rank=[(c, rank) for c in ctx.hand_cards],
        hand_size_mod=-1,
    )


# ---------------------------------------------------------------------------
# Edition spectrals
# ---------------------------------------------------------------------------


@register_consumable("c_aura")
def _aura(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Aura: add a random Foil/Holo/Poly edition to 1 highlighted card (no Negative).

    Source: card.lua:1262.  poll_edition guaranteed mode, no_neg=True.  Seed: 'aura'.
    """
    if ctx.rng is None or not ctx.highlighted:
        return ConsumableResult()
    target = ctx.highlighted[0]
    edition = poll_edition("aura", ctx.rng, no_neg=True, guaranteed=True)
    return ConsumableResult(add_edition={"target": target, "edition": edition})


@register_consumable("c_ectoplasm")
def _ectoplasm(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Ectoplasm: add Negative edition to a random editionless joker, -1 hand size.

    Source: card.lua:1270.  Seed: 'ectoplasm'.
    hand_size_mod=-1 is cumulative (each use subtracts 1 permanently).
    """
    if ctx.rng is None:
        return ConsumableResult()
    editionless = [j for j in (ctx.jokers or []) if not j.edition and not j.debuff]
    if not editionless:
        return ConsumableResult()
    target, _ = ctx.rng.element(editionless, ctx.rng.seed("ectoplasm"))
    return ConsumableResult(
        add_edition={"target": target, "edition": {"negative": True}},
        hand_size_mod=-1,
    )


@register_consumable("c_hex")
def _hex(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Hex: add Polychrome to a random editionless joker, destroy all other non-eternal jokers.

    Source: card.lua:1278.  Seed: 'hex'.
    """
    if ctx.rng is None:
        return ConsumableResult()
    editionless = [j for j in (ctx.jokers or []) if not j.edition and not j.debuff]
    if not editionless:
        return ConsumableResult()
    target, _ = ctx.rng.element(editionless, ctx.rng.seed("hex"))
    others = [j for j in (ctx.jokers or []) if j is not target and not j.eternal]
    return ConsumableResult(
        add_edition={"target": target, "edition": {"polychrome": True}},
        destroy_jokers=others if others else None,
    )


# ---------------------------------------------------------------------------
# Joker-creating spectrals
# ---------------------------------------------------------------------------


@register_consumable("c_wraith")
def _wraith(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Wraith: create 1 Rare joker, set money to $0.

    Source: card.lua:1430.  Rarity 3 = Rare.  Seed: 'wra'.
    """
    return ConsumableResult(
        create=[{"type": "Joker", "count": 1, "seed": "wra", "rarity": 3}],
        money_set=0,
    )


@register_consumable("c_ankh")
def _ankh(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Ankh: pick a random joker, destroy all other non-eternal jokers, duplicate chosen.

    Source: card.lua:1437.  Seed: 'ankh_choice'.
    """
    if ctx.rng is None or not ctx.jokers:
        return ConsumableResult()
    chosen, _ = ctx.rng.element(list(ctx.jokers), ctx.rng.seed("ankh_choice"))
    others = [j for j in ctx.jokers if j is not chosen and not j.eternal]
    return ConsumableResult(
        destroy_jokers=others if others else None,
        create=[{"type": "Joker", "copy_of": chosen}],
    )


@register_consumable("c_soul")
def _soul(_card: Card, ctx: ConsumableContext) -> ConsumableResult:
    """Soul: create 1 Legendary joker.

    Source: card.lua:1455.  Rarity 4 = Legendary.  Seed: 'soul'.
    """
    return ConsumableResult(
        create=[{"type": "Joker", "count": 1, "seed": "soul", "rarity": 4}],
    )
