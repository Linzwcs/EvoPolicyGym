"""Joker evaluation dispatch system.

Ports ``Card:calculate_joker`` from ``card.lua:2291`` — a 1,770-line
if/elseif chain — as a dispatch table keyed by center key.

Each joker handler is a pure function ``(Card, JokerContext) -> JokerResult | None``
registered via the ``@register`` decorator.  The ``calculate_joker`` entry point
checks the debuff flag, looks up the handler, and dispatches.

Source: card.lua:2291-4060 (calculate_joker), common_events.lua:571-1065 (call sites).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.hand_levels import HandLevels
    from jackdaw.engine.rng import PseudoRandom


# ---------------------------------------------------------------------------
# GameSnapshot — immutable game state, built ONCE per score_hand call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSnapshot:
    """Pre-computed game state shared across all JokerContext instances.

    Built once at the start of ``score_hand`` and passed by reference to
    every JokerContext created during the scoring pipeline.  This avoids
    copying ~20 game-state fields into every context object.
    """

    joker_count: int = 0
    joker_slots: int = 5
    money: int = 0
    deck_cards_remaining: int = 0
    starting_deck_size: int = 52
    playing_cards_count: int = 52
    stone_tally: int = 0
    steel_tally: int = 0
    enhanced_card_count: int = 0
    hands_left: int = 0
    hands_played: int = 0
    discards_left: int = 0
    discards_used: int = 0
    probabilities_normal: float = 1.0
    consumable_usage_tarot: int = 0
    mail_card_id: int | None = None
    idol_card: dict[str, Any] | None = None
    ancient_suit: str | None = None
    castle_card_suit: str | None = None
    skips: int = 0


_DEFAULT_GAME = GameSnapshot()


# ---------------------------------------------------------------------------
# JokerContext — lightweight per-call context referencing shared GameSnapshot
# ---------------------------------------------------------------------------


@dataclass
class JokerContext:
    """Context table passed to each joker handler.

    Exactly one phase flag should be set per call, matching the source's
    context-based branching in card.lua:2291+.

    Game state lives on the shared :class:`GameSnapshot` referenced by
    ``self.game``.  For backward compatibility, game-state fields can be
    passed as keyword arguments — they are ``InitVar`` parameters that
    auto-build a ``GameSnapshot`` in ``__post_init__`` when ``game`` is
    not provided directly.
    """

    # Phase flags (exactly one set per call)
    before: bool = False
    individual: bool = False
    repetition: bool = False
    joker_main: bool = False
    after: bool = False
    other_joker: Card | None = None
    setting_blind: bool = False
    first_hand_drawn: bool = False
    end_of_round: bool = False
    discard: bool = False
    pre_discard: bool = False
    destroying_card: Card | None = None
    cards_destroyed: list[Card] | None = None
    buying_card: bool = False
    selling_self: bool = False
    selling_card: bool = False
    open_booster: bool = False
    skip_blind: bool = False
    skipping_booster: bool = False
    reroll_shop: bool = False
    ending_shop: bool = False
    debuffed_hand: bool = False
    using_consumeable: bool = False
    playing_card_added: bool = False
    individual_hand_end: bool = False

    # Per-call context data
    cardarea: str | None = None
    other_card: Card | None = None
    full_hand: list[Card] | None = None
    scoring_hand: list[Card] | None = None
    scoring_name: str | None = None
    poker_hands: dict[str, Any] | None = None
    jokers: list[Card] | None = None
    rng: PseudoRandom | None = None
    blind: Blind | None = None
    hand_levels: HandLevels | None = None
    held_cards: list[Card] | None = None
    consumeable: Card | None = None
    cards: list[Card] | None = None

    # Blueprint
    blueprint: int = 0
    blueprint_card: Card | None = None

    # Meta joker flags
    smeared: bool = False
    pareidolia: bool = False

    # Shared game snapshot (built once per score_hand call)
    game: GameSnapshot = field(default_factory=lambda: _DEFAULT_GAME)

    # --- Init-only fields for backward compatibility -------------------------
    # Accepted in __init__, used to build GameSnapshot when game is not
    # provided explicitly.  Not stored as instance attributes.
    joker_count: InitVar[int] = 0
    joker_slots: InitVar[int] = 5
    money: InitVar[int] = 0
    deck_cards_remaining: InitVar[int] = 0
    starting_deck_size: InitVar[int] = 52
    playing_cards_count: InitVar[int] = 52
    stone_tally: InitVar[int] = 0
    steel_tally: InitVar[int] = 0
    enhanced_card_count: InitVar[int] = 0
    hands_left: InitVar[int] = 0
    hands_played: InitVar[int] = 0
    discards_left: InitVar[int] = 0
    discards_used: InitVar[int] = 0
    probabilities_normal: InitVar[float] = 1.0
    consumable_usage_tarot: InitVar[int] = 0
    mail_card_id: InitVar[int | None] = None
    idol_card: InitVar[dict[str, Any] | None] = None
    ancient_suit: InitVar[str | None] = None
    skips: InitVar[int] = 0

    def __post_init__(
        self,
        joker_count: int,
        joker_slots: int,
        money: int,
        deck_cards_remaining: int,
        starting_deck_size: int,
        playing_cards_count: int,
        stone_tally: int,
        steel_tally: int,
        enhanced_card_count: int,
        hands_left: int,
        hands_played: int,
        discards_left: int,
        discards_used: int,
        probabilities_normal: float,
        consumable_usage_tarot: int,
        mail_card_id: int | None,
        idol_card: dict[str, Any] | None,
        ancient_suit: str | None,
        skips: int,
    ) -> None:
        """Build GameSnapshot from flat kwargs when game is the default."""
        if self.game is _DEFAULT_GAME:
            # Check if any non-default value was passed
            has_custom = (
                joker_count != 0
                or joker_slots != 5
                or money != 0
                or deck_cards_remaining != 0
                or starting_deck_size != 52
                or playing_cards_count != 52
                or stone_tally != 0
                or steel_tally != 0
                or enhanced_card_count != 0
                or hands_left != 0
                or hands_played != 0
                or discards_left != 0
                or discards_used != 0
                or probabilities_normal != 1.0
                or consumable_usage_tarot != 0
                or mail_card_id is not None
                or idol_card is not None
                or ancient_suit is not None
                or skips != 0
            )
            if has_custom:
                object.__setattr__(
                    self,
                    "game",
                    GameSnapshot(
                        joker_count=joker_count,
                        joker_slots=joker_slots,
                        money=money,
                        deck_cards_remaining=deck_cards_remaining,
                        starting_deck_size=starting_deck_size,
                        playing_cards_count=playing_cards_count,
                        stone_tally=stone_tally,
                        steel_tally=steel_tally,
                        enhanced_card_count=enhanced_card_count,
                        hands_left=hands_left,
                        hands_played=hands_played,
                        discards_left=discards_left,
                        discards_used=discards_used,
                        probabilities_normal=probabilities_normal,
                        consumable_usage_tarot=consumable_usage_tarot,
                        mail_card_id=mail_card_id,
                        idol_card=idol_card,
                        ancient_suit=ancient_suit,
                        skips=skips,
                    ),
                )


# ---------------------------------------------------------------------------
# JokerResult — return value from joker handlers
# ---------------------------------------------------------------------------


@dataclass
class JokerResult:
    """Return value from a joker handler.

    Fields match BOTH return formats used by the source:
    - ``individual`` context returns: chips, mult, x_mult, dollars
    - ``joker_main`` context returns: chip_mod, mult_mod, Xmult_mod
    - ``repetition`` context returns: repetitions
    """

    # Individual context returns
    chips: float = 0
    mult: float = 0
    x_mult: float = 0
    h_mult: float = 0  # held-card mult (Raised Fist, Shoot the Moon)
    dollars: int = 0

    # Joker main context returns (different naming!)
    chip_mod: float = 0
    mult_mod: float = 0
    Xmult_mod: float = 0  # noqa: N815 — matches Lua naming

    # Repetition context
    repetitions: int = 0

    # State change signals
    level_up: bool = False
    saved: bool = False
    remove: bool = False
    message: str = ""

    # Extra (for unusual effects like Perkeo, Showman, etc.)
    extra: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

JokerHandler = Callable[["Card", JokerContext], JokerResult | None]
_REGISTRY: dict[str, JokerHandler] = {}


def register(key: str) -> Callable[[JokerHandler], JokerHandler]:
    """Decorator to register a joker handler by center key.

    Usage::

        @register("j_joker")
        def _joker(card: Card, ctx: JokerContext) -> JokerResult | None:
            if ctx.joker_main:
                return JokerResult(mult_mod=card.ability["extra"])
            return None
    """

    def wrapper(fn: JokerHandler) -> JokerHandler:
        _REGISTRY[key] = fn
        return fn

    return wrapper


def calculate_joker(card: Card, context: JokerContext) -> JokerResult | None:
    """Main dispatch — mirrors Card:calculate_joker from card.lua:2291.

    Returns ``None`` if the joker is debuffed, unregistered, or has no
    effect in the given context.
    """
    if card.debuff:
        return None
    handler = _REGISTRY.get(card.center_key)
    if handler is None:
        return None
    return handler(card, context)


def registered_jokers() -> list[str]:
    """Return sorted list of all registered joker center keys."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Dollar bonus registry (calc_dollar_bonus — separate from calculate_joker)
# Source: card.lua:1658-1677 — per-round dollar payout
# ---------------------------------------------------------------------------

DollarHandler = Callable[["Card", "GameSnapshot"], int]
_DOLLAR_REGISTRY: dict[str, DollarHandler] = {}


def register_dollars(key: str) -> Callable[[DollarHandler], DollarHandler]:
    """Decorator to register a per-round dollar bonus handler."""

    def wrapper(fn: DollarHandler) -> DollarHandler:
        _DOLLAR_REGISTRY[key] = fn
        return fn

    return wrapper


def calc_dollar_bonus(card: Card, game: GameSnapshot) -> int:
    """Per-round dollar bonus. Mirrors card.lua:1658 (calc_dollar_bonus)."""
    if card.debuff:
        return 0
    handler = _DOLLAR_REGISTRY.get(card.center_key)
    if handler is None:
        return 0
    return handler(card, game)


def on_end_of_round(
    jokers: list[Card],
    game: GameSnapshot,
    rng: PseudoRandom | None = None,
) -> dict[str, Any]:
    """Process all joker end-of-round effects.

    Returns a dict with:
        dollars_earned: total dollars from calc_dollar_bonus
        jokers_removed: list of jokers that self-destructed
        mutations: list of side-effect descriptors
    """
    dollars = 0
    removed: list[Card] = []
    mutations: list[dict[str, Any]] = []

    # 1. Dollar bonuses (calc_dollar_bonus)
    for joker in jokers:
        dollars += calc_dollar_bonus(joker, game)

    # 2. End-of-round calculate_joker effects
    ctx = JokerContext(
        end_of_round=True,
        jokers=jokers,
        rng=rng,
        game=game,
    )
    for joker in jokers:
        if joker.debuff:
            continue
        result = calculate_joker(joker, ctx)
        if result:
            if result.remove:
                removed.append(joker)
            if result.extra:
                mutations.append(result.extra)

    return {
        "dollars_earned": dollars,
        "jokers_removed": removed,
        "mutations": mutations,
    }


# ---------------------------------------------------------------------------
# Joker handlers — simple unconditional bonuses
# ---------------------------------------------------------------------------


@register("j_joker")
def _joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Joker: +mult (default 4). Source: card.lua:3980."""
    if ctx.joker_main:
        return JokerResult(mult_mod=card.ability.get("mult", 4))
    return None


@register("j_misprint")
def _misprint(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Misprint: random mult between extra.min and extra.max each hand.

    Source: card.lua:3700 — ``pseudorandom('misprint', min, max)``.
    Rolls a fresh value every hand via the 'misprint' RNG stream.
    """
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        lo = extra.get("min", 0)
        hi = extra.get("max", 23)
        if ctx.rng is not None:
            roll = ctx.rng.random("misprint", min_val=lo, max_val=hi)
        else:
            roll = 0
        return JokerResult(mult_mod=roll)
    return None


@register("j_stuntman")
def _stuntman(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Stuntman: +chip_mod chips (default 250). Source: card.lua:3713."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        return JokerResult(chip_mod=extra.get("chip_mod", 250))
    return None


# ---------------------------------------------------------------------------
# Helper: poker_hands containment check
# ---------------------------------------------------------------------------


def _contains(ctx: JokerContext, hand_type: str) -> bool:
    """Check if poker_hands contains a non-empty entry for *hand_type*.

    Equivalent to Lua's ``next(context.poker_hands[type])``.
    """
    if ctx.poker_hands is None:
        return False
    entries = ctx.poker_hands.get(hand_type)
    return bool(entries)


# ---------------------------------------------------------------------------
# Joker handlers — hand-type mult bonuses (Category A pattern, but uses
# poker_hands containment check per source card.lua:3660)
# ---------------------------------------------------------------------------


@register("j_jolly")
def _jolly(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Jolly Joker: +t_mult if hand contains Pair. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 8))
    return None


@register("j_zany")
def _zany(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Zany Joker: +t_mult if hand contains Three of a Kind. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 12))
    return None


@register("j_mad")
def _mad(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mad Joker: +t_mult if hand contains Two Pair. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Two Pair")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 10))
    return None


@register("j_crazy")
def _crazy(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Crazy Joker: +t_mult if hand contains Straight. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 12))
    return None


@register("j_droll")
def _droll(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Droll Joker: +t_mult if hand contains Flush. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 10))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type chip bonuses (poker_hands containment check)
# Source: card.lua:3666
# ---------------------------------------------------------------------------


@register("j_sly")
def _sly(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Sly Joker: +t_chips if hand contains Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 50))
    return None


@register("j_wily")
def _wily(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Wily Joker: +t_chips if hand contains Three of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 100))
    return None


@register("j_clever")
def _clever(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Clever Joker: +t_chips if hand contains Two Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Two Pair")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 80))
    return None


@register("j_devious")
def _devious(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Devious Joker: +t_chips if hand contains Straight."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 100))
    return None


@register("j_crafty")
def _crafty(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Crafty Joker: +t_chips if hand contains Flush."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 80))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type xMult bonuses (poker_hands containment check)
# Source: card.lua:3653
# ---------------------------------------------------------------------------


@register("j_duo")
def _duo(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Duo: xMult if hand contains Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        x = card.ability.get("x_mult", 2)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_trio")
def _trio(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Trio: xMult if hand contains Three of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        x = card.ability.get("x_mult", 3)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_family")
def _family(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Family: xMult if hand contains Four of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Four of a Kind")):
        x = card.ability.get("x_mult", 4)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_order")
def _order(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Order: xMult if hand contains Straight."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        x = card.ability.get("x_mult", 3)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_tribe")
def _tribe(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Tribe: xMult if hand contains Flush."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        x = card.ability.get("x_mult", 2)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


# ---------------------------------------------------------------------------
# Joker handlers — scoring_name-based (exact hand type match)
# ---------------------------------------------------------------------------


@register("j_supernova")
def _supernova(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Supernova: +mult = times this hand type played in the run.

    Source: card.lua:3731 — uses ``G.GAME.hands[context.scoring_name].played``.
    """
    if ctx.joker_main and ctx.scoring_name and ctx.hand_levels is not None:
        played = ctx.hand_levels[ctx.scoring_name].played
        return JokerResult(mult_mod=played)
    return None


@register("j_card_sharp")
def _card_sharp(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Card Sharp: xMult if same hand type played twice this round.

    Source: card.lua:4040 — checks ``G.GAME.hands[scoring_name].played_this_round > 1``.
    """
    if ctx.joker_main and ctx.scoring_name and ctx.hand_levels is not None:
        ptr = ctx.hand_levels[ctx.scoring_name].played_this_round
        if ptr > 1:
            extra = card.ability.get("extra", {})
            return JokerResult(Xmult_mod=extra.get("Xmult", 3))
    return None


# ---------------------------------------------------------------------------
# Helper: suit check on other_card
# ---------------------------------------------------------------------------


def _is_suit(ctx: JokerContext, suit: str) -> bool:
    """Check if ctx.other_card matches *suit* using Card.is_suit.

    Passes ``smeared`` from context. Wild Cards match any suit automatically
    via Card.is_suit internals.
    """
    if ctx.other_card is None:
        return False
    return ctx.other_card.is_suit(suit, smeared=ctx.smeared)


# ---------------------------------------------------------------------------
# Joker handlers — suit-conditional per scored card (individual context)
# Source: card.lua:3065 (Suit Mult), 3224-3260 (named suit jokers)
# ---------------------------------------------------------------------------


@register("j_greedy_joker")
def _greedy_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Greedy Joker: +s_mult per Diamond scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Diamonds")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_lusty_joker")
def _lusty_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Lusty Joker: +s_mult per Heart scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Hearts")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_wrathful_joker")
def _wrathful_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Wrathful Joker: +s_mult per Spade scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Spades")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_gluttenous_joker")
def _gluttenous_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Gluttonous Joker: +s_mult per Club scored. Source: card.lua:3065.

    Note: center key has typo ``j_gluttenous_joker`` (double t) matching source.
    """
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Clubs")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_arrowhead")
def _arrowhead(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Arrowhead: +chips per Spade scored. Source: card.lua:3240."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Spades"):
            return JokerResult(chips=card.ability.get("extra", 50))
    return None


@register("j_onyx_agate")
def _onyx_agate(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Onyx Agate: +mult per Club scored. Source: card.lua:3233."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Clubs"):
            return JokerResult(mult=card.ability.get("extra", 7))
    return None


@register("j_rough_gem")
def _rough_gem(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Rough Gem: +$1 per Diamond scored. Source: card.lua:3224."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Diamonds"):
            return JokerResult(dollars=card.ability.get("extra", 1))
    return None


@register("j_bloodstone")
def _bloodstone(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bloodstone: probabilistic xMult per Heart scored.

    Source: card.lua:3247 — ``pseudorandom('bloodstone') < normal/odds``.
    """
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Hearts"):
            extra = card.ability.get("extra", {})
            odds = extra.get("odds", 2)
            if ctx.rng is not None:
                roll = ctx.rng.random("bloodstone")
                if roll < ctx.game.probabilities_normal / odds:
                    return JokerResult(x_mult=extra.get("Xmult", 1.5))
            # No RNG → no effect (same as Lucky Card)
    return None


@register("j_ancient")
def _ancient(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ancient Joker: xMult if scored card matches current round's ancient suit.

    Source: card.lua:3255 — uses ``G.GAME.current_round.ancient_card.suit``.
    """
    if ctx.individual and ctx.cardarea == "play":
        if ctx.game.ancient_suit and _is_suit(ctx, ctx.game.ancient_suit):
            return JokerResult(x_mult=card.ability.get("extra", 1.5))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — rank-conditional per scored card (individual context)
# Source: card.lua:3065-3270
# ---------------------------------------------------------------------------

_FIBONACCI_IDS = frozenset({2, 3, 5, 8, 14})  # Ace=14


@register("j_fibonacci")
def _fibonacci(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Fibonacci: +mult for Ace/2/3/5/8. Source: card.lua:3185."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() in _FIBONACCI_IDS:
            return JokerResult(mult=card.ability.get("extra", 8))
    return None


@register("j_scholar")
def _scholar(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Scholar: +chips and +mult for Ace. Source: card.lua:3159."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 14:
            extra = card.ability.get("extra", {})
            return JokerResult(
                chips=extra.get("chips", 20),
                mult=extra.get("mult", 4),
            )
    return None


@register("j_walkie_talkie")
def _walkie_talkie(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Walkie Talkie: +chips and +mult for 10 or 4. Source: card.lua:3167."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if oid == 10 or oid == 4:
            extra = card.ability.get("extra", {})
            return JokerResult(
                chips=extra.get("chips", 10),
                mult=extra.get("mult", 4),
            )
    return None


@register("j_even_steven")
def _even_steven(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Even Steven: +mult for even numbered cards (2/4/6/8/10). Source: card.lua:3196."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if 0 <= oid <= 10 and oid % 2 == 0:
            return JokerResult(mult=card.ability.get("extra", 4))
    return None


@register("j_odd_todd")
def _odd_todd(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Odd Todd: +chips for odd numbered cards (3/5/7/9) and Ace. Source: card.lua:3206."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if (0 <= oid <= 10 and oid % 2 == 1) or oid == 14:
            return JokerResult(chips=card.ability.get("extra", 31))
    return None


@register("j_scary_face")
def _scary_face(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Scary Face: +chips for face cards. Source: card.lua:3136."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            return JokerResult(chips=card.ability.get("extra", 30))
    return None


@register("j_smiley")
def _smiley(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Smiley Face: +mult for face cards. Source: card.lua:3143."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            return JokerResult(mult=card.ability.get("extra", 5))
    return None


@register("j_photograph")
def _photograph(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Photograph: xMult on FIRST face card in scoring hand only.

    Source: card.lua:3093 — loops scoring_hand to find first is_face(),
    then checks ``context.other_card == first_face`` by identity.
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.scoring_hand:
            first_face = None
            for sc in ctx.scoring_hand:
                if sc.is_face(pareidolia=ctx.pareidolia):
                    first_face = sc
                    break
            if ctx.other_card is first_face:
                return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


@register("j_triboulet")
def _triboulet(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Triboulet: x2 mult for King or Queen. Source: card.lua:3262."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if oid == 12 or oid == 13:
            return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


@register("j_idol")
def _idol(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Idol: xMult if scored card matches idol_card rank AND suit.

    Source: card.lua:3127 — checks both get_id() and is_suit().
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.game.idol_card is not None:
            if ctx.other_card.get_id() == ctx.game.idol_card.get("id") and _is_suit(
                ctx, ctx.game.idol_card.get("suit", "")
            ):
                return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — rank-conditional retrigger (repetition context)
# Source: card.lua:3374
# ---------------------------------------------------------------------------

_HACK_IDS = frozenset({2, 3, 4, 5})


@register("j_hack")
def _hack(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hack: retrigger 2/3/4/5 cards. Source: card.lua:3374."""
    if ctx.repetition and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() in _HACK_IDS:
            return JokerResult(repetitions=card.ability.get("extra", 1))
    return None


@register("j_sock_and_buskin")
def _sock_and_buskin(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Sock and Buskin: +1 retrigger for face cards scored. Source: card.lua:3344."""
    if ctx.repetition and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            return JokerResult(repetitions=card.ability.get("extra", 1))
    return None


@register("j_hanging_chad")
def _hanging_chad(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hanging Chad: +2 retriggers for FIRST scored card. Source: card.lua:3352."""
    if ctx.repetition and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.scoring_hand and ctx.other_card is ctx.scoring_hand[0]:
            return JokerResult(repetitions=card.ability.get("extra", 2))
    return None


@register("j_dusk")
def _dusk(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Dusk: +1 retrigger for all scored cards on last hand. Source: card.lua:3360."""
    if ctx.repetition and ctx.cardarea == "play":
        if ctx.game.hands_left == 0:
            return JokerResult(repetitions=card.ability.get("extra", 1))
    return None


@register("j_selzer")
def _seltzer(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Seltzer: +1 retrigger for all scored cards. Decrements, self-destructs.

    Source: card.lua:3367 (repetition), 3601 (after).
    Key is ``j_selzer`` (typo in source). Starts with extra=10 uses.
    """
    if ctx.repetition and ctx.cardarea == "play":
        return JokerResult(repetitions=1)
    if ctx.after and not ctx.blueprint:
        uses = card.ability.get("extra", 10)
        if uses - 1 <= 0:
            return JokerResult(remove=True)
        card.ability["extra"] = uses - 1
        return JokerResult()
    return None


@register("j_mime")
def _mime(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mime: +1 retrigger for held cards. Source: card.lua:3387."""
    if ctx.repetition and ctx.cardarea == "hand":
        return JokerResult(repetitions=card.ability.get("extra", 1))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — game-state-dependent (joker_main context)
# ---------------------------------------------------------------------------


@register("j_half")
def _half(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Half Joker: +mult if ≤3 cards played. Source: card.lua:3672."""
    if ctx.joker_main and ctx.full_hand is not None:
        extra = card.ability.get("extra", {})
        if len(ctx.full_hand) <= extra.get("size", 3):
            return JokerResult(mult_mod=extra.get("mult", 20))
    return None


@register("j_abstract")
def _abstract(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Abstract Joker: +3 mult per joker owned. Source: card.lua:3678."""
    if ctx.joker_main:
        return JokerResult(
            mult_mod=ctx.game.joker_count * card.ability.get("extra", 3),
        )
    return None


@register("j_acrobat")
def _acrobat(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Acrobat: x3 mult on last hand of round. Source: card.lua:3688."""
    if ctx.joker_main and ctx.game.hands_left == 0:
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_mystic_summit")
def _mystic_summit(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mystic Summit: +15 mult when 0 discards left. Source: card.lua:3694."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        if ctx.game.discards_left == extra.get("d_remaining", 0):
            return JokerResult(mult_mod=extra.get("mult", 15))
    return None


@register("j_banner")
def _banner(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Banner: +30 chips per discard remaining. Source: card.lua:3707."""
    if ctx.joker_main and ctx.game.discards_left > 0:
        return JokerResult(
            chip_mod=ctx.game.discards_left * card.ability.get("extra", 30),
        )
    return None


@register("j_blue_joker")
def _blue_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blue Joker: +2 chips per card remaining in deck. Source: card.lua:3887."""
    if ctx.joker_main and ctx.game.deck_cards_remaining > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 2) * ctx.game.deck_cards_remaining,
        )
    return None


@register("j_erosion")
def _erosion(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Erosion: +4 mult per card below starting deck size. Source: card.lua:3894."""
    if ctx.joker_main:
        below = ctx.game.starting_deck_size - ctx.game.playing_cards_count
        if below > 0:
            return JokerResult(
                mult_mod=card.ability.get("extra", 4) * below,
            )
    return None


@register("j_stone")
def _stone_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Stone Joker: +25 chips per Stone Card in full deck. Source: card.lua:3922."""
    if ctx.joker_main and ctx.game.stone_tally > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 25) * ctx.game.stone_tally,
        )
    return None


@register("j_steel_joker")
def _steel_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Steel Joker: x(1 + 0.2 × steel_count) mult. Source: card.lua:3929."""
    if ctx.joker_main and ctx.game.steel_tally > 0:
        return JokerResult(
            Xmult_mod=1 + card.ability.get("extra", 0.2) * ctx.game.steel_tally,
        )
    return None


@register("j_bull")
def _bull(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bull: +2 chips per $1 held. Source: card.lua:3936."""
    if ctx.joker_main and ctx.game.money > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 2) * max(0, ctx.game.money),
        )
    return None


@register("j_drivers_license")
def _drivers_license(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Driver's License: x3 if ≥16 enhanced cards in deck. Source: card.lua:3943."""
    if ctx.joker_main and ctx.game.enhanced_card_count >= 16:
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_blackboard")
def _blackboard(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blackboard: x3 if ALL held cards are Spades or Clubs. Source: card.lua:3951.

    Uses ``flush_calc=True`` in is_suit calls (matching source).
    """
    if ctx.joker_main and ctx.held_cards is not None:
        if not ctx.held_cards:
            return None
        for c in ctx.held_cards:
            if not (c.is_suit("Clubs", flush_calc=True) or c.is_suit("Spades", flush_calc=True)):
                return None
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_stencil")
def _stencil(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Joker Stencil: xMult = empty joker slots. Source: card.lua:3966.

    Formula: ``joker_slots - joker_count + stencil_count`` where stencil_count
    includes this stencil (so stencils don't count against empty slots).
    """
    if ctx.joker_main:
        stencil_count = sum(
            1
            for j in (ctx.jokers or [])
            if getattr(j, "center_key", None) == "j_stencil" and not getattr(j, "debuff", False)
        )
        x = ctx.game.joker_slots - ctx.game.joker_count + stencil_count
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_flower_pot")
def _flower_pot(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Flower Pot: xMult if scoring hand contains all 4 suits. Source: card.lua:3807.

    Non-Wild cards counted first, then Wild cards fill remaining suits.
    """
    if ctx.joker_main and ctx.scoring_hand:
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for c in ctx.scoring_hand:
            if c.ability.get("name") != "Wild Card":
                for s in suits:
                    if suits[s] == 0 and c.is_suit(s, bypass_debuff=True):
                        suits[s] += 1
                        break
        for c in ctx.scoring_hand:
            if c.ability.get("name") == "Wild Card":
                for s in suits:
                    if suits[s] == 0 and c.is_suit(s):
                        suits[s] += 1
                        break
        if all(v > 0 for v in suits.values()):
            return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_seeing_double")
def _seeing_double(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Seeing Double: xMult if hand has Club + card of another suit. Source: card.lua:3840.

    Non-Wild cards counted first, then Wild cards fill (Clubs priority).
    """
    if ctx.joker_main and ctx.scoring_hand:
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for c in ctx.scoring_hand:
            if c.ability.get("name") != "Wild Card":
                for s in suits:
                    if c.is_suit(s):
                        suits[s] += 1
        for c in ctx.scoring_hand:
            if c.ability.get("name") == "Wild Card":
                if suits["Clubs"] == 0 and c.is_suit("Clubs"):
                    suits["Clubs"] += 1
                elif suits["Diamonds"] == 0 and c.is_suit("Diamonds"):
                    suits["Diamonds"] += 1
                elif suits["Spades"] == 0 and c.is_suit("Spades"):
                    suits["Spades"] += 1
                elif suits["Hearts"] == 0 and c.is_suit("Hearts"):
                    suits["Hearts"] += 1
        has_club = suits["Clubs"] > 0
        has_other = suits["Hearts"] > 0 or suits["Diamonds"] > 0 or suits["Spades"] > 0
        if has_club and has_other:
            return JokerResult(Xmult_mod=card.ability.get("extra", 2))
    return None


@register("j_bootstraps")
def _bootstraps(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bootstraps: +2 mult per $5 held. Source: card.lua:4046."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        per = extra.get("dollars", 5)
        buckets = ctx.game.money // per if per > 0 else 0
        if buckets >= 1:
            return JokerResult(mult_mod=extra.get("mult", 2) * buckets)
    return None


@register("j_fortune_teller")
def _fortune_teller(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Fortune Teller: +mult = total tarot uses in run. Source: card.lua:4016."""
    if ctx.joker_main and ctx.game.consumable_usage_tarot > 0:
        return JokerResult(mult_mod=ctx.game.consumable_usage_tarot)
    return None


@register("j_loyalty_card")
def _loyalty_card(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Loyalty Card: xMult every N+1 hands. Source: card.lua:3632.

    Formula: ``(every-1-(hands_played-hands_at_create)) % (every+1)``.
    Triggers when result equals ``every``.
    """
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        every = extra.get("every", 5)
        hands_at_create = card.ability.get("hands_played_at_create", 0)
        remaining = (every - 1 - (ctx.game.hands_played - hands_at_create)) % (every + 1)
        if remaining == every:
            return JokerResult(Xmult_mod=extra.get("Xmult", 4))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — held-card effects (individual context, cardarea='hand')
# ---------------------------------------------------------------------------


@register("j_raised_fist")
def _raised_fist(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Raised Fist: +2× lowest held card's rank as mult. Source: card.lua:3320.

    Finds the lowest-rank non-Stone card in held_cards, then fires only
    when other_card is that specific card (identity check).
    """
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.held_cards:
            lowest_id = 15
            lowest_card = None
            for c in ctx.held_cards:
                if c.ability.get("effect") != "Stone Card" and c.get_id() < lowest_id:
                    lowest_id = c.get_id()
                    lowest_card = c
            if ctx.other_card is lowest_card:
                if ctx.other_card.debuff:
                    return JokerResult(message="Debuffed")
                nominal = ctx.other_card.base.nominal if ctx.other_card.base else 0
                return JokerResult(h_mult=2 * nominal)
    return None


@register("j_shoot_the_moon")
def _shoot_the_moon(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Shoot the Moon: +13 mult per Queen held. Source: card.lua:3272."""
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 12:
            if ctx.other_card.debuff:
                return JokerResult(message="Debuffed")
            return JokerResult(h_mult=card.ability.get("extra", 13))
    return None


@register("j_baron")
def _baron(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Baron: x1.5 per King held. Source: card.lua:3287."""
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 13:
            if ctx.other_card.debuff:
                return JokerResult(message="Debuffed")
            return JokerResult(x_mult=card.ability.get("extra", 1.5))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — scoring-phase economy (individual context)
# ---------------------------------------------------------------------------


@register("j_ticket")
def _golden_ticket(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Golden Ticket: +$4 per Gold Card played. Source: card.lua:3150."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.ability.get("name") == "Gold Card":
            return JokerResult(dollars=card.ability.get("extra", 4))
    return None


@register("j_business")
def _business(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Business Card: face card played → 1/2 chance → +$2. Source: card.lua:3175.

    Probabilistic: ``pseudorandom('business') < normal/extra``.
    Returns hardcoded $2 (not from config).
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            if ctx.rng is not None:
                odds = card.ability.get("extra", 2)
                if ctx.rng.random("business") < ctx.game.probabilities_normal / odds:
                    return JokerResult(dollars=2)
    return None


@register("j_reserved_parking")
def _reserved_parking(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Reserved Parking: face card held → 1/2 chance → +$1. Source: card.lua:3302.

    Fires in individual/hand context (held cards). Checks debuff.
    """
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            extra = card.ability.get("extra", {})
            odds = extra.get("odds", 2)
            if ctx.rng is not None:
                if ctx.rng.random("parking") < ctx.game.probabilities_normal / odds:
                    if ctx.other_card.debuff:
                        return JokerResult(message="Debuffed")
                    return JokerResult(dollars=extra.get("dollars", 1))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — discard-phase economy (discard context)
# ---------------------------------------------------------------------------


@register("j_faceless")
def _faceless(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Faceless Joker: +$5 if ≥3 face cards discarded. Source: card.lua:2858.

    Fires once per discard when other_card is the last card in full_hand.
    """
    if ctx.discard and ctx.other_card is not None and ctx.full_hand:
        if ctx.other_card is ctx.full_hand[-1]:
            extra = card.ability.get("extra", {})
            face_count = sum(1 for c in ctx.full_hand if c.is_face(pareidolia=ctx.pareidolia))
            if face_count >= extra.get("faces", 3):
                return JokerResult(dollars=extra.get("dollars", 5))
    return None


@register("j_mail")
def _mail(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mail-In Rebate: +$5 per card discarded matching mail_card rank.

    Source: card.lua:2825. Fires per discarded card.
    """
    if ctx.discard and ctx.other_card is not None:
        if not ctx.other_card.debuff and ctx.game.mail_card_id is not None:
            if ctx.other_card.get_id() == ctx.game.mail_card_id:
                return JokerResult(dollars=card.ability.get("extra", 5))
    return None


@register("j_trading")
def _trading(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Trading Card: first discard of round, single card → +$3, destroy card.

    Source: card.lua:2802. Returns extra={'destroy': True} to signal destruction.
    """
    if ctx.discard and ctx.full_hand is not None:
        if ctx.game.discards_used <= 0 and len(ctx.full_hand) == 1:
            return JokerResult(
                dollars=card.ability.get("extra", 3),
                remove=True,
                extra={"destroy": True},
            )
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type economy (joker_main context)
# ---------------------------------------------------------------------------


@register("j_todo_list")
def _to_do_list(card: Card, ctx: JokerContext) -> JokerResult | None:
    """To Do List: +$4 if hand matches to_do_poker_hand. Source: card.lua:3491."""
    if ctx.joker_main and ctx.scoring_name:
        target = card.ability.get("to_do_poker_hand")
        if target and ctx.scoring_name == target:
            extra = card.ability.get("extra", {})
            return JokerResult(dollars=extra.get("dollars", 4))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — boss blind reaction
# ---------------------------------------------------------------------------


@register("j_matador")
def _matador(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Matador: +$8 when boss blind's debuff effect triggers. Source: card.lua:2736."""
    if ctx.debuffed_hand and ctx.blind is not None and ctx.blind.triggered:
        return JokerResult(dollars=card.ability.get("extra", 8))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — copy/delegation (Blueprint, Brainstorm)
# Source: card.lua:2304-2334
# ---------------------------------------------------------------------------


def _find_right_neighbor(card: Card, ctx: JokerContext) -> Card | None:
    """Find the joker immediately to the right of *card* in the joker list."""
    if ctx.jokers is None:
        return None
    for i, j in enumerate(ctx.jokers):
        if j is card and i + 1 < len(ctx.jokers):
            return ctx.jokers[i + 1]
    return None


def _find_leftmost(card: Card, ctx: JokerContext) -> Card | None:
    """Find the leftmost joker that isn't *card*."""
    if ctx.jokers is None:
        return None
    for j in ctx.jokers:
        if j is not card:
            return j
    return None


@register("j_blueprint")
def _blueprint(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blueprint: copy the joker to the right. Source: card.lua:2304.

    Loop prevention: ``blueprint > len(jokers) + 1`` matches source.
    """
    joker_count = len(ctx.jokers) if ctx.jokers else 0
    bp = (ctx.blueprint + 1) if ctx.blueprint else 1
    if bp > joker_count + 1:
        return None
    target = _find_right_neighbor(card, ctx)
    if target is None or target.debuff:
        return None
    new_ctx = replace(
        ctx,
        blueprint=bp,
        blueprint_card=ctx.blueprint_card or card,
    )
    return calculate_joker(target, new_ctx)


@register("j_brainstorm")
def _brainstorm(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Brainstorm: copy the leftmost joker. Source: card.lua:2321.

    Loop prevention same as Blueprint.
    """
    joker_count = len(ctx.jokers) if ctx.jokers else 0
    bp = (ctx.blueprint + 1) if ctx.blueprint else 1
    if bp > joker_count + 1:
        return None
    target = _find_leftmost(card, ctx)
    if target is None or target.debuff:
        return None
    new_ctx = replace(
        ctx,
        blueprint=bp,
        blueprint_card=ctx.blueprint_card or card,
    )
    return calculate_joker(target, new_ctx)


# ---------------------------------------------------------------------------
# Joker handlers — scaling (mutate ability state per hand/discard/round)
# ---------------------------------------------------------------------------


@register("j_green_joker")
def _green_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Green Joker: +1 mult per hand, -1 per discard. Source: card.lua:3563, 2846.

    Before context: ability.mult += hand_add (1).
    Discard context: ability.mult -= discard_sub (1), clamped to 0.
    Joker main: returns accumulated mult.
    Blueprint skips mutation (``not context.blueprint``).
    """
    extra = card.ability.get("extra", {})
    if ctx.before and not ctx.blueprint:
        card.ability["mult"] = card.ability.get("mult", 0) + extra.get("hand_add", 1)
        return JokerResult()
    if ctx.discard and ctx.other_card is not None and ctx.full_hand:
        if not ctx.blueprint and ctx.other_card is ctx.full_hand[-1]:
            prev = card.ability.get("mult", 0)
            card.ability["mult"] = max(0, prev - extra.get("discard_sub", 1))
            return JokerResult() if card.ability["mult"] != prev else None
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_ride_the_bus")
def _ride_the_bus(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ride the Bus: +1 mult per consecutive hand with no face cards scored.

    Source: card.lua:3525. Resets to 0 when a face card IS scored.
    """
    if ctx.before and not ctx.blueprint and ctx.scoring_hand is not None:
        has_face = any(c.is_face() for c in ctx.scoring_hand)
        if has_face:
            card.ability["mult"] = 0
        else:
            card.ability["mult"] = card.ability.get("mult", 0) + card.ability.get("extra", 1)
        return JokerResult()
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_trousers")
def _spare_trousers(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Spare Trousers: +2 mult when Two Pair or Full House played.

    Source: card.lua:3412. Checks poker_hands containment for both.
    """
    if ctx.before and not ctx.blueprint and ctx.poker_hands is not None:
        has_tp = bool(ctx.poker_hands.get("Two Pair"))
        has_fh = bool(ctx.poker_hands.get("Full House"))
        if has_tp or has_fh:
            card.ability["mult"] = card.ability.get("mult", 0) + card.ability.get("extra", 2)
            return JokerResult()
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_square")
def _square_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Square Joker: +4 chips when exactly 4 cards played.

    Source: card.lua:3427. Mutates ability.extra.chips.
    """
    if ctx.before and not ctx.blueprint and ctx.full_hand is not None:
        if len(ctx.full_hand) == 4:
            extra = card.ability.get("extra", {})
            extra["chips"] = extra.get("chips", 0) + extra.get("chip_mod", 4)
            card.ability["extra"] = extra
            return JokerResult()
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 0)
        if chips > 0:
            return JokerResult(chip_mod=chips)
    return None


@register("j_ice_cream")
def _ice_cream(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ice Cream: starts +100 chips, -5 per hand. Self-destructs at 0.

    Source: card.lua:3571. Mutation in after context.
    """
    if ctx.after and not ctx.blueprint:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 100)
        chip_mod = extra.get("chip_mod", 5)
        if chips - chip_mod <= 0:
            return JokerResult(remove=True)
        extra["chips"] = chips - chip_mod
        card.ability["extra"] = extra
        return JokerResult()
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 100)
        if chips > 0:
            return JokerResult(chip_mod=chips)
    return None


@register("j_popcorn")
def _popcorn(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Popcorn: starts +20 mult, -4 per round. Self-destructs at 0.

    Source: card.lua:2945. Mutation in end_of_round context.
    """
    if ctx.end_of_round and not ctx.blueprint:
        m = card.ability.get("mult", 20)
        sub = card.ability.get("extra", 4)
        if m - sub <= 0:
            return JokerResult(remove=True)
        card.ability["mult"] = m - sub
        return JokerResult()
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_flash")
def _flash_card(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Flash Card: +2 mult per shop reroll. Source: card.lua:2403."""
    if ctx.reroll_shop and not ctx.blueprint:
        card.ability["mult"] = card.ability.get("mult", 0) + card.ability.get("extra", 2)
        return JokerResult()
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_red_card")
def _red_card(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Red Card: +3 mult per booster pack skipped. Source: card.lua:2441."""
    if ctx.skipping_booster and not ctx.blueprint:
        card.ability["mult"] = card.ability.get("mult", 0) + card.ability.get("extra", 3)
        return JokerResult()
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


@register("j_wee")
def _wee_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Wee Joker: +8 chips per 2-rank card scored. Source: card.lua:3083.

    Accumulates on ability.extra.chips. Mutation in individual/play context.
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 2 and not ctx.blueprint:
            extra = card.ability.get("extra", {})
            extra["chips"] = extra.get("chips", 0) + extra.get("chip_mod", 8)
            card.ability["extra"] = extra
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 0)
        if chips > 0:
            return JokerResult(chip_mod=chips)
    return None


@register("j_lucky_cat")
def _lucky_cat(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Lucky Cat: +0.25 xMult per Lucky Card trigger. Source: card.lua:3076.

    Accumulates on ability.x_mult. Scored via the generic x_mult > 1 handler
    pattern (card.lua:3653) with type=''.
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if getattr(ctx.other_card, "lucky_trigger", False) and not ctx.blueprint:
            card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get("extra", 0.25)
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


# ---------------------------------------------------------------------------
# Joker handlers — xMult scaling (accumulate multiplicative mult over time)
# ---------------------------------------------------------------------------


@register("j_campfire")
def _campfire(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Campfire: +0.25 xMult per card sold. Resets on boss blind.

    Source: card.lua:2396 (selling_card), 2889 (end_of_round boss reset).
    """
    if ctx.selling_card and not ctx.blueprint:
        card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get("extra", 0.25)
        return JokerResult()
    if ctx.end_of_round and not ctx.blueprint:
        if ctx.blind and getattr(ctx.blind, "boss", False) and card.ability.get("x_mult", 1) > 1:
            card.ability["x_mult"] = 1
            return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_hologram")
def _hologram(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hologram: +0.25 xMult per card added to deck. Source: card.lua:2457."""
    if ctx.playing_card_added and not ctx.blueprint and ctx.cards:
        card.ability["x_mult"] = card.ability.get("x_mult", 1) + len(ctx.cards) * card.ability.get(
            "extra", 0.25
        )
        return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_constellation")
def _constellation(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Constellation: +0.1 xMult per Planet card used. Source: card.lua:2727."""
    if ctx.using_consumeable and not ctx.blueprint:
        if ctx.consumeable and ctx.consumeable.ability.get("set") == "Planet":
            card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get("extra", 0.1)
            return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_glass")
def _glass_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Glass Joker: +0.75 xMult per Glass Card destroyed. Source: card.lua:2647."""
    if ctx.cards_destroyed and not ctx.blueprint:
        glass_count = sum(
            1
            for c in ctx.cards_destroyed
            if c.ability.get("effect") == "Glass Card" or c.ability.get("name") == "Glass Card"
        )
        if glass_count > 0:
            card.ability["x_mult"] = (
                card.ability.get("x_mult", 1) + card.ability.get("extra", 0.75) * glass_count
            )
            return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_caino")
def _caino(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Caino: +1.0 xMult per face card destroyed. Source: card.lua:2623.

    Uses ``ability.caino_xmult`` (separate from x_mult).
    """
    if ctx.cards_destroyed and not ctx.blueprint:
        face_count = sum(1 for c in ctx.cards_destroyed if c.is_face())
        if face_count > 0:
            card.ability["caino_xmult"] = (
                card.ability.get("caino_xmult", 1) + card.ability.get("extra", 1) * face_count
            )
            return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("caino_xmult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_vampire")
def _vampire(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Vampire: +0.1 xMult per enhancement stripped from scored cards.

    Source: card.lua:3465. Fires in individual_hand_end context.
    Side effect: strips enhancement from scored cards (sets ability to c_base).
    """
    if ctx.individual_hand_end and not ctx.blueprint and ctx.scoring_hand:
        enhanced_count = 0
        for c in ctx.scoring_hand:
            if (
                c.ability.get("effect", "") not in ("", "Default Base")
                and c.ability.get("name", "") != "Default Base"
                and not c.debuff
                and not getattr(c, "vampired", False)
            ):
                enhanced_count += 1
                c.vampired = True
                c.set_ability("c_base")
        if enhanced_count > 0:
            card.ability["x_mult"] = (
                card.ability.get("x_mult", 1) + card.ability.get("extra", 0.1) * enhanced_count
            )
            return JokerResult(Xmult_mod=card.ability["x_mult"])
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_obelisk")
def _obelisk(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Obelisk: +0.2 xMult per consecutive non-most-played hand.

    Source: card.lua:3543. Resets to 1 when most-played hand IS played.
    """
    if ctx.individual_hand_end and not ctx.blueprint:
        if ctx.scoring_name and ctx.hand_levels is not None:
            current_played = ctx.hand_levels[ctx.scoring_name].played
            is_most_played = True
            for ht, info in ctx.hand_levels._hands.items():
                if ht.value != ctx.scoring_name and info.played >= current_played and info.visible:
                    is_most_played = False
                    break
            if is_most_played:
                if card.ability.get("x_mult", 1) > 1:
                    card.ability["x_mult"] = 1
            else:
                card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get(
                    "extra", 0.2
                )
        return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_madness")
def _madness(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Madness: +0.5 xMult per non-boss blind. Destroys random joker.

    Source: card.lua:2503. Side effect returned as extra.
    """
    if ctx.setting_blind and not ctx.blueprint:
        if ctx.blind and not getattr(ctx.blind, "boss", False):
            card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get("extra", 0.5)
            return JokerResult(extra={"destroy_random_joker": True})
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_hit_the_road")
def _hit_the_road(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hit the Road: +0.5 xMult per Jack discarded this round.

    Source: card.lua:2835. Resets at end of round.
    """
    if ctx.discard and ctx.other_card is not None and not ctx.blueprint:
        if not ctx.other_card.debuff and ctx.other_card.get_id() == 11:
            card.ability["x_mult"] = card.ability.get("x_mult", 1) + card.ability.get("extra", 0.5)
            return JokerResult(Xmult_mod=card.ability["x_mult"])
    if ctx.end_of_round and not ctx.blueprint:
        if card.ability.get("x_mult", 1) > 1:
            card.ability["x_mult"] = 1
            return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_throwback")
def _throwback(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Throwback: xMult = 1 + 0.25 * total_blinds_skipped.

    Source: card.lua:4176. Formula-based, not cumulative.
    """
    if ctx.joker_main:
        x = 1 + ctx.game.skips * card.ability.get("extra", 0.25)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_yorick")
def _yorick(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Yorick: xMult +1 every 23 discards. Source: card.lua:2788.

    Counter on ability.yorick_discards decrements per card discarded.
    """
    if ctx.discard and not ctx.blueprint:
        counter = card.ability.get("yorick_discards", 23)
        if counter <= 1:
            extra = card.ability.get("extra", {})
            card.ability["yorick_discards"] = extra.get("discards", 23)
            card.ability["x_mult"] = card.ability.get("x_mult", 1) + extra.get("xmult", 1)
            return JokerResult(Xmult_mod=card.ability["x_mult"])
        card.ability["yorick_discards"] = counter - 1
        return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 1)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_ceremonial")
def _ceremonial(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ceremonial Dagger: +2× right neighbor's sell_cost as mult when destroyed.

    Source: card.lua:2561. Fires in setting_blind. Returns mult_mod (NOT xMult).
    Side effect: signals destruction of right neighbor via extra.
    """
    if ctx.setting_blind and not ctx.blueprint and ctx.jokers:
        my_pos = None
        for i, j in enumerate(ctx.jokers):
            if j is card:
                my_pos = i
                break
        if my_pos is not None and my_pos + 1 < len(ctx.jokers):
            target = ctx.jokers[my_pos + 1]
            if (
                not getattr(target, "eternal", False)
                and not getattr(target, "getting_sliced", False)
                and not getattr(card, "getting_sliced", False)
            ):
                card.ability["mult"] = card.ability.get("mult", 0) + target.sell_cost * 2
                return JokerResult(
                    extra={"destroy_joker": target},
                )
    if ctx.joker_main and card.ability.get("mult", 0) > 0:
        return JokerResult(mult_mod=card.ability["mult"])
    return None


# ---------------------------------------------------------------------------
# Joker handlers — joker-on-joker (other_joker context, Phase 9c)
# ---------------------------------------------------------------------------


def _get_rarity(joker_card: Card) -> int:
    """Look up rarity for a joker from centers.json. Returns 0 if unknown."""
    from jackdaw.engine.card import _resolve_center

    try:
        center = _resolve_center(joker_card.center_key)
        return center.get("rarity", 0)
    except (KeyError, AttributeError):
        return 0


@register("j_baseball")
def _baseball(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Baseball Card: x1.5 mult if other_joker is Uncommon (rarity 2).

    Source: card.lua:3396. Fires in other_joker context.
    Does NOT check debuff on the other joker (source omits this check).
    """
    if ctx.other_joker is not None and ctx.other_joker is not card:
        if _get_rarity(ctx.other_joker) == 2:
            return JokerResult(Xmult_mod=card.ability.get("extra", 1.5))
    return None


@register("j_swashbuckler")
def _swashbuckler(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Swashbuckler: +mult = sum of all other jokers' sell values.

    Source: card.lua:4240 (computed in Card:update, not calculate_joker).
    We compute it in joker_main since we don't have a game tick loop.
    """
    if ctx.joker_main and ctx.jokers:
        sell_total = sum(j.sell_cost for j in ctx.jokers if j is not card and not j.debuff)
        if sell_total > 0:
            return JokerResult(mult_mod=sell_total)
    return None


# ---------------------------------------------------------------------------
# Joker handlers — card creation (stubbed as side-effect descriptors)
# Actual card creation deferred to M10 (pool generation).
# ---------------------------------------------------------------------------


@register("j_certificate")
def _certificate(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Certificate: create playing card with random seal on first hand drawn.

    Source: card.lua:2462. Seeds: 'cert_fr' (card), 'certsl' (seal).
    """
    if ctx.first_hand_drawn:
        return JokerResult(
            extra={
                "create": {"type": "playing_card", "seal": True, "key": "cert"},
            }
        )
    return None


@register("j_marble")
def _marble(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Marble Joker: add Stone Card to deck on setting_blind.

    Source: card.lua:2580. Seed: 'marb_fr'.
    """
    if ctx.setting_blind and not getattr(card, "getting_sliced", False):
        return JokerResult(
            extra={
                "create": {"type": "playing_card", "enhancement": "m_stone", "key": "marble"},
            }
        )
    return None


@register("j_dna")
def _dna(card: Card, ctx: JokerContext) -> JokerResult | None:
    """DNA: copy first card into deck on first hand (1 card played).

    Source: card.lua:3501. Fires in before context when hands_played == 0
    and exactly 1 card in full_hand.
    """
    if ctx.before and not ctx.blueprint:
        if ctx.game.hands_played == 0 and ctx.full_hand and len(ctx.full_hand) == 1:
            return JokerResult(
                extra={
                    "create": {
                        "type": "playing_card_copy",
                        "source_card": ctx.full_hand[0],
                        "key": "dna",
                    },
                }
            )
    return None


@register("j_riff_raff")
def _riff_raff(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Riff-raff: create up to 2 Common jokers on setting_blind.

    Source: card.lua:2529. Checks joker slot availability.
    """
    if ctx.setting_blind and not getattr(card, "getting_sliced", False):
        joker_count = ctx.game.joker_count if ctx.game else 0
        joker_slots = ctx.game.joker_slots if ctx.game else 5
        available = joker_slots - joker_count
        if available > 0:
            count = min(card.ability.get("extra", 2), available)
            return JokerResult(
                extra={
                    "create": {
                        "type": "Joker",
                        "rarity": "Common",
                        "count": count,
                        "key": "rif",
                    },
                }
            )
    return None


@register("j_cartomancer")
def _cartomancer(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Cartomancer: create Tarot card on setting_blind.

    Source: card.lua:2545. Checks consumable slot.
    """
    if ctx.setting_blind and not getattr(card, "getting_sliced", False):
        return JokerResult(
            extra={
                "create": {"type": "Tarot", "key": "car"},
            }
        )
    return None


@register("j_8_ball")
def _eight_ball(card: Card, ctx: JokerContext) -> JokerResult | None:
    """8 Ball: scored rank 8 → 1/4 chance → create Tarot.

    Source: card.lua:3106. Seed: '8ball'. Probability: normal/extra (1/4).
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 8:
            odds = card.ability.get("extra", 4)
            if ctx.rng is not None:
                if ctx.rng.random("8ball") < ctx.game.probabilities_normal / odds:
                    return JokerResult(
                        extra={
                            "create": {"type": "Tarot", "key": "8ba"},
                        }
                    )
    return None


@register("j_vagabond")
def _vagabond(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Vagabond: create Tarot if money ≤ $4. Source: card.lua:3743."""
    if ctx.joker_main:
        if ctx.game.money <= card.ability.get("extra", 4):
            return JokerResult(
                extra={
                    "create": {"type": "Tarot", "key": "vag"},
                }
            )
    return None


@register("j_superposition")
def _superposition(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Superposition: Ace in hand + Straight → create Tarot.

    Source: card.lua:3762. Checks scoring_hand for Ace and poker_hands
    for Straight.
    """
    if ctx.joker_main and ctx.scoring_hand and ctx.poker_hands:
        has_ace = any(c.get_id() == 14 for c in ctx.scoring_hand)
        has_straight = bool(ctx.poker_hands.get("Straight"))
        if has_ace and has_straight:
            return JokerResult(
                extra={
                    "create": {"type": "Tarot", "key": "sup"},
                }
            )
    return None


@register("j_seance")
def _seance(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Seance: hand matches target type → create Spectral.

    Source: card.lua:3787. Default target: Straight Flush.
    """
    if ctx.joker_main and ctx.poker_hands:
        extra = card.ability.get("extra", {})
        target = extra.get("poker_hand", "Straight Flush")
        if ctx.poker_hands.get(target):
            return JokerResult(
                extra={
                    "create": {"type": "Spectral", "key": "sea"},
                }
            )
    return None


@register("j_sixth_sense")
def _sixth_sense(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Sixth Sense: rank 6, first hand, 1 card played → create Spectral + destroy.

    Source: card.lua:2604. Fires in destroying_card context.
    """
    if ctx.destroying_card is not None and not ctx.blueprint:
        if (
            ctx.full_hand
            and len(ctx.full_hand) == 1
            and ctx.full_hand[0].get_id() == 6
            and ctx.game.hands_played == 0
        ):
            return JokerResult(
                remove=True,
                extra={"create": {"type": "Spectral", "key": "sixth"}},
            )
    return None


@register("j_hallucination")
def _hallucination(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hallucination: open_booster → probability roll → create Tarot.

    Source: card.lua:2335. Seed: 'halu'+ante. Probability: normal/extra (1/2).
    """
    if ctx.open_booster:
        odds = card.ability.get("extra", 2)
        if ctx.rng is not None:
            if ctx.rng.random("hallucination") < ctx.game.probabilities_normal / odds:
                return JokerResult(
                    extra={
                        "create": {"type": "Tarot", "key": "hal"},
                    }
                )
    return None


# ---------------------------------------------------------------------------
# Joker handlers — destructive / rule-modifying side effects
# ---------------------------------------------------------------------------


@register("j_gros_michel")
def _gros_michel(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Gros Michel: +15 mult. 1/6 chance of extinction per round.

    Source: card.lua:3019 (end_of_round), scoring via mult_mod.
    Sets pool_flags.gros_michel_extinct on destruction.
    """
    if ctx.end_of_round and not ctx.blueprint:
        extra = card.ability.get("extra", {})
        odds = extra.get("odds", 6)
        if ctx.rng is not None:
            if ctx.rng.random("gros_michel") < ctx.game.probabilities_normal / odds:
                return JokerResult(
                    remove=True,
                    extra={"pool_flag": "gros_michel_extinct"},
                )
        return JokerResult(saved=True)
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        m = extra.get("mult", 15)
        if m > 0:
            return JokerResult(mult_mod=m)
    return None


@register("j_cavendish")
def _cavendish(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Cavendish: x3 mult. 1/1000 chance of extinction per round.

    Source: card.lua:3019. Same destruction pattern as Gros Michel.
    """
    if ctx.end_of_round and not ctx.blueprint:
        extra = card.ability.get("extra", {})
        odds = extra.get("odds", 1000)
        if ctx.rng is not None:
            if ctx.rng.random("cavendish") < ctx.game.probabilities_normal / odds:
                return JokerResult(remove=True)
        return JokerResult(saved=True)
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        x = extra.get("Xmult", 3)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_chicot")
def _chicot(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Chicot: disable boss blind on setting_blind. Persists (does NOT self-destruct).

    Source: card.lua:2492. Also fires when added to deck (card.lua:596).
    """
    if ctx.setting_blind and not ctx.blueprint:
        if ctx.blind and getattr(ctx.blind, "boss", False):
            if not getattr(ctx.blind, "disabled", False):
                return JokerResult(extra={"disable_blind": True})
    return None


@register("j_luchador")
def _luchador(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Luchador: when sold, disable the current boss blind.

    Source: card.lua:2354. Checks blind is boss and not already disabled.
    """
    if ctx.selling_self:
        if ctx.blind and not getattr(ctx.blind, "disabled", False):
            if getattr(ctx.blind, "boss", False):
                return JokerResult(extra={"disable_blind": True})
    return None


@register("j_burglar")
def _burglar(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Burglar: on setting_blind, +3 hands, remove all discards.

    Source: card.lua:2522. Returns side-effect descriptor.
    """
    if ctx.setting_blind and not getattr(card, "getting_sliced", False):
        extra_hands = card.ability.get("extra", 3)
        return JokerResult(
            extra={
                "set_hands": extra_hands,
                "set_discards": 0,
            }
        )
    return None


@register("j_midas_mask")
def _midas_mask(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Midas Mask: convert all face cards in scoring hand to Gold Cards.

    Source: card.lua:3443. Fires in before context. Mutates card ability
    in-place — enhancement change persists for future hands.
    """
    if ctx.before and not ctx.blueprint and ctx.scoring_hand:
        faces = []
        for c in ctx.scoring_hand:
            if c.is_face(pareidolia=ctx.pareidolia) and not c.debuff:
                faces.append(c)
        if faces:
            for c in faces:
                c.set_ability("m_gold")
            return JokerResult()
    return None


@register("j_hiker")
def _hiker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hiker: +5 permanent chip bonus per scored card.

    Source: card.lua:3067. Fires in individual/play context.
    Mutates other_card.ability.perma_bonus in-place.
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if not ctx.blueprint:
            pb = ctx.other_card.ability.get("perma_bonus", 0)
            ctx.other_card.ability["perma_bonus"] = pb + card.ability.get("extra", 5)
        return JokerResult()
    return None


# ---------------------------------------------------------------------------
# Joker handlers — end-of-round dollar bonuses (calc_dollar_bonus)
# Source: card.lua:1658-1677
# ---------------------------------------------------------------------------


@register_dollars("j_golden")
def _golden_dollars(card: Card, game: GameSnapshot) -> int:
    """Golden Joker: +$4 per round. Source: card.lua:1658."""
    return card.ability.get("extra", 4)


@register_dollars("j_cloud_9")
def _cloud_9_dollars(card: Card, game: GameSnapshot) -> int:
    """Cloud 9: +$1 per 9-rank card in full deck. Source: card.lua:1661."""
    tally = card.ability.get("nine_tally", 0)
    if tally > 0:
        return card.ability.get("extra", 1) * tally
    return 0


@register_dollars("j_rocket")
def _rocket_dollars(card: Card, game: GameSnapshot) -> int:
    """Rocket: +$dollars (grows per boss beaten). Source: card.lua:1666."""
    extra = card.ability.get("extra", {})
    return extra.get("dollars", 1)


@register_dollars("j_satellite")
def _satellite_dollars(card: Card, game: GameSnapshot) -> int:
    """Satellite: +$1 per unique Planet type used. Source: card.lua:1669."""
    planet_types = card.ability.get("planet_types_used", 0)
    if planet_types > 0:
        return card.ability.get("extra", 1) * planet_types
    return 0


@register_dollars("j_delayed_grat")
def _delayed_grat_dollars(card: Card, game: GameSnapshot) -> int:
    """Delayed Gratification: +$2 per discard remaining IF none used.

    Source: card.lua:1675. Checks discards_used == 0 AND discards_left > 0.
    """
    if game.discards_used == 0 and game.discards_left > 0:
        return game.discards_left * card.ability.get("extra", 2)
    return 0


# ---------------------------------------------------------------------------
# Joker handlers — end-of-round mutations (calculate_joker end_of_round)
# Source: card.lua:2896-3010
# ---------------------------------------------------------------------------


@register("j_rocket")
def _rocket(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Rocket: +$2 to dollar payout after each boss blind. Source: card.lua:2896."""
    if ctx.end_of_round and not ctx.blueprint:
        if ctx.blind and getattr(ctx.blind, "boss", False):
            extra = card.ability.get("extra", {})
            extra["dollars"] = extra.get("dollars", 1) + extra.get("increase", 2)
            card.ability["extra"] = extra
            return JokerResult()
    return None


@register("j_egg")
def _egg(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Egg: +$3 to sell value per round. Source: card.lua:2940."""
    if ctx.end_of_round and not ctx.blueprint:
        inc = card.ability.get("extra", 3)
        card.ability["extra_value"] = card.ability.get("extra_value", 0) + inc
        card.sell_cost = card.sell_cost + inc
        return JokerResult()
    return None


@register("j_gift")
def _gift_card(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Gift Card: +$1 sell value to ALL jokers and consumables.

    Source: card.lua:2920. Iterates all jokers (and consumables).
    """
    if ctx.end_of_round and not ctx.blueprint and ctx.jokers:
        increment = card.ability.get("extra", 1)
        for j in ctx.jokers:
            j.ability["extra_value"] = j.ability.get("extra_value", 0) + increment
            j.sell_cost = j.sell_cost + increment
        return JokerResult()
    return None


@register("j_invisible")
def _invisible(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Invisible Joker: count rounds. When threshold reached + sold, duplicate.

    Source: card.lua:2909. Counter: ability.invis_rounds.
    """
    if ctx.end_of_round and not ctx.blueprint:
        card.ability["invis_rounds"] = card.ability.get("invis_rounds", 0) + 1
        return JokerResult()
    if ctx.selling_self and not ctx.blueprint:
        threshold = card.ability.get("extra", 2)
        if card.ability.get("invis_rounds", 0) >= threshold:
            return JokerResult(extra={"duplicate_random_joker": True})
    return None


@register("j_diet_cola")
def _diet_cola(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Diet Cola: on sell, create Double Tag. Source: card.lua:2361."""
    if ctx.selling_self:
        return JokerResult(extra={"create": {"type": "Tag", "key": "tag_double"}})
    return None


@register("j_space")
def _space_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Space Joker: 1/4 chance to level up played hand type.

    Source: card.lua:3420. Fires in before context. Seed: 'space'.
    """
    if ctx.before:
        odds = card.ability.get("extra", 4)
        if ctx.rng is not None:
            if ctx.rng.random("space") < ctx.game.probabilities_normal / odds:
                return JokerResult(level_up=True)
    return None


@register("j_to_the_moon")
def _to_the_moon(card: Card, ctx: JokerContext) -> JokerResult | None:
    """To the Moon: +$1 extra interest per $5 held.

    Source: card.lua:613. Modifies interest_amount (side effect descriptor).
    Not a calculate_joker effect — fires on add_to_deck. Stubbed here for
    completeness; actual interest modification handled by state machine.
    """
    return None


@register("j_golden")
def _golden_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Golden Joker: scoring stub — dollars handled by calc_dollar_bonus."""
    return None


# Dollar-only jokers — no-op in calculate_joker, payout via calc_dollar_bonus
@register("j_cloud_9")
def _cloud_9(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Cloud 9: dollars handled by calc_dollar_bonus."""
    return None


@register("j_delayed_grat")
def _delayed_grat(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Delayed Gratification: dollars handled by calc_dollar_bonus."""
    return None


@register("j_satellite")
def _satellite(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Satellite: dollars handled by calc_dollar_bonus."""
    return None


# ---------------------------------------------------------------------------
# Joker handlers — active effects for remaining jokers
# ---------------------------------------------------------------------------


@register("j_castle")
def _castle(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Castle: +3 chips per card of matching suit discarded. Source: card.lua:2857.

    Accumulates on ability.extra.chips. Suit target: current_round.castle_card.suit.
    Fires in discard context.
    """
    if ctx.discard and ctx.other_card is not None and not ctx.blueprint:
        # Suit lives on G.GAME.current_round.castle_card (card.lua:2857),
        # not on the joker — read it from the game snapshot.
        castle_suit = (ctx.game.castle_card_suit if ctx.game else None) or card.ability.get(
            "castle_card_suit"
        )
        if castle_suit and ctx.other_card.is_suit(castle_suit, smeared=ctx.smeared):
            extra = card.ability.get("extra", {})
            extra["chips"] = extra.get("chips", 0) + extra.get("chip_mod", 3)
            card.ability["extra"] = extra
            return JokerResult()
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 0)
        if chips > 0:
            return JokerResult(chip_mod=chips)
    return None


@register("j_runner")
def _runner(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Runner: +15 chips per Straight played. Source: card.lua:3435.

    Accumulates on ability.extra.chips in before context.
    """
    if ctx.before and not ctx.blueprint and ctx.poker_hands:
        if ctx.poker_hands.get("Straight"):
            extra = card.ability.get("extra", {})
            extra["chips"] = extra.get("chips", 0) + extra.get("chip_mod", 15)
            card.ability["extra"] = extra
            return JokerResult()
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        chips = extra.get("chips", 0)
        if chips > 0:
            return JokerResult(chip_mod=chips)
    return None


@register("j_ramen")
def _ramen(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ramen: x2 mult, loses 0.01 per card discarded. Self-destructs at x1.

    Source: card.lua:2757 (discard), 3653 (scoring via generic x_mult).
    """
    if ctx.discard and not ctx.blueprint:
        x = card.ability.get("x_mult", 2)
        loss = card.ability.get("extra", 0.01)
        if x - loss <= 1:
            return JokerResult(remove=True)
        card.ability["x_mult"] = x - loss
        return JokerResult()
    if ctx.joker_main:
        x = card.ability.get("x_mult", 2)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_mr_bones")
def _mr_bones(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mr. Bones: prevents death if score ≥ 25% of blind. Source: card.lua:3047.

    Returns saved=True and remove=True (self-destructs after saving).
    """
    if getattr(ctx, "game_over", False):
        return JokerResult(saved=True, remove=True)
    return None


@register("j_burnt")
def _burnt(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Burnt Joker: level up discard hand on first discard of round.

    Source: card.lua:2749. Fires in discard context when discards_used <= 0.
    """
    if ctx.discard and not ctx.blueprint:
        if ctx.game.discards_used <= 0 and ctx.other_card is not None:
            if ctx.full_hand and ctx.other_card is ctx.full_hand[-1]:
                return JokerResult(level_up=True)
    return None


@register("j_turtle_bean")
def _turtle_bean(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Turtle Bean: +5 hand size, -1 per round. Self-destructs at 0.

    Source: card.lua:2903. Effect is via h_size (passive). Decay in end_of_round.
    """
    if ctx.end_of_round and not ctx.blueprint:
        extra = card.ability.get("extra", {})
        h_size = extra.get("h_size", 5)
        h_mod = extra.get("h_mod", 1)
        if h_size - h_mod <= 0:
            return JokerResult(remove=True)
        extra["h_size"] = h_size - h_mod
        card.ability["extra"] = extra
        return JokerResult()
    return None


@register("j_perkeo")
def _perkeo(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Perkeo: copy random consumable when leaving shop. Source: card.lua:2413."""
    if ctx.ending_shop and not ctx.blueprint:
        return JokerResult(
            extra={
                "create": {"type": "consumable_copy", "edition": "negative", "key": "perkeo"},
            }
        )
    return None


# ---------------------------------------------------------------------------
# Passive/meta jokers — no calculate_joker effect.
# Their effects are applied via add_to_deck/remove_from_deck or checked
# inline by other systems (hand eval, shop, set_cost, etc.).
# Registered as no-op handlers for completeness.
# ---------------------------------------------------------------------------

_PASSIVE_JOKERS = {
    "j_four_fingers": "Hand eval: flushes/straights with 4 cards",
    "j_shortcut": "Hand eval: straights with gaps",
    "j_pareidolia": "Hand eval: all cards are face cards",
    "j_smeared": "Hand eval: suit pairs interchangeable",
    "j_splash": "Scoring: all played cards score",
    "j_ring_master": "Pool: allows duplicate jokers",
    "j_juggler": "Passive: +1 hand size (h_size)",
    "j_drunkard": "Passive: +1 discard (d_size)",
    "j_troubadour": "Passive: +2 hand size, -1 hand",
    "j_merry_andy": "Passive: +3 discards, -1 hand size",
    "j_oops": "Passive: doubles probabilities.normal",
    "j_credit_card": "Passive: allows -$20 debt",
    "j_chaos": "Shop: 1 free reroll per shop visit",
    "j_astronomer": "Shop: Planet cards cost $0",
}

for _key, _doc in _PASSIVE_JOKERS.items():

    def _make_passive(doc: str) -> JokerHandler:
        def _passive(card: Card, ctx: JokerContext) -> JokerResult | None:
            return None

        _passive.__doc__ = f"Passive/meta: {doc}"
        return _passive

    register(_key)(_make_passive(_doc))
