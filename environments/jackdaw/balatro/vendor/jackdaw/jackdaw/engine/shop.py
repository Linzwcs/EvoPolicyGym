"""Shop population and related helpers.

Ports:
- ``create_card_for_shop`` (``UI_definitions.lua:742``) — weighted-random type
  selection and Illusion voucher playing-card modifiers.
- ``get_pack`` (``common_events.lua:1944``) — booster pack selection with
  first-shop Buffoon guarantee.
- ``populate_shop`` — unified shop build (joker slots + voucher + boosters).
- ``buy_card`` (``button_callbacks.lua:2404``) — purchase a shop card.
- ``sell_card`` (``card.lua:1590``) — sell a joker/consumable.
- ``reroll_shop`` (``button_callbacks.lua:2855``) — reroll shop joker slots.
- ``calculate_reroll_cost`` (``common_events.lua:2263``) — current reroll cost.

Source references
-----------------
- UI_definitions.lua:742   — ``create_card_for_shop``
- common_events.lua:1944   — ``get_pack``
- common_events.lua:2082   — ``create_card`` (called after type is selected)
- common_events.lua:2263   — ``calculate_reroll_cost``
- button_callbacks.lua:2404 — ``buy_from_shop``
- button_callbacks.lua:2855 — ``reroll_shop``
- card.lua:1590             — ``Card:sell_card``
- game.lua:3099             — shop setup loop
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.card_area import CardArea
    from jackdaw.engine.rng import PseudoRandom

# ---------------------------------------------------------------------------
# Card type constants
# ---------------------------------------------------------------------------

TYPE_JOKER = "Joker"
TYPE_TAROT = "Tarot"
TYPE_PLANET = "Planet"
TYPE_SPECTRAL = "Spectral"
TYPE_PLAYING_CARD = "PlayingCard"

# Available enhancements for Illusion playing cards (ordered as in CENTER_POOLS)
_ENHANCEMENTS: list[str] = CENTER_POOLS.get("Enhanced", [])

# Seal options for Illusion playing cards
_SEALS: list[str] = ["Red", "Blue", "Gold", "Purple"]

# ---------------------------------------------------------------------------
# Card type selection — UI_definitions.lua:742
# ---------------------------------------------------------------------------


def select_shop_card_type(
    rng: PseudoRandom,
    ante: int,
    *,
    joker_rate: float = 20.0,
    tarot_rate: float = 4.0,
    planet_rate: float = 4.0,
    spectral_rate: float = 0.0,
    playing_card_rate: float = 0.0,
) -> str:
    """Select what type of card fills a shop joker slot.

    Mirrors the weighted-random branch in ``create_card_for_shop``
    (``UI_definitions.lua:742``).  The caller is responsible for passing
    the correct rates (modified by vouchers/deck/etc. before calling).

    RNG: one draw from stream ``'cdt' + str(ante)``.

    Default rates and base probabilities (total = 28):

    +---------------+------+----------+
    | Type          | Rate | Base %   |
    +===============+======+==========+
    | Joker         | 20   | ~71.4 %  |
    +---------------+------+----------+
    | Tarot         |  4   | ~14.3 %  |
    +---------------+------+----------+
    | Planet        |  4   | ~14.3 %  |
    +---------------+------+----------+
    | Spectral      |  0   | 0 %      |
    +---------------+------+----------+
    | Playing Card  |  0   | 0 %      |
    +---------------+------+----------+

    Voucher rate examples (passed in as modified parameters):

    * Tarot Merchant — ``tarot_rate = 9.6``
    * Planet Merchant — ``planet_rate = 9.6``
    * Magic Trick — ``playing_card_rate = 4``
    * Ghost Deck — sets ``spectral_rate`` at run start

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Advances
        stream ``'cdt' + str(ante)`` by one draw.
    ante:
        Current ante number.
    joker_rate, tarot_rate, planet_rate, spectral_rate, playing_card_rate:
        Weights for each card type.  Pass voucher-modified values from the
        caller.

    Returns
    -------
    str
        One of ``'Joker'``, ``'Tarot'``, ``'Planet'``, ``'Spectral'``, or
        ``'PlayingCard'``.
    """
    total = joker_rate + tarot_rate + planet_rate + spectral_rate + playing_card_rate

    poll = rng.random("cdt" + str(ante)) * total

    if poll < joker_rate:
        return TYPE_JOKER
    poll -= joker_rate

    if poll < tarot_rate:
        return TYPE_TAROT
    poll -= tarot_rate

    if poll < planet_rate:
        return TYPE_PLANET
    poll -= planet_rate

    if poll < spectral_rate:
        return TYPE_SPECTRAL

    return TYPE_PLAYING_CARD


# ---------------------------------------------------------------------------
# Illusion voucher modifiers — UI_definitions.lua:~780
# ---------------------------------------------------------------------------

# Probability thresholds
_ILLUSION_ENH_THRESHOLD = 0.4  # roll > 0.4 → enhanced (60% chance)
_ILLUSION_EDI_CHANCE_THRESHOLD = 0.8  # roll > 0.8 → get edition (20% chance)

# Edition distribution for Illusion: Foil 50%, Holo 35%, Poly 15%
# Evaluated top-down: Poly > 0.85, Holo > 0.5, else Foil
_ILLUSION_POLY_THRESHOLD = 0.85
_ILLUSION_HOLO_THRESHOLD = 0.5


def roll_illusion_modifiers(
    rng: PseudoRandom,
    ante: int,
    *,
    append: str = "",
) -> dict[str, Any]:
    """Roll Illusion voucher modifiers for a playing card drawn from the shop.

    Mirrors the Illusion-specific path in ``create_card_for_shop``
    (``UI_definitions.lua``).

    Probabilities
    ~~~~~~~~~~~~~
    * **60%** the card receives a random enhancement (one of the 8 standard
      playing-card enhancements); otherwise base card.
    * **20%** the card receives a random edition:
      Foil 50 %, Holo 35 %, Polychrome 15 %.
    * Seal is not determined here — deferred to the post-creation hook pass.

    RNG streams consumed (always, for determinism):

    1. ``'illusion_enh' + append + str(ante)`` — enhancement chance roll
    2. ``'illusion_enh_pick' + append + str(ante)`` — enhancement selection
       (seeded via :meth:`~jackdaw.engine.rng.PseudoRandom.seed` passed to
       :meth:`~jackdaw.engine.rng.PseudoRandom.element`) — consumed only
       when enhancement is granted
    3. ``'illusion_edi_chance' + append + str(ante)`` — edition chance roll
    4. ``'illusion_edi' + append + str(ante)`` — edition type roll (consumed
       only when edition is granted)

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    append:
        Optional seed-key suffix for context disambiguation.

    Returns
    -------
    dict
        Dict with zero or more of the following keys:

        * ``'enhancement'`` : str — e.g. ``'m_glass'``
        * ``'edition'`` : dict — e.g. ``{'foil': True}``
    """
    result: dict[str, Any] = {}

    suffix = append + str(ante)

    # -- Enhancement (60%) --
    enh_roll = rng.random("illusion_enh" + suffix)
    if enh_roll > _ILLUSION_ENH_THRESHOLD:
        enh_seed = rng.seed("illusion_enh_pick" + suffix)
        enhancement, _ = rng.element(_ENHANCEMENTS, enh_seed)
        result["enhancement"] = enhancement

    # -- Edition (20%) --
    edi_chance = rng.random("illusion_edi_chance" + suffix)
    if edi_chance > _ILLUSION_EDI_CHANCE_THRESHOLD:
        edi_roll = rng.random("illusion_edi" + suffix)
        if edi_roll > _ILLUSION_POLY_THRESHOLD:
            result["edition"] = {"polychrome": True}
        elif edi_roll > _ILLUSION_HOLO_THRESHOLD:
            result["edition"] = {"holo": True}
        else:
            result["edition"] = {"foil": True}

    return result


# ---------------------------------------------------------------------------
# Booster pool — module-level cache
# ---------------------------------------------------------------------------

_BOOSTER_POOL: list[str] = CENTER_POOLS.get("Booster", [])

# First-shop Buffoon guarantee: Lua uses math.random(1, 2) which is
# non-deterministic; we deterministically return _1 as the fallback.
_FIRST_SHOP_BUFFOON_PACK = "p_buffoon_normal_1"
_FIRST_SHOP_BUFFOON_KEY = "first_shop_buffoon"


# ---------------------------------------------------------------------------
# get_pack — common_events.lua:1944
# ---------------------------------------------------------------------------


def get_pack(
    rng: PseudoRandom,
    ante: int,
    key: str = "shop_pack",
    *,
    first_shop: bool = False,
    banned_keys: set[str] | None = None,
) -> str:
    """Select a booster pack type.

    Mirrors ``get_pack`` (``common_events.lua:1944``).

    First-shop guarantee
    ~~~~~~~~~~~~~~~~~~~~
    When *first_shop* is ``True`` and ``'p_buffoon_normal_1'`` is not in
    *banned_keys*, the function returns ``'p_buffoon_normal_1'`` immediately
    without consuming an RNG draw.  The Lua source picks variant 1 or 2 via
    the non-deterministic ``math.random(1, 2)``; we always return variant 1
    as the deterministic equivalent.

    The caller is responsible for tracking when the guarantee has been
    consumed (see :func:`populate_shop`).

    Normal selection
    ~~~~~~~~~~~~~~~~
    Compute cumulative weight of all non-banned packs, draw
    ``rng.random(key + str(ante)) * total_weight``, walk the pool.

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Advances
        stream ``key + str(ante)`` by one draw (skipped when the first-shop
        guarantee fires).
    ante:
        Current ante number.
    key:
        RNG stream base key.  ``'shop_pack'`` for standard shop usage.
    first_shop:
        When ``True``, apply the first-shop Buffoon guarantee (if not
        banned).  Caller must pass ``True`` only for the very first booster
        slot of a run.
    banned_keys:
        Set of banned center keys.  Banned packs are excluded from both the
        guarantee and the weighted draw.

    Returns
    -------
    str
        A key from ``CENTER_POOLS['Booster']``.  Falls back to the last
        eligible pack if the RNG lands exactly on the cumulative total.
    """
    banned: set[str] = banned_keys or set()

    # -- First-shop Buffoon guarantee (game.lua / common_events.lua:1945) --
    if first_shop and _FIRST_SHOP_BUFFOON_PACK not in banned:
        return _FIRST_SHOP_BUFFOON_PACK

    # -- Weighted random walk --
    cume = sum(BOOSTERS[k].weight for k in _BOOSTER_POOL if k not in banned)

    poll = rng.random(key + str(ante)) * cume

    it = 0.0
    for k in _BOOSTER_POOL:
        if k in banned:
            continue
        w = BOOSTERS[k].weight
        it += w
        if it >= poll and it - w <= poll:
            return k

    # Floating-point edge: poll == cume exactly → last eligible pack
    for k in reversed(_BOOSTER_POOL):
        if k not in banned:
            return k
    return _BOOSTER_POOL[-1]  # unreachable if pool is non-empty


# ---------------------------------------------------------------------------
# populate_shop — game.lua:3099 / UI_definitions.lua:742
# ---------------------------------------------------------------------------

# key_append used by create_card when called from the shop joker area
_SHOP_APPEND = "sho"


def populate_shop(
    rng: PseudoRandom,
    ante: int,
    game_state: dict,
) -> dict[str, list[Card] | Card | None]:
    """Build the full shop for the current round.

    Mirrors the shop-setup block in ``game.lua:3099`` and
    ``create_card_for_shop`` (``UI_definitions.lua:742``).

    Flow
    ~~~~
    1. **Joker slots** — for each of ``shop['joker_max']`` (default 2) slots:
       call :func:`select_shop_card_type` then
       :func:`~jackdaw.engine.card_factory.create_card` with
       ``area='shop'`` and ``append='sho'``.
    2. **Voucher** — create a voucher card from
       ``game_state['current_round']['voucher']`` (key pre-determined by
       :func:`~jackdaw.engine.vouchers.get_next_voucher_key`).  ``None`` if
       absent.
    3. **Boosters** — 2 packs via :func:`get_pack` with key
       ``'shop_pack'``.

    .. note::
       Tag hooks (``store_joker_create``, ``store_joker_modify``,
       ``voucher_add``, ``shop_final_pass``) are **not** applied here;
       they are deferred to the M11 tag system.

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    game_state:
        Game-state dict.  Relevant keys:

        * ``shop`` → ``joker_max`` (int, default 2)
        * ``joker_rate``, ``tarot_rate``, ``planet_rate``,
          ``spectral_rate``, ``playing_card_rate`` — type-selection weights
        * ``current_round`` → ``voucher`` (str | None)
        * ``banned_keys`` (dict) — passed to :func:`get_pack`
        * All keys forwarded to
          :func:`~jackdaw.engine.card_factory.create_card` via
          *game_state*

    Returns
    -------
    dict
        ``{'jokers': list[Card], 'voucher': Card | None,
        'boosters': list[Card]}``
    """
    from jackdaw.engine.card import Card as _Card
    from jackdaw.engine.card_factory import create_card, create_voucher

    gs = game_state

    shop_joker_max: int = gs.get("shop", {}).get("joker_max", 2)

    joker_rate: float = gs.get("joker_rate", 20.0)
    tarot_rate: float = gs.get("tarot_rate", 4.0)
    planet_rate: float = gs.get("planet_rate", 4.0)
    spectral_rate: float = gs.get("spectral_rate", 0.0)
    playing_card_rate: float = gs.get("playing_card_rate", 0.0)

    banned_keys: set[str] = set(gs.get("banned_keys") or {})

    # -- 1. Joker slots --
    jokers: list[_Card] = []
    for _ in range(shop_joker_max):
        card_type = select_shop_card_type(
            rng,
            ante,
            joker_rate=joker_rate,
            tarot_rate=tarot_rate,
            planet_rate=planet_rate,
            spectral_rate=spectral_rate,
            playing_card_rate=playing_card_rate,
        )
        card = create_card(
            card_type,
            rng,
            ante,
            area="shop",
            append=_SHOP_APPEND,
            game_state=gs,
        )
        jokers.append(card)

    # -- 2. Voucher --
    voucher: _Card | None = None
    voucher_key: str | None = gs.get("current_round", {}).get("voucher")
    if voucher_key:
        voucher = create_voucher(voucher_key)
        voucher.set_cost(
            inflation=gs.get("inflation", 0),
            discount_percent=gs.get("discount_percent", 0),
            ante=ante,
        )

    # -- 3. Boosters (always exactly 2 slots) --
    boosters: list[_Card] = []
    for i in range(2):
        first_shop = i == 0 and not gs.get(_FIRST_SHOP_BUFFOON_KEY, False)
        pack_key = get_pack(rng, ante, "shop_pack", first_shop=first_shop, banned_keys=banned_keys)
        # Mark guarantee consumed when it fires (pack returned and not banned)
        if first_shop and _FIRST_SHOP_BUFFOON_PACK not in banned_keys:
            gs[_FIRST_SHOP_BUFFOON_KEY] = True
        pack_card = _Card()
        pack_card.set_ability(pack_key)
        pack_card.set_cost(
            inflation=gs.get("inflation", 0),
            discount_percent=gs.get("discount_percent", 0),
            ante=ante,
            booster_ante_scaling=gs.get("booster_ante_scaling", False),
            has_astronomer=gs.get("has_astronomer", False),
        )
        boosters.append(pack_card)

    return {"jokers": jokers, "voucher": voucher, "boosters": boosters}


# ---------------------------------------------------------------------------
# calculate_reroll_cost — common_events.lua:2263
# ---------------------------------------------------------------------------

# Default reroll cost at round start (game.lua:1958)
_DEFAULT_BASE_REROLL_COST = 5


def calculate_reroll_cost(game_state: dict) -> int:
    """Return the current reroll cost and update *game_state* in-place.

    Mirrors ``calculate_reroll_cost`` (``common_events.lua:2263``).

    Priority
    ~~~~~~~~
    1. If ``current_round.free_rerolls > 0`` → cost is **0** (free reroll
       consumed elsewhere; this function only reads the count).
    2. Otherwise: ``base_reroll_cost + current_round.reroll_cost_increase``

    The base cost comes from ``round_resets.reroll_cost`` (or
    ``round_resets.temp_reroll_cost`` if set), which defaults to **5**.

    This function does **not** increment ``reroll_cost_increase`` — that is
    the caller's responsibility (done inside :func:`reroll_shop`).

    Parameters
    ----------
    game_state:
        Must contain ``current_round`` sub-dict; may also contain
        ``round_resets`` sub-dict.

    Returns
    -------
    int
        The cost in dollars for the next reroll.
    """
    cr = game_state.setdefault("current_round", {})
    rr = game_state.get("round_resets", {})

    # Clamp free_rerolls
    free = max(0, cr.get("free_rerolls", 0))
    cr["free_rerolls"] = free

    if free > 0:
        cr["reroll_cost"] = 0
        return 0

    increase = cr.get("reroll_cost_increase", 0)
    base = rr.get("temp_reroll_cost") or rr.get("reroll_cost", _DEFAULT_BASE_REROLL_COST)
    cost = base + increase
    cr["reroll_cost"] = cost
    return cost


# ---------------------------------------------------------------------------
# buy_card — button_callbacks.lua:2404
# ---------------------------------------------------------------------------

# Playing-card ability sets (cards that go into the deck, not the joker area)
_PLAYING_CARD_SETS = frozenset({"Default", "Enhanced"})


def buy_card(
    card: Card,
    from_area: CardArea,
    to_area: CardArea,
    game_state: dict,
) -> dict[str, Any]:
    """Execute a shop purchase.

    Mirrors ``G.FUNCS.buy_from_shop`` (``button_callbacks.lua:2404``) and
    ``check_for_buy_space`` (``button_callbacks.lua:2393``).

    Flow
    ~~~~
    1. **Space check** — ``to_area.has_space(negative_bonus)`` where
       *negative_bonus* is 1 if the card has a Negative edition.  Returns
       ``{'ok': False, 'reason': 'no_space'}`` if full.
    2. **Funds check** — ``game_state['dollars'] >= card.cost``.  Returns
       ``{'ok': False, 'reason': 'insufficient_funds'}`` if short.
    3. **Remove** card from *from_area*.
    4. **Passive effects** — ``card.add_to_deck(game_state)``.
    5. **Place** card in *to_area*.
    6. **Playing-card bookkeeping** — if the card is a Default/Enhanced
       playing card, append to ``game_state['playing_cards']`` and notify
       all jokers in ``game_state['jokers']`` with
       ``calculate_joker({playing_card_added=True, cards=[card]})``.
    7. **Deduct cost** — ``game_state['dollars'] -= card.cost``.
    8. **Inflation** — if ``game_state.get('inflation_modifier')`` is True,
       increment ``game_state['inflation']`` and call
       ``card.set_cost(inflation=…)`` on every card in
       ``game_state.get('all_shop_cards', [])``.
    9. **Track** — ``game_state['cards_purchased'] += 1`` and, for Jokers,
       ``game_state['used_jokers'][card.center_key] = True``.

    Parameters
    ----------
    card:
        The card being purchased (still in *from_area* at call time).
    from_area:
        The shop area the card is being removed from.
    to_area:
        Destination area (``G.jokers``, ``G.consumeables``, ``G.deck``).
    game_state:
        Mutable game-state dict.  Relevant keys:

        * ``dollars`` (int) — current money.
        * ``inflation`` (int) — cumulative inflation count.
        * ``inflation_modifier`` (bool) — whether inflation is active.
        * ``discount_percent`` (int) — 0 / 25 / 50.
        * ``cards_purchased`` (int) — running tally this round.
        * ``used_jokers`` (dict) — tracks which joker keys have been seen.
        * ``playing_cards`` (list) — all playing cards in run.
        * ``jokers`` (list[Card]) — active jokers (for notifications).
        * ``all_shop_cards`` (list[Card]) — cards to recalculate on
          inflation (optional).

    Returns
    -------
    dict
        ``{'ok': True}`` on success, or ``{'ok': False, 'reason': str}``
        on failure.
    """
    from jackdaw.engine.jokers import JokerContext, calculate_joker

    # -- 1. Space check --
    negative_bonus = 1 if (card.edition and card.edition.get("negative")) else 0
    if not to_area.has_space(negative_bonus):
        return {"ok": False, "reason": "no_space"}

    # -- 2. Funds check --
    if game_state.get("dollars", 0) < card.cost:
        return {"ok": False, "reason": "insufficient_funds"}

    # -- 3. Remove from shop --
    from_area.remove(card)

    # -- 4. Passive add_to_deck effects --
    card.add_to_deck(game_state)

    # -- 5. Place in destination --
    to_area.add(card)

    # -- 6. Playing-card bookkeeping --
    if card.ability.get("set") in _PLAYING_CARD_SETS:
        playing_cards: list[Card] = game_state.setdefault("playing_cards", [])
        playing_cards.append(card)
        for joker in game_state.get("jokers", []):
            ctx = JokerContext(playing_card_added=True, cards=[card])
            calculate_joker(joker, ctx)
    else:
        # buying_card notification for all active jokers
        for joker in game_state.get("jokers", []):
            ctx = JokerContext(buying_card=True, card=card)
            calculate_joker(joker, ctx)

    # -- 7. Deduct cost --
    game_state["dollars"] = game_state.get("dollars", 0) - card.cost

    # -- 8. Inflation --
    if game_state.get("inflation_modifier"):
        game_state["inflation"] = game_state.get("inflation", 0) + 1
        inflation = game_state["inflation"]
        discount = game_state.get("discount_percent", 0)
        ante = game_state.get("round_resets", {}).get("ante", 1)
        for shop_card in game_state.get("all_shop_cards", []):
            if hasattr(shop_card, "set_cost"):
                shop_card.set_cost(
                    inflation=inflation,
                    discount_percent=discount,
                    ante=ante,
                    booster_ante_scaling=game_state.get("booster_ante_scaling", False),
                    has_astronomer=game_state.get("has_astronomer", False),
                )

    # -- 9. Track --
    game_state["cards_purchased"] = game_state.get("cards_purchased", 0) + 1
    if card.ability.get("set") == "Joker":
        game_state.setdefault("used_jokers", {})[card.center_key] = True

    return {"ok": True}


# ---------------------------------------------------------------------------
# sell_card — card.lua:1590 + button_callbacks.lua:2318
# ---------------------------------------------------------------------------


def sell_card(
    card: Card,
    from_area: CardArea,
    game_state: dict,
) -> dict[str, Any]:
    """Execute a card sale.

    Mirrors ``Card:sell_card`` (``card.lua:1590``) and
    ``G.FUNCS.sell_card`` (``button_callbacks.lua:2318``).

    Flow
    ~~~~
    1. **Eligibility check** — eternal cards and cards not in a joker/
       consumable area cannot be sold.  Returns
       ``{'ok': False, 'reason': 'eternal'}`` or ``'not_sellable'``.
    2. **Selling-self notification** — call
       ``calculate_joker(card, {selling_self=True})``.
    3. **Selling-card notification** — call
       ``calculate_joker(j, {selling_card=True, card=card})`` for every
       other joker in ``game_state['jokers']``.
    4. **Reverse passive effects** — ``card.remove_from_deck(game_state)``.
    5. **Award sell value** — ``game_state['dollars'] += card.sell_cost``.
    6. **Remove** card from *from_area*.

    Parameters
    ----------
    card:
        The card being sold (still in *from_area* at call time).
    from_area:
        The area the card currently occupies.
    game_state:
        Mutable game-state dict.  Relevant keys:

        * ``dollars`` (int).
        * ``jokers`` (list[Card]) — active jokers for sell notifications.

    Returns
    -------
    dict
        ``{'ok': True, 'dollars_gained': int}`` on success, or
        ``{'ok': False, 'reason': str}`` on failure.
    """
    from jackdaw.engine.jokers import JokerContext, calculate_joker

    # -- 1. Eligibility --
    if card.eternal:
        return {"ok": False, "reason": "eternal"}
    if from_area.type not in ("joker", "consumeable"):
        return {"ok": False, "reason": "not_sellable"}

    # -- 2. Selling-self notification --
    calculate_joker(card, JokerContext(selling_self=True))

    # -- 3. Selling-card notification to other jokers --
    for joker in game_state.get("jokers", []):
        if joker is not card:
            calculate_joker(joker, JokerContext(selling_card=True, card=card))

    # -- 4. Reverse passive effects --
    card.remove_from_deck(game_state)

    # -- 5. Award money --
    dollars_gained = card.sell_cost
    game_state["dollars"] = game_state.get("dollars", 0) + dollars_gained

    # -- 6. Remove from area --
    from_area.remove(card)

    return {"ok": True, "dollars_gained": dollars_gained}


# ---------------------------------------------------------------------------
# reroll_shop — button_callbacks.lua:2855
# ---------------------------------------------------------------------------


def reroll_shop(
    shop_jokers: CardArea,
    rng: PseudoRandom,
    ante: int,
    game_state: dict,
) -> dict[str, Any]:
    """Reroll the shop's joker/consumable slots.

    Mirrors ``G.FUNCS.reroll_shop`` (``button_callbacks.lua:2855``).

    Flow
    ~~~~
    1. **Cost** — ``calculate_reroll_cost(game_state)`` (reads current cost
       without incrementing yet).  If the player has insufficient funds,
       return ``{'ok': False, 'reason': 'insufficient_funds'}``.
    2. **Decrement free_rerolls** — ``current_round.free_rerolls -= 1``
       (clamped to 0), recorded as *was_free*.
    3. **Deduct cost** — if cost > 0, ``game_state['dollars'] -= cost``.
    4. **Increment reroll_cost_increase** — if not *was_free*, increment
       ``current_round.reroll_cost_increase`` by 1 then recalculate.
    5. **Clear shop** — remove all cards from *shop_jokers*.
    6. **Repopulate** — fill slots up to ``shop['joker_max']`` (default 2)
       via :func:`~jackdaw.engine.shop.populate_shop` logic (calls
       :func:`select_shop_card_type` then
       :func:`~jackdaw.engine.card_factory.create_card`).
    7. **Notify jokers** — ``calculate_joker(j, {reroll_shop=True})`` for
       all jokers in ``game_state['jokers']``.

    Parameters
    ----------
    shop_jokers:
        The shop joker area to repopulate.
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    game_state:
        Mutable game-state dict.  Relevant keys:

        * ``dollars`` (int).
        * ``current_round`` → ``free_rerolls``, ``reroll_cost``,
          ``reroll_cost_increase``.
        * ``round_resets`` → ``reroll_cost`` (base cost, default 5).
        * ``shop`` → ``joker_max`` (default 2).
        * All keys passed to :func:`select_shop_card_type` and
          :func:`~jackdaw.engine.card_factory.create_card`.
        * ``jokers`` (list[Card]) — for reroll_shop notifications.

    Returns
    -------
    dict
        ``{'ok': True, 'cost': int, 'was_free': bool,
        'new_cards': list[Card]}`` on success, or
        ``{'ok': False, 'reason': str}`` on failure.
    """
    from jackdaw.engine.card_factory import create_card
    from jackdaw.engine.jokers import JokerContext, calculate_joker

    cr = game_state.setdefault("current_round", {})

    # -- 1. Cost --
    cost = calculate_reroll_cost(game_state)
    if game_state.get("dollars", 0) < cost:
        return {"ok": False, "reason": "insufficient_funds"}

    # -- 2. Decrement free_rerolls --
    was_free = cr.get("free_rerolls", 0) > 0
    cr["free_rerolls"] = max(0, cr.get("free_rerolls", 0) - 1)

    # -- 3. Deduct --
    if cost > 0:
        game_state["dollars"] = game_state.get("dollars", 0) - cost

    # -- 4. Increment reroll_cost_increase and recalculate --
    if not was_free:
        cr["reroll_cost_increase"] = cr.get("reroll_cost_increase", 0) + 1
    calculate_reroll_cost(game_state)

    # -- 5. Clear shop --
    shop_jokers.cards.clear()

    # -- 6. Repopulate --
    shop_joker_max: int = game_state.get("shop", {}).get("joker_max", 2)
    joker_rate: float = game_state.get("joker_rate", 20.0)
    tarot_rate: float = game_state.get("tarot_rate", 4.0)
    planet_rate: float = game_state.get("planet_rate", 4.0)
    spectral_rate: float = game_state.get("spectral_rate", 0.0)
    playing_card_rate: float = game_state.get("playing_card_rate", 0.0)

    new_cards: list[Card] = []
    slots_needed = shop_joker_max - len(shop_jokers.cards)
    for _ in range(slots_needed):
        card_type = select_shop_card_type(
            rng,
            ante,
            joker_rate=joker_rate,
            tarot_rate=tarot_rate,
            planet_rate=planet_rate,
            spectral_rate=spectral_rate,
            playing_card_rate=playing_card_rate,
        )
        new_card = create_card(
            card_type,
            rng,
            ante,
            area="shop",
            append=_SHOP_APPEND,
            game_state=game_state,
        )
        shop_jokers.add(new_card)
        new_cards.append(new_card)

    # -- 7. Notify active jokers --
    for joker in game_state.get("jokers", []):
        calculate_joker(joker, JokerContext(reroll_shop=True))

    return {"ok": True, "cost": cost, "was_free": was_free, "new_cards": new_cards}
