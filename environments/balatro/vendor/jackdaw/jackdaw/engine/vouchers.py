"""Voucher effect system.

Ports ``Card:apply_to_run`` from ``card.lua:1880`` as a pure function that
mutates game_state and returns a mutations dict.  Also provides prerequisite
checking and pool selection matching ``get_next_voucher_key`` from
``common_events.lua:1901``.

Each voucher's effect is keyed by the voucher **name** (as in the Lua
center_table dispatch) but the public API is keyed by center **key**
(``v_overstock_norm``, etc.).

Source references
-----------------
- card.lua:1880  — ``Card:apply_to_run``
- common_events.lua:1901 — ``get_next_voucher_key``
- common_events.lua:1956 — ``get_current_pool`` (Voucher culling)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jackdaw.engine.data.prototypes import CENTER_POOLS, VOUCHERS

if TYPE_CHECKING:
    from jackdaw.engine.rng import PseudoRandom


# ---------------------------------------------------------------------------
# Prerequisite checking
# ---------------------------------------------------------------------------


def check_voucher_prerequisites(key: str, used_vouchers: dict[str, bool]) -> bool:
    """Return True if all prerequisites for *key* are satisfied.

    Mirrors the ``v.requires`` check inside ``get_current_pool``
    (common_events.lua:1989).
    """
    proto = VOUCHERS.get(key)
    if proto is None:
        return False
    return all(used_vouchers.get(req) for req in proto.requires)


# ---------------------------------------------------------------------------
# Pool building — mirrors get_current_pool for Vouchers
# ---------------------------------------------------------------------------


def get_available_voucher_pool(
    used_vouchers: dict[str, bool],
    in_shop: list[str] | None = None,
) -> list[str]:
    """Return the sorted list of eligible voucher keys for the current state.

    Mirrors the culling loop inside ``get_current_pool`` for set='Voucher'
    (common_events.lua:1985):
    - Excludes already-used vouchers.
    - Excludes vouchers whose ``requires`` list is not fully satisfied.
    - Excludes vouchers currently in the shop (``in_shop``).

    Order preserved from ``CENTER_POOLS['Voucher']`` (sorted by ``order``
    field), matching the Lua source-pool ordering.
    """
    in_shop_set = set(in_shop or [])
    return [
        key
        for key in CENTER_POOLS["Voucher"]
        if not used_vouchers.get(key)
        and key not in in_shop_set
        and check_voucher_prerequisites(key, used_vouchers)
    ]


# ---------------------------------------------------------------------------
# get_next_voucher_key — mirrors common_events.lua:1901
# ---------------------------------------------------------------------------


def get_next_voucher_key(
    rng: PseudoRandom,
    used_vouchers: dict[str, bool],
    in_shop: list[str] | None = None,
    *,
    from_tag: bool = False,
    ante: int = 1,
    banned_keys: set[str] | None = None,
) -> str | None:
    """Select the next voucher key for the shop using the RNG stream.

    Mirrors ``get_next_voucher_key`` (common_events.lua:1901):

    1. Build pool via ``get_current_pool('Voucher')`` which preserves
       ``'UNAVAILABLE'`` sentinels at ineligible positions (matching Lua's
       array-with-holes approach so ``pseudorandom_element`` picks from a
       fixed-size array).
    2. Pick via ``pseudorandom_element(pool, pseudoseed(pool_key))``.
    3. Resample with ``pool_key + '_resample' + str(it)`` if the draw
       lands on an UNAVAILABLE slot.

    The seed key for the normal path is ``'Voucher' + str(ante)`` (e.g.
    ``'Voucher1'`` at ante 1), matching ``get_current_pool``'s return at
    ``common_events.lua:2052``.  For tag-triggered vouchers, the key is
    ``'Voucher_fromtag'`` (no ante appended).

    ``banned_keys`` is applied while building the pool, before its RNG-backed
    selection. This is the same unavailable-sentinel behavior used by
    challenge restrictions elsewhere in the engine.

    Returns ``None`` if the eligible pool is empty (shouldn't happen in
    practice due to the ``v_blank`` fallback in ``get_current_pool``).
    """
    from jackdaw.engine.pools import UNAVAILABLE, get_current_pool

    shop_vouchers_set = set(in_shop) if in_shop else set()
    used_vouchers_set = {k for k, v in used_vouchers.items() if v}

    pool, pool_key = get_current_pool(
        "Voucher",
        rng,
        ante,
        used_vouchers=used_vouchers_set,
        shop_vouchers=shop_vouchers_set,
        banned_keys=banned_keys,
    )

    if not pool:
        return None

    # Lua: if _from_tag then _pool_key = 'Voucher_fromtag' end
    # Otherwise pool_key from get_current_pool is 'Voucher'; Lua appends
    # ante at common_events.lua:2052 → 'Voucher1'.
    if from_tag:
        full_key = "Voucher_fromtag"
    else:
        full_key = pool_key + str(ante)

    seed = rng.seed(full_key)
    result, _ = rng.element(pool, seed)

    # Resample loop matching the Lua while-loop (common_events.lua:1906-1909).
    it = 1
    while result == UNAVAILABLE:
        it += 1
        seed = rng.seed(full_key + "_resample" + str(it))
        result, _ = rng.element(pool, seed)

    return result


# ---------------------------------------------------------------------------
# apply_voucher — mirrors Card:apply_to_run (card.lua:1880)
# ---------------------------------------------------------------------------


def apply_voucher(key: str, game_state: dict[str, Any]) -> dict[str, Any]:
    """Apply a voucher's permanent effect to *game_state* in-place.

    Returns a ``{field_path: new_value}`` mutations dict describing what
    changed (dot-notation for nested keys, e.g. ``'round_resets.hands'``).

    Mirrors ``Card:apply_to_run`` (card.lua:1880).  Effects are keyed by
    voucher **name** (matching the Lua ``center_table.name`` dispatch).
    """
    proto = VOUCHERS.get(key)
    if proto is None:
        return {}

    extra: Any = proto.config.get("extra") if isinstance(proto.config, dict) else None
    name = proto.name
    mutations: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Shop size — change_shop_size(1)
    # card.lua:1886
    # -----------------------------------------------------------------------
    if name in ("Overstock", "Overstock Plus"):
        shop = game_state.setdefault("shop", {})
        shop["joker_max"] = shop.get("joker_max", 2) + 1
        mutations["shop.joker_max"] = shop["joker_max"]

    # -----------------------------------------------------------------------
    # Tarot rate — G.GAME.tarot_rate = 4 * extra
    # card.lua:1891  (Merchant extra=2.4 → 9.6; Tycoon extra=8 → 32)
    # -----------------------------------------------------------------------
    elif name in ("Tarot Merchant", "Tarot Tycoon"):
        val = 4 * extra
        game_state["tarot_rate"] = val
        mutations["tarot_rate"] = val

    # -----------------------------------------------------------------------
    # Planet rate — G.GAME.planet_rate = 4 * extra
    # card.lua:1896  (Merchant extra=2.4 → 9.6; Tycoon extra=8 → 32)
    # -----------------------------------------------------------------------
    elif name in ("Planet Merchant", "Planet Tycoon"):
        val = 4 * extra
        game_state["planet_rate"] = val
        mutations["planet_rate"] = val

    # -----------------------------------------------------------------------
    # Edition rate — G.GAME.edition_rate = extra
    # card.lua:1901  (Hone=2, Glow Up=4)
    # -----------------------------------------------------------------------
    elif name in ("Hone", "Glow Up"):
        game_state["edition_rate"] = extra
        mutations["edition_rate"] = extra

    # -----------------------------------------------------------------------
    # Playing card rate — G.GAME.playing_card_rate = extra
    # card.lua:1906  (Magic Trick=4, Illusion=4)
    # -----------------------------------------------------------------------
    elif name in ("Magic Trick", "Illusion"):
        game_state["playing_card_rate"] = extra
        mutations["playing_card_rate"] = extra

    # -----------------------------------------------------------------------
    # Consumable slots — consumeables.config.card_limit += 1
    # card.lua:1910
    # -----------------------------------------------------------------------
    elif name == "Crystal Ball":
        game_state["consumable_slots"] = game_state.get("consumable_slots", 2) + 1
        mutations["consumable_slots"] = game_state["consumable_slots"]

    # -----------------------------------------------------------------------
    # Omen Globe — spectral cards appear in standard packs (passive flag)
    # No explicit apply_to_run body; handled as a run-flag.
    # -----------------------------------------------------------------------
    elif name == "Omen Globe":
        game_state["omen_globe"] = True
        mutations["omen_globe"] = True

    # -----------------------------------------------------------------------
    # Discount — G.GAME.discount_percent = extra
    # card.lua:1914  (Clearance Sale=25, Liquidation=50)
    # -----------------------------------------------------------------------
    elif name in ("Clearance Sale", "Liquidation"):
        game_state["discount_percent"] = extra
        mutations["discount_percent"] = extra

    # -----------------------------------------------------------------------
    # Reroll cost — round_resets.reroll_cost -= extra
    # card.lua:1921  (Reroll Surplus extra=2, Reroll Glut extra=2)
    # -----------------------------------------------------------------------
    elif name in ("Reroll Surplus", "Reroll Glut"):
        rr = game_state.setdefault("round_resets", {})
        rr["reroll_cost"] = rr.get("reroll_cost", 5) - extra
        cr = game_state.setdefault("current_round", {})
        cr["reroll_cost"] = max(0, cr.get("reroll_cost", rr["reroll_cost"] + extra) - extra)
        mutations["round_resets.reroll_cost"] = rr["reroll_cost"]
        mutations["current_round.reroll_cost"] = cr["reroll_cost"]

    # -----------------------------------------------------------------------
    # Interest cap — G.GAME.interest_cap = extra
    # card.lua:1927  (Seed Money=50, Money Tree=100)
    # -----------------------------------------------------------------------
    elif name in ("Seed Money", "Money Tree"):
        game_state["interest_cap"] = extra
        mutations["interest_cap"] = extra

    # -----------------------------------------------------------------------
    # Hands per round — round_resets.hands += extra
    # card.lua:1931  (Grabber=1, Nacho Tong=1)
    # -----------------------------------------------------------------------
    elif name in ("Grabber", "Nacho Tong"):
        rr = game_state.setdefault("round_resets", {})
        rr["hands"] = rr.get("hands", 4) + extra
        mutations["round_resets.hands"] = rr["hands"]

    # -----------------------------------------------------------------------
    # Hand size — G.hand:change_size(1)
    # card.lua:1934  (Paint Brush=+1, Palette=+1)
    # -----------------------------------------------------------------------
    elif name in ("Paint Brush", "Palette"):
        game_state["hand_size"] = game_state.get("hand_size", 8) + 1
        mutations["hand_size"] = game_state["hand_size"]

    # -----------------------------------------------------------------------
    # Discards per round — round_resets.discards += extra
    # card.lua:1937  (Wasteful=1, Recyclomancy=1)
    # -----------------------------------------------------------------------
    elif name in ("Wasteful", "Recyclomancy"):
        rr = game_state.setdefault("round_resets", {})
        rr["discards"] = rr.get("discards", 3) + extra
        mutations["round_resets.discards"] = rr["discards"]

    # -----------------------------------------------------------------------
    # Joker slots — jokers.config.card_limit += 1
    # card.lua:1945
    # -----------------------------------------------------------------------
    elif name == "Antimatter":
        game_state["joker_slots"] = game_state.get("joker_slots", 5) + 1
        mutations["joker_slots"] = game_state["joker_slots"]

    # -----------------------------------------------------------------------
    # Hieroglyph — ante−=extra, blind_ante−=extra, hands−=extra
    # card.lua:1950
    # -----------------------------------------------------------------------
    elif name == "Hieroglyph":
        rr = game_state.setdefault("round_resets", {})
        rr["ante"] = rr.get("ante", 1) - extra
        if "blind_ante" not in rr:
            rr["blind_ante"] = rr["ante"] + extra  # retroactively set before decrement
        rr["blind_ante"] = rr["blind_ante"] - extra
        rr["hands"] = rr.get("hands", 4) - extra
        mutations["round_resets.ante"] = rr["ante"]
        mutations["round_resets.blind_ante"] = rr["blind_ante"]
        mutations["round_resets.hands"] = rr["hands"]

    # -----------------------------------------------------------------------
    # Petroglyph — ante−=extra, blind_ante−=extra, discards−=extra
    # card.lua:1957
    # -----------------------------------------------------------------------
    elif name == "Petroglyph":
        rr = game_state.setdefault("round_resets", {})
        rr["ante"] = rr.get("ante", 1) - extra
        if "blind_ante" not in rr:
            rr["blind_ante"] = rr["ante"] + extra
        rr["blind_ante"] = rr["blind_ante"] - extra
        rr["discards"] = rr.get("discards", 3) - extra
        mutations["round_resets.ante"] = rr["ante"]
        mutations["round_resets.blind_ante"] = rr["blind_ante"]
        mutations["round_resets.discards"] = rr["discards"]

    # -----------------------------------------------------------------------
    # Director's Cut — enable 1 boss blind reroll per ante at cost extra ($10)
    # (not present in extracted apply_to_run; modelled from game behaviour)
    # -----------------------------------------------------------------------
    elif name == "Director's Cut":
        game_state["boss_blind_rerolls"] = game_state.get("boss_blind_rerolls", 0) + 1
        game_state["boss_blind_reroll_cost"] = extra
        mutations["boss_blind_rerolls"] = game_state["boss_blind_rerolls"]
        mutations["boss_blind_reroll_cost"] = game_state["boss_blind_reroll_cost"]

    # -----------------------------------------------------------------------
    # Retcon — unlimited boss blind rerolls, free
    # -----------------------------------------------------------------------
    elif name == "Retcon":
        game_state["boss_blind_rerolls"] = -1  # -1 = unlimited
        game_state["boss_blind_reroll_cost"] = 0
        mutations["boss_blind_rerolls"] = -1
        mutations["boss_blind_reroll_cost"] = 0

    # -----------------------------------------------------------------------
    # Blank — unlock check only; no gameplay mutation (card.lua:1943)
    # Telescope, Observatory — passive effects handled in scoring/pack logic
    # -----------------------------------------------------------------------

    return mutations
