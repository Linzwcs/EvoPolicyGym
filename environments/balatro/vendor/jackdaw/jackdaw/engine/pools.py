"""Pool generation and selection for Balatro.

Ports two functions from ``common_events.lua``:

* ``get_current_pool`` (line 1963) — builds a filtered list of center keys
  with ``"UNAVAILABLE"`` sentinels preserving deterministic index alignment.
* Selection helpers (line 2128) — ``select_from_pool`` picks one key from a
  pre-built pool; ``pick_card_from_pool`` combines both steps.
"""

from __future__ import annotations

from jackdaw.engine.data.prototypes import (
    CENTER_POOLS,
    JOKER_RARITY_POOLS,
    JOKERS,
    SPECTRALS,
    TAGS,
    VOUCHERS,
)
from jackdaw.engine.rng import PseudoRandom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNAVAILABLE = "UNAVAILABLE"

_FALLBACKS: dict[str, str] = {
    "Joker": "j_joker",
    "Tarot": "c_strength",
    "Planet": "c_pluto",
    "Spectral": "c_incantation",
    "Voucher": "v_blank",
    "Tag": "tag_handy",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_current_pool(
    pool_type: str,
    rng: PseudoRandom,
    ante: int,
    *,
    rarity: int | None = None,
    legendary: bool = False,
    append: str = "",
    used_jokers: set[str] | None = None,
    used_vouchers: set[str] | None = None,
    banned_keys: set[str] | None = None,
    pool_flags: dict[str, bool] | None = None,
    has_showman: bool = False,
    deck_enhancements: set[str] | None = None,
    playing_card_count: int = 52,
    played_hand_types: set[str] | None = None,
    shop_vouchers: set[str] | None = None,
    discovered: set[str] | None = None,
    profile_unlocked: set[str] | None = None,
) -> tuple[list[str], str]:
    """Return ``(pool, seed_key)`` for the given pool type.

    *pool* is a list of center keys with ``"UNAVAILABLE"`` in the place of
    any entry that was filtered out.  *seed_key* is the string to pass to
    ``rng.seed()`` before drawing an element (``pool_type + append``).

    Parameters
    ----------
    pool_type:
        One of ``"Joker"``, ``"Tarot"``, ``"Planet"``, ``"Spectral"``,
        ``"Voucher"``, ``"Enhanced"``, or ``"Tag"``.
    rng:
        Live ``PseudoRandom`` instance (rarity roll advances the stream).
    ante:
        Current ante number (used in rarity roll key and Tag ``min_ante``).
    rarity:
        Force a specific Joker rarity (1–4).  ``None`` → roll.
    legendary:
        Force rarity 4 without rolling (``Wraith`` / ``Soul`` joker draw).
    append:
        Seed-key suffix (e.g. ``"shop"``, ``"pack"``).  Appended to both
        the rarity roll key and the returned seed key.
    used_jokers:
        Set of joker keys already held; duplicates become UNAVAILABLE unless
        ``has_showman`` is True.
    used_vouchers:
        Set of voucher keys already purchased; used for prerequisite checks.
    banned_keys:
        Keys explicitly excluded (e.g. from challenge rules).
    pool_flags:
        Active boolean game flags (e.g. ``{"gros_michel_extinct": True}``).
    has_showman:
        When True, duplicate-joker filtering is skipped.
    deck_enhancements:
        Set of enhancement keys present on cards in the deck.  Required for
        ``enhancement_gate`` jokers.
    playing_card_count:
        Number of playing cards in the deck (unused in current filters but
        matches the Lua signature for forward-compatibility).
    played_hand_types:
        Set of hand-type names played at least once this run.  Required for
        Planet softlock filtering.
    shop_vouchers:
        Set of voucher keys currently displayed in the shop (excluded from
        Voucher pool to avoid duplicates).
    """
    used_jokers = used_jokers or set()
    used_vouchers = used_vouchers or set()
    banned_keys = banned_keys or set()
    pool_flags = pool_flags or {}
    deck_enhancements = deck_enhancements or set()
    played_hand_types = played_hand_types or set()
    shop_vouchers = shop_vouchers or set()

    # ------------------------------------------------------------------
    # 1. Build the ordered base key list
    # ------------------------------------------------------------------
    if pool_type == "Joker":
        if legendary:
            rarity = 4
        elif rarity is None:
            roll = rng.random("rarity" + str(ante) + append)
            if roll > 0.95:
                rarity = 3
            elif roll > 0.7:
                rarity = 2
            else:
                rarity = 1
        base_keys: list[str] = list(JOKER_RARITY_POOLS.get(rarity, []))
    elif pool_type == "Tag":
        base_keys = sorted(TAGS.keys(), key=lambda k: TAGS[k].order)
    else:
        base_keys = list(CENTER_POOLS.get(pool_type, []))

    # ------------------------------------------------------------------
    # 2. Filter each key
    # ------------------------------------------------------------------
    result: list[str] = []
    for key in base_keys:
        entry = _filter_key(
            key=key,
            pool_type=pool_type,
            ante=ante,
            used_jokers=used_jokers,
            used_vouchers=used_vouchers,
            banned_keys=banned_keys,
            pool_flags=pool_flags,
            has_showman=has_showman,
            deck_enhancements=deck_enhancements,
            played_hand_types=played_hand_types,
            shop_vouchers=shop_vouchers,
            discovered=discovered,
            profile_unlocked=profile_unlocked,
        )
        result.append(entry)

    # ------------------------------------------------------------------
    # 3. Empty-pool fallback
    # ------------------------------------------------------------------
    if all(e == UNAVAILABLE for e in result):
        fallback = _FALLBACKS.get(pool_type, UNAVAILABLE)
        result = [fallback]

    # Lua's _pool_key includes rarity for Jokers: 'Joker'..rarity..append
    # For non-Jokers: pool_type..append
    if pool_type == "Joker" and rarity is not None:
        seed_key = pool_type + str(rarity) + (append if not legendary else "")
    else:
        seed_key = pool_type + append
    return result, seed_key


# ---------------------------------------------------------------------------
# Soul / Black Hole chance — common_events.lua:2087
# ---------------------------------------------------------------------------

_SOUL_THRESHOLD = 0.997


def check_soul_chance(
    pool_type: str,
    rng: PseudoRandom,
    ante: int,
    soulable: bool = True,
) -> str | None:
    """Check for a forced Soul or Black Hole card (0.3% chance per roll).

    This fires **before** pool selection in ``create_card``.  If it returns
    a key, that key is used directly and pool generation is skipped entirely.

    The roll key is ``'soul_' + pool_type + str(ante)`` (advances the named
    stream via :meth:`~PseudoRandom.random`).

    * **Joker** — one roll; hit → ``'c_soul'``
    * **Planet** — one roll; hit → ``'c_black_hole'``
    * **Spectral** — *two* rolls on the same stream key; first hit → ``'c_soul'``,
      second hit → ``'c_black_hole'``
    * **All other types** — no roll is made

    Parameters
    ----------
    pool_type:
        Pool type string (``"Joker"``, ``"Planet"``, ``"Spectral"``, etc.).
    rng:
        Live ``PseudoRandom`` instance.
    ante:
        Current ante number (appended to the stream key).
    soulable:
        When ``False`` (e.g. for playing-card draws), the check is bypassed
        entirely and ``None`` is returned without consuming any RNG state.

    Returns
    -------
    str | None
        ``'c_soul'``, ``'c_black_hole'``, or ``None``.
    """
    if not soulable:
        return None

    roll_key = "soul_" + pool_type + str(ante)

    if pool_type == "Joker":
        if rng.random(roll_key) > _SOUL_THRESHOLD:
            return "c_soul"
        return None

    if pool_type == "Planet":
        if rng.random(roll_key) > _SOUL_THRESHOLD:
            return "c_black_hole"
        return None

    if pool_type == "Spectral":
        if rng.random(roll_key) > _SOUL_THRESHOLD:
            return "c_soul"
        if rng.random(roll_key) > _SOUL_THRESHOLD:
            return "c_black_hole"
        return None

    return None


# ---------------------------------------------------------------------------
# Selection — common_events.lua:2128
# ---------------------------------------------------------------------------

_MAX_RESAMPLES = 20


def select_from_pool(
    pool: list[str],
    rng: PseudoRandom,
    pool_key: str,
    ante: int,
    *,
    pool_type: str = "",
) -> str:
    """Select one key from a pre-built pool using deterministic RNG.

    Mirrors the Lua pattern at ``common_events.lua:2128``::

        center = pseudorandom_element(_pool, pseudoseed(_pool_key))
        -- if UNAVAILABLE, retry with _pool_key..'_resample'..i

    Parameters
    ----------
    pool:
        List of center keys (with ``"UNAVAILABLE"`` sentinels) as returned
        by :func:`get_current_pool`.
    rng:
        Live ``PseudoRandom`` instance.  Each call to :meth:`~PseudoRandom.seed`
        advances the named stream.
    pool_key:
        Base pool type string (``"Joker"``, ``"Tarot"``, etc.).  Combined
        with *append* and *ante* to form the RNG stream key.
    ante:
        Current ante number.  Appended to the stream key so each ante uses
        an independent stream.
    append:
        Seed-key suffix (e.g. ``"shop"``, ``"pack"``).

    Returns
    -------
    str
        The selected center key, or the fallback key for this pool type if
        every resample attempt also hits ``UNAVAILABLE``.
    """
    # The full seed key matches Lua's _pool_key which is
    # pool_type + append + str(ante) (returned by get_current_pool at
    # common_events.lua:2052).  E.g. 'Tag1', 'Joker1sho1', 'Voucher1'.
    # NOTE: pool_key already contains append (from get_current_pool),
    # so we only add ante here — NOT append again.
    full_key = pool_key + str(ante)

    # Initial draw: pseudoseed(full_key)
    seed_val = rng.seed(full_key)
    value, _ = rng.element(pool, seed_val)

    if value != UNAVAILABLE:
        return value

    # Resample loop (up to _MAX_RESAMPLES attempts).
    # Lua starts it=1, increments BEFORE use: first resample is it=2.
    # Key: _pool_key..'_resample'..it  (e.g. 'Tag1_resample2')
    for it in range(2, _MAX_RESAMPLES + 2):
        seed_val = rng.seed(full_key + "_resample" + str(it))
        value, _ = rng.element(pool, seed_val)
        if value != UNAVAILABLE:
            return value

    # All resamples exhausted — return type fallback
    return _FALLBACKS.get(pool_type, _FALLBACKS.get(pool_key, UNAVAILABLE))


def pick_card_from_pool(
    pool_type: str,
    rng: PseudoRandom,
    ante: int,
    **kwargs,
) -> str:
    """Build a filtered pool then select one card key from it.

    Combines :func:`get_current_pool` and :func:`select_from_pool` into a
    single call, matching the usage pattern at ``common_events.lua:2128``.

    Parameters
    ----------
    pool_type:
        One of ``"Joker"``, ``"Tarot"``, ``"Planet"``, ``"Spectral"``,
        ``"Voucher"``, ``"Enhanced"``, or ``"Tag"``.
    rng:
        Live ``PseudoRandom`` instance.
    ante:
        Current ante number.
    **kwargs:
        Forwarded to :func:`get_current_pool` (``rarity``, ``legendary``,
        ``append``, ``used_jokers``, ``banned_keys``, etc.).

    Returns
    -------
    str
        The selected center key.
    """
    pool, pool_key = get_current_pool(pool_type, rng, ante, **kwargs)
    return select_from_pool(pool, rng, pool_key, ante, pool_type=pool_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_key(
    *,
    key: str,
    pool_type: str,
    ante: int,
    used_jokers: set[str],
    used_vouchers: set[str],
    banned_keys: set[str],
    pool_flags: dict[str, bool],
    has_showman: bool,
    deck_enhancements: set[str],
    played_hand_types: set[str],
    shop_vouchers: set[str],
    discovered: set[str] | None = None,
    profile_unlocked: set[str] | None = None,
) -> str:
    """Return *key* if it passes all filters, otherwise ``UNAVAILABLE``."""

    # --- Universal: banned keys ---
    if key in banned_keys:
        return UNAVAILABLE

    # --- Unlock check SKIPPED (common_events.lua:1988) ---
    # Sim assumes a fully-unlocked profile so all items are available.
    # Rarity 4 (legendary) jokers always bypass per hardcoded Lua rule
    # (v.unlocked ~= false or v.rarity == 4); see _filter_joker.

    # --- Enhanced / Tag: no used_jokers check (separate branches in Lua) ---
    if pool_type == "Enhanced":
        return key

    if pool_type == "Tag":
        return _filter_tag(key=key, ante=ante, used_vouchers=used_vouchers, discovered=discovered)

    # --- Universal duplicate check (common_events.lua:1987) ---
    # Lua checks G.GAME.used_jokers[v.key] for ALL remaining types
    # (Joker, Tarot, Planet, Spectral, Voucher).  Despite the name,
    # used_jokers tracks every center key that has been created via
    # Card:set_ability (card.lua:349-354).
    if not has_showman and key in used_jokers:
        return UNAVAILABLE

    if pool_type == "Joker":
        return _filter_joker(
            key=key,
            pool_flags=pool_flags,
            deck_enhancements=deck_enhancements,
        )

    if pool_type == "Spectral":
        return _filter_spectral(key)

    if pool_type == "Voucher":
        return _filter_voucher(
            key=key,
            used_vouchers=used_vouchers,
            shop_vouchers=shop_vouchers,
        )

    if pool_type == "Planet":
        return _filter_planet(key=key, played_hand_types=played_hand_types)

    # Tarot / unknown → no extra filtering beyond the universal checks
    return key


def _filter_joker(
    *,
    key: str,
    pool_flags: dict[str, bool],
    deck_enhancements: set[str],
) -> str:
    proto = JOKERS.get(key)
    if proto is None:
        return UNAVAILABLE

    # Soul / Black Hole: never in pool (handled via hidden flag analogue)
    # In the joker set these are not present, but guard anyway
    if key in ("c_soul", "c_black_hole"):
        return UNAVAILABLE

    # Duplicate check now handled universally in _filter_key

    # Unlock check (common_events.lua:1988):
    # Rarity 4 (legendary) jokers always bypass the unlock gate —
    # hardcoded Lua rule: (v.unlocked ~= false or v.rarity == 4).
    # All other items are treated as fully-unlocked (no filtering).
    if proto.rarity == 4:
        pass  # always available regardless of unlock state

    # no_pool_flag: excluded when flag is True
    if proto.no_pool_flag and pool_flags.get(proto.no_pool_flag):
        return UNAVAILABLE

    # yes_pool_flag: excluded when flag is NOT True
    if proto.yes_pool_flag and not pool_flags.get(proto.yes_pool_flag):
        return UNAVAILABLE

    # enhancement_gate: excluded when enhancement not in deck
    if proto.enhancement_gate and proto.enhancement_gate not in deck_enhancements:
        return UNAVAILABLE

    return key


def _filter_spectral(key: str) -> str:
    proto = SPECTRALS.get(key)
    if proto is None:
        return UNAVAILABLE
    if proto.hidden:
        return UNAVAILABLE
    return key


def _filter_voucher(
    *,
    key: str,
    used_vouchers: set[str],
    shop_vouchers: set[str],
) -> str:
    proto = VOUCHERS.get(key)
    if proto is None:
        return UNAVAILABLE

    # Already purchased
    if key in used_vouchers:
        return UNAVAILABLE

    # Currently in shop (avoid showing twice)
    if key in shop_vouchers:
        return UNAVAILABLE

    # Prerequisites: all required vouchers must be purchased
    for req in proto.requires:
        if req not in used_vouchers:
            return UNAVAILABLE

    return key


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
    "c_black_hole": "all",
}


def _is_softlocked_planet(key: str, visible_hand_types: set[str]) -> bool:
    """Return True if *key* is a planet whose hand type is not yet visible.

    In Lua, softlocked planets are simply absent from the pool — they are
    never added, so they don't affect pool size or index alignment.
    """
    hand_type = _PLANET_HAND.get(key)
    if hand_type is None or hand_type == "all":
        return False
    return hand_type not in visible_hand_types


_SOFTLOCK_PLANETS: dict[str, str] = {
    "c_planet_x": "Five of a Kind",
    "c_ceres": "Flush House",
    "c_eris": "Flush Five",
}


def _filter_planet(*, key: str, played_hand_types: set[str]) -> str:
    """Exclude softlock planets until their hand type has been played this run.

    common_events.lua:2008-2011 gates only planets with ``config.softlock``
    (Planet X / Ceres / Eris) on ``G.GAME.hands[hand_type].played > 0``.
    All other planets are always eligible regardless of play history.
    """
    hand_type = _SOFTLOCK_PLANETS.get(key)
    if hand_type is not None and hand_type not in played_hand_types:
        return UNAVAILABLE
    return key


def _filter_tag(
    *,
    key: str,
    ante: int,
    used_vouchers: set[str],
    discovered: set[str] | None = None,
) -> str:
    proto = TAGS.get(key)
    if proto is None:
        return UNAVAILABLE

    # min_ante gate
    if proto.min_ante is not None and ante < proto.min_ante:
        return UNAVAILABLE

    # requires: in Lua, this checks G.P_CENTERS[v.requires].discovered —
    # whether the item has been SEEN in a previous run, not purchased.
    # The engine assumes a fully-unlocked profile (consistent with the
    # joker unlock skip at line 377), so if discovered is None we treat
    # all items as discovered.  Pass an explicit set to restrict.
    if proto.requires and discovered is not None:
        if "all" not in discovered and proto.requires not in discovered:
            return UNAVAILABLE

    return key
