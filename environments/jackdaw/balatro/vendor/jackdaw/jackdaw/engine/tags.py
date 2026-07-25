"""Tag class and effect dispatch for all 24 Balatro tags.

Tags are rewards granted when a player skips a blind.  Each tag fires at a
specific game moment (``context``) and returns a :class:`TagResult` describing
the effect, or ``None`` if the context doesn't match.

Source references
-----------------
- ``tag.lua`` — full tag effect dispatch
- ``jackdaw/engine/data/tags.json`` — 24 tag configs

Context types
-------------

+------------------------+-------------------------------------------------------+
| Context                | Tags                                                  |
+========================+=======================================================+
| ``immediate``          | economy, garbage, handy, orbital, skip, top_up        |
+------------------------+-------------------------------------------------------+
| ``new_blind_choice``   | boss, buffoon, charm, ethereal, meteor, standard       |
+------------------------+-------------------------------------------------------+
| ``eval``               | investment (only when last blind was boss)             |
+------------------------+-------------------------------------------------------+
| ``tag_add``            | double (only when added tag != tag_double)             |
+------------------------+-------------------------------------------------------+
| ``round_start_bonus``  | juggle (+3 hand size for one round)                   |
+------------------------+-------------------------------------------------------+
| ``store_joker_create`` | rare (force rarity 3), uncommon (force rarity 2)      |
+------------------------+-------------------------------------------------------+
| ``shop_start``         | d_six (free rerolls this shop)                        |
+------------------------+-------------------------------------------------------+
| ``store_joker_modify`` | foil, holo, polychrome, negative                      |
+------------------------+-------------------------------------------------------+
| ``shop_final_pass``    | coupon (free all shop items)                          |
+------------------------+-------------------------------------------------------+
| ``voucher_add``        | voucher (add a free voucher)                          |
+------------------------+-------------------------------------------------------+

Pack key conventions
--------------------
- Charm Tag:   ``p_arcana_mega_1``  (always variant 1; see rng.py non-determinism note)
- Meteor Tag:  ``p_celestial_mega_1``
- Ethereal Tag: ``p_spectral_normal_1``
- Standard Tag: ``p_standard_mega_1``
- Buffoon Tag:  ``p_buffoon_mega_1``

Blind-tag generation
--------------------

Tags are generated once per ante in ``Game:start_run / reset_blinds``
(``game.lua:2177-2180``).  The call order is strictly:

1. ``get_new_boss()``        — boss blind for this ante
2. ``get_next_voucher_key()`` — voucher for the shop after the boss
3. ``get_next_tag_key()``    — skip reward for the Small blind
4. ``get_next_tag_key()``    — skip reward for the Big blind

Preserving this order is critical for seed determinism.

Public generators
-----------------
:func:`generate_blind_tags` — produce the ``{Small, Big}`` tag dict.
:func:`assign_ante_blinds`  — full ante setup returning blind choices,
    blind tags, and voucher in one call; also updates ``game_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jackdaw.engine.data.hands import HandType
from jackdaw.engine.data.prototypes import TAGS

if TYPE_CHECKING:
    from jackdaw.engine.rng import PseudoRandom


@dataclass
class TagResult:
    """Descriptor for the effect produced by a :class:`Tag`.

    All fields are optional / zero-valued by default.  Callers inspect
    only the fields relevant to their context.
    """

    dollars: int = 0
    """Immediate cash reward (economy, garbage, handy, skip, investment)."""

    create_pack: str | None = None
    """Pack key to open immediately (buffoon, charm, ethereal, meteor, standard)."""

    create_voucher: bool = False
    """Create a free random voucher from the current pool (voucher tag)."""

    force_rarity: int | None = None
    """Force the next shop joker to this rarity (1=Common, 2=Uncommon, 3=Rare)."""

    force_edition: str | None = None
    """Apply this edition to the shop joker (foil/holo/polychrome/negative)."""

    free_rerolls: int = 0
    """Number of free rerolls this shop session (d_six)."""

    double: bool = False
    """Double Tag: duplicate the tag that was just added."""

    level_up: tuple[HandType, int] | None = None
    """(hand_type, levels) — level up a hand type N times (orbital)."""

    reroll_boss: bool = False
    """Reroll the current boss blind (boss tag)."""

    hand_size_delta: int = 0
    """Temporary hand-size increase for this round (juggle)."""

    create_jokers: int = 0
    """Number of Common jokers to add for free (top-up, up to joker slot cap)."""

    coupon: bool = False
    """Make all shop items free (coupon tag)."""


# Hand types available for Orbital Tag selection (all 12 hand types).
# Order matches ``G.GAME.hands`` iteration order — Lua's pseudorandom_element
# sorts by ``sort_id`` (= order field in HandBaseData).
_ORBITAL_HANDS: list[HandType] = [
    HandType.HIGH_CARD,
    HandType.PAIR,
    HandType.TWO_PAIR,
    HandType.THREE_OF_A_KIND,
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.STRAIGHT_FLUSH,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
]


class Tag:
    """Runtime representation of a Balatro tag.

    Parameters
    ----------
    key:
        Tag key from ``TAGS`` (e.g. ``"tag_economy"``).
    tag_id:
        Optional integer identifier used by the game to track which specific
        tag instance triggered.  Not required for effect dispatch.
    """

    __slots__ = ("key", "proto", "name", "config", "id", "triggered")

    def __init__(self, key: str, tag_id: int | None = None) -> None:
        self.key: str = key
        self.proto = TAGS[key]
        self.name: str = self.proto.name
        self.config: dict[str, Any] = dict(self.proto.config) if self.proto.config else {}
        self.id: int | None = tag_id
        self.triggered: bool = False

    # ------------------------------------------------------------------
    # Effect dispatch
    # ------------------------------------------------------------------

    def apply(
        self,
        context: str,
        game_state: dict[str, Any],
        rng: PseudoRandom | None = None,
        **kwargs: Any,
    ) -> TagResult | None:
        """Fire the tag's effect.

        Returns a :class:`TagResult` when the tag fires, ``None`` when the
        context does not match (tag does not apply here).

        Parameters
        ----------
        context:
            The game moment string (e.g. ``"immediate"``, ``"eval"``, …).
        game_state:
            Mutable run-state dict.  Some tags read values from this dict
            (e.g. ``game_state.get("dollars")`` for Economy Tag).
        rng:
            Active :class:`~jackdaw.engine.rng.PseudoRandom` instance.
            Required for Orbital Tag's random hand selection.
        **kwargs:
            Context-specific extra data:

            * ``last_blind_is_boss`` (bool) — Investment Tag
            * ``added_tag_key`` (str)       — Double Tag
        """
        tag_type = self.config.get("type", "")

        # ----------------------------------------------------------------
        # immediate — fires right when the tag is collected
        # ----------------------------------------------------------------
        if context == "immediate":
            if tag_type != "immediate":
                return None

            if self.key == "tag_economy":
                current = game_state.get("dollars", 0)
                reward = min(self.config["max"], max(0, current))
                return TagResult(dollars=reward)

            if self.key == "tag_garbage":
                unused = game_state.get("unused_discards", 0)
                return TagResult(dollars=unused * self.config["dollars_per_discard"])

            if self.key == "tag_handy":
                played = game_state.get("hands_played", 0)
                return TagResult(dollars=played * self.config["dollars_per_hand"])

            if self.key == "tag_skip":
                skips = game_state.get("skips", 0)
                return TagResult(dollars=skips * self.config["skip_bonus"])

            if self.key == "tag_top_up":
                return TagResult(create_jokers=self.config["spawn_jokers"])

            if self.key == "tag_orbital":
                if rng is None:
                    raise ValueError("tag_orbital requires an rng instance")
                seed_val = rng.seed("orbital")
                # Select a random hand type — mirrors pseudorandom_element
                # over the sorted hand list (sorted by order/sort_id in Lua)
                from jackdaw.engine.data.hands import HAND_BASE

                hands_by_order = sorted(HAND_BASE.keys(), key=lambda h: HAND_BASE[h].order)
                idx = rng.random(seed_val, 1, len(hands_by_order))
                chosen = hands_by_order[idx - 1]
                levels = self.config["levels"]
                return TagResult(level_up=(chosen, levels))

            # Unknown immediate tag — no effect
            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # new_blind_choice — fires when entering a new blind (skip reward)
        # ----------------------------------------------------------------
        if context == "new_blind_choice":
            if tag_type != "new_blind_choice":
                return None

            if self.key == "tag_boss":
                return TagResult(reroll_boss=True)

            if self.key == "tag_buffoon":
                return TagResult(create_pack="p_buffoon_mega_1")

            if self.key == "tag_charm":
                # Lua uses math.random(1,2) here — non-deterministic.
                # We always pick variant 1 (see rng.py Known Non-Determinism).
                return TagResult(create_pack="p_arcana_mega_1")

            if self.key == "tag_ethereal":
                return TagResult(create_pack="p_spectral_normal_1")

            if self.key == "tag_meteor":
                # Same non-determinism caveat as Charm — always variant 1.
                return TagResult(create_pack="p_celestial_mega_1")

            if self.key == "tag_standard":
                return TagResult(create_pack="p_standard_mega_1")

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # eval — fires after each blind is scored
        # ----------------------------------------------------------------
        if context == "eval":
            if tag_type != "eval":
                return None

            if self.key == "tag_investment":
                if kwargs.get("last_blind_is_boss", False):
                    return TagResult(dollars=self.config["dollars"])
            return None

        # ----------------------------------------------------------------
        # tag_add — fires when any tag is added to the run
        # ----------------------------------------------------------------
        if context == "tag_add":
            if tag_type != "tag_add":
                return None

            if self.key == "tag_double":
                added_key = kwargs.get("added_tag_key", "")
                if added_key and added_key != "tag_double":
                    return TagResult(double=True)
            return None

        # ----------------------------------------------------------------
        # round_start_bonus — fires at the start of each round
        # ----------------------------------------------------------------
        if context == "round_start_bonus":
            if tag_type != "round_start_bonus":
                return None

            if self.key == "tag_juggle":
                return TagResult(hand_size_delta=self.config["h_size"])

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # store_joker_create — fires when a shop joker slot is being filled
        # ----------------------------------------------------------------
        if context == "store_joker_create":
            if tag_type != "store_joker_create":
                return None

            if self.key == "tag_uncommon":
                return TagResult(force_rarity=2)

            if self.key == "tag_rare":
                return TagResult(force_rarity=3)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # shop_start — fires at the beginning of the shop phase
        # ----------------------------------------------------------------
        if context == "shop_start":
            if tag_type != "shop_start":
                return None

            if self.key == "tag_d_six":
                return TagResult(free_rerolls=1)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # store_joker_modify — fires when a shop joker exists, adds edition
        # ----------------------------------------------------------------
        if context == "store_joker_modify":
            if tag_type != "store_joker_modify":
                return None

            edition = self.config.get("edition")
            if edition:
                return TagResult(force_edition=edition)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # shop_final_pass — fires at the end of shop generation
        # ----------------------------------------------------------------
        if context == "shop_final_pass":
            if tag_type != "shop_final_pass":
                return None

            if self.key == "tag_coupon":
                return TagResult(coupon=True)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # voucher_add — fires when a voucher is being added to the shop
        # ----------------------------------------------------------------
        if context == "voucher_add":
            if tag_type != "voucher_add":
                return None

            if self.key == "tag_voucher":
                return TagResult(create_voucher=True)

            return None  # pragma: no cover

        # Unknown context — tag does not apply
        return None

    def __repr__(self) -> str:
        return f"Tag({self.key!r}, id={self.id})"


# ---------------------------------------------------------------------------
# Blind-tag generation
# ---------------------------------------------------------------------------


def generate_blind_tags(
    ante: int,
    rng: PseudoRandom,
    game_state: dict[str, Any],
) -> dict[str, str]:
    """Generate tags for Small and Big blind skip rewards.

    Calls :func:`~jackdaw.engine.pools.pick_card_from_pool` twice with the
    ``"Tag"`` pool, applying ``min_ante`` and ``requires`` filters from
    *game_state*.

    Parameters
    ----------
    ante:
        Current ante number (1–8).
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Each
        call advances the ``"Tag{ante}"`` stream once.
    game_state:
        Run-state dict.  Reads:

        * ``used_vouchers`` (iterable of str, optional) — purchased voucher
          keys; used to gate tags with a ``requires`` field.

    Returns
    -------
    dict
        ``{"Small": tag_key, "Big": tag_key}``
    """
    from jackdaw.engine.pools import pick_card_from_pool

    used_vouchers: set[str] = set(game_state.get("used_vouchers", []))
    discovered: set[str] | None = game_state.get("discovered")
    small = pick_card_from_pool(
        "Tag", rng, ante, used_vouchers=used_vouchers, discovered=discovered
    )
    big = pick_card_from_pool("Tag", rng, ante, used_vouchers=used_vouchers, discovered=discovered)
    return {"Small": small, "Big": big}


def assign_ante_blinds(
    ante: int,
    rng: PseudoRandom,
    game_state: dict[str, Any],
) -> dict[str, Any]:
    """Set up blinds and tags for a new ante.

    RNG calls are issued in the exact Lua source order to preserve seed
    determinism.  The Lua game generates these across two call sites:

    1. ``get_next_voucher_key()`` → state_events.lua:263 (boss defeat)
    2. ``get_next_tag_key()``     → button_callbacks.lua:2951 (cash_out)
    3. ``get_next_tag_key()``     → button_callbacks.lua:2952 (cash_out)
    4. ``get_new_boss()``         → common_events.lua:2333 (reset_blinds)

    Side effects on *game_state*
    ----------------------------
    * ``game_state["bosses_used"]`` is created/updated in-place (boss usage
      tracking that prevents the same boss from repeating too soon).
    * ``game_state["round_resets"]["blind_tags"]`` is set to
      ``{"Small": tag_key, "Big": tag_key}``.

    Parameters
    ----------
    ante:
        Current ante number (1–8+).
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    game_state:
        Mutable run-state dict.  Reads:

        * ``bosses_used`` (dict[str, int], created if absent)
        * ``used_vouchers`` (iterable of str, optional)

    Returns
    -------
    dict
        ``{"blind_choices": {"Small": "bl_small", "Big": "bl_big",
        "Boss": boss_key}, "blind_tags": {"Small": tag_key,
        "Big": tag_key}, "voucher": voucher_key_or_None}``
    """
    from jackdaw.engine.blind import get_new_boss
    from jackdaw.engine.pools import pick_card_from_pool
    from jackdaw.engine.vouchers import get_next_voucher_key

    bosses_used: dict[str, int] = game_state.setdefault("bosses_used", {})
    used_vouchers: set[str] = set(game_state.get("used_vouchers", []))
    used_v_dict: dict[str, bool] = {k: True for k in used_vouchers}
    banned_keys: set[str] = {
        key for key, banned in game_state.get("banned_keys", {}).items() if banned
    }

    # RNG call order must match Lua exactly:
    #   1. get_next_voucher_key() — state_events.lua:263 (boss defeat)
    #   2. get_next_tag_key() x2   — button_callbacks.lua:2949-2952 (cash_out)
    #   3. get_new_boss()          — common_events.lua:2333 (reset_blinds)

    # 1. Voucher
    #    In Lua, get_next_voucher_key() fires at boss defeat
    #    (state_events.lua:263).  By this point G.shop:remove() has already
    #    run (button_callbacks.lua:2500), which calls CardArea:remove() on
    #    G.shop_vouchers — setting its .cards to nil (cardarea.lua:659).
    #    Therefore the shop-exclusion guard in get_current_pool
    #    (common_events.lua:1999-2003) never fires and no voucher is
    #    excluded from the pool.
    voucher = get_next_voucher_key(
        rng,
        used_v_dict,
        in_shop=None,
        ante=ante,
        banned_keys=banned_keys,
    )

    # 2-3. Tags (Small then Big)
    small = pick_card_from_pool(
        "Tag",
        rng,
        ante,
        used_vouchers=used_vouchers,
        banned_keys=banned_keys,
    )
    big = pick_card_from_pool(
        "Tag",
        rng,
        ante,
        used_vouchers=used_vouchers,
        banned_keys=banned_keys,
    )

    # 4. Boss
    boss = get_new_boss(ante, bosses_used, rng)

    blind_tags: dict[str, str] = {"Small": small, "Big": big}

    # Persist to game_state (mirrors G.GAME.round_resets.blind_tags)
    round_resets: dict[str, Any] = game_state.setdefault("round_resets", {})
    round_resets["blind_tags"] = blind_tags

    return {
        "blind_choices": {"Small": "bl_small", "Big": "bl_big", "Boss": boss},
        "blind_tags": blind_tags,
        "voucher": voucher,
    }
