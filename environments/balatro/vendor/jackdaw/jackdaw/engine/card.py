"""Card class — the central data structure for all game objects.

Every playing card, joker, tarot, planet, spectral, voucher, and booster
is a Card instance.  Mirrors the Lua ``Card`` object structure from
``card.lua``, keeping the ``ability`` dict untyped to match Lua's dynamic
table semantics.

Source: card.lua lines 5-77 (init), 97-145 (set_base), 223-342 (set_ability).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.data.enums import Rank, Suit

# ---------------------------------------------------------------------------
# Module-level sort_id counter (matches G.sort_id in globals.lua)
# ---------------------------------------------------------------------------

_sort_id_counter: int = 0


def _next_sort_id() -> int:
    global _sort_id_counter  # noqa: PLW0603
    _sort_id_counter += 1
    return _sort_id_counter


def reset_sort_id_counter() -> None:
    """Reset to 0 (call at run start, matching G.sort_id = 0)."""
    global _sort_id_counter  # noqa: PLW0603
    _sort_id_counter = 0


# ---------------------------------------------------------------------------
# Face nominal values (from Card:set_base, card.lua:131-134)
# ---------------------------------------------------------------------------

_FACE_NOMINAL: dict[str, float] = {
    "Jack": 0.1,
    "Queen": 0.2,
    "King": 0.3,
    "Ace": 0.4,
}

_RANK_NOMINAL: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 10,
    "Queen": 10,
    "King": 10,
    "Ace": 11,
}

_RANK_ID: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 11,
    "Queen": 12,
    "King": 13,
    "Ace": 14,
}

_SUIT_NOMINAL: dict[str, float] = {
    "Diamonds": 0.01,
    "Clubs": 0.02,
    "Hearts": 0.03,
    "Spades": 0.04,
}

_SUIT_NOMINAL_ORIGINAL: dict[str, float] = {
    "Diamonds": 0.001,
    "Clubs": 0.002,
    "Hearts": 0.003,
    "Spades": 0.004,
}


# ---------------------------------------------------------------------------
# CardBase — playing card identity
# ---------------------------------------------------------------------------


@dataclass
class CardBase:
    """Base identity for a playing card (suit, rank, and derived numeric values).

    Corresponds to ``self.base`` in card.lua, populated by ``Card:set_base``.
    """

    suit: Suit
    rank: Rank  # the value string: "Ace", "King", ..., "2"
    id: int  # numeric rank id (2-14, Ace=14)
    nominal: int  # chip value
    suit_nominal: float  # suit tiebreaker (S=0.04, H=0.03, C=0.02, D=0.01)
    suit_nominal_original: float  # preserved original suit tiebreaker
    face_nominal: float  # 0.0 for non-face, 0.1-0.4 for J/Q/K/A
    original_value: Rank  # original rank before Strength tarot changes
    times_played: int = 0

    @staticmethod
    def from_card_key(card_key: str, suit: str, value: str) -> CardBase:
        """Build a CardBase from P_CARDS data, matching Card:set_base."""
        rank = Rank(value)
        suit_enum = Suit(suit)
        return CardBase(
            suit=suit_enum,
            rank=rank,
            id=_RANK_ID[value],
            nominal=_RANK_NOMINAL[value],
            suit_nominal=_SUIT_NOMINAL[suit],
            suit_nominal_original=_SUIT_NOMINAL_ORIGINAL[suit],
            face_nominal=_FACE_NOMINAL.get(value, 0.0),
            original_value=rank,
            times_played=0,
        )


# ---------------------------------------------------------------------------
# Center resolution helper
# ---------------------------------------------------------------------------

# Lazy import to avoid circular dependency at module load time.
# prototypes.py imports nothing from card.py, so this is safe.
_centers_cache: dict[str, dict[str, Any]] | None = None


def _resolve_center(key: str) -> dict[str, Any]:
    """Look up a P_CENTERS entry by key, returning a plain dict.

    Converts the frozen dataclass to a dict on first call and caches the
    result for fast repeated access.
    """
    global _centers_cache  # noqa: PLW0603
    if _centers_cache is None:
        from jackdaw.engine.data.prototypes import _load_json

        _centers_cache = _load_json("centers.json")
    if key not in _centers_cache:
        raise KeyError(f"Unknown center key: {key!r}")
    return _centers_cache[key]


# ---------------------------------------------------------------------------
# Card — the main object
# ---------------------------------------------------------------------------


@dataclass
class Card:
    """A card in the game — playing card, joker, tarot, planet, spectral, etc.

    The ``ability`` dict is intentionally untyped (``dict[str, Any]``) to match
    Lua's dynamic table semantics.  It is populated by :meth:`set_ability` and
    freely mutated by joker effects during gameplay.
    """

    # Identity
    sort_id: int = field(default_factory=_next_sort_id)

    # Base (playing cards only — None for jokers/consumables/vouchers)
    base: CardBase | None = None

    # Center (prototype reference)
    center_key: str = "c_base"  # P_CENTERS key
    card_key: str | None = None  # P_CARDS key for playing cards

    # Mutable ability state
    ability: dict[str, Any] = field(default_factory=dict)

    # Modifiers
    edition: dict[str, bool] | None = None  # None, {"foil": True}, etc.
    seal: str | None = None  # None, "Red", "Blue", "Gold", "Purple"
    debuff: bool = False

    # Status
    playing_card: int | None = None  # index in playing_cards list
    facing: str = "front"  # "front" or "back"

    # Economy
    base_cost: int = 0
    cost: int = 0
    sell_cost: int = 0
    extra_cost: int = 0

    # Stickers
    eternal: bool = False
    perishable: bool = False
    perish_tally: int = 5  # rounds until perish
    rental: bool = False

    def set_base(self, card_key: str, suit: str, value: str) -> None:
        """Populate base fields from P_CARDS data, matching Card:set_base."""
        self.card_key = card_key
        self.base = CardBase.from_card_key(card_key, suit, value)

    def set_ability(
        self,
        center: dict[str, Any] | str,
        *,
        hands_played: int = 0,
    ) -> None:
        """Populate ability from a prototype, matching card.lua:223 (Card:set_ability).

        Args:
            center: Either a P_CENTERS key string (e.g. ``"j_joker"``) or a raw
                    dict with ``name``, ``set``, ``config``, etc. fields.
            hands_played: Current ``G.GAME.hands_played`` for the
                          ``hands_played_at_create`` field.
        """
        if isinstance(center, str):
            center = _resolve_center(center)

        config = center.get("config") or {}
        if isinstance(config, list):
            config = {}  # empty Lua table [] → {}

        # Preserve fields that survive center changes (card.lua:295-296)
        old_perma_bonus = self.ability.get("perma_bonus", 0)
        old_forced_selection = self.ability.get("forced_selection")

        # Deep copy extra so each card instance owns its mutable state
        extra = config.get("extra")
        if extra is not None:
            extra = copy.deepcopy(extra)

        self.ability = {
            "name": center.get("name", ""),
            "effect": center.get("effect", ""),
            "set": center.get("set", ""),
            "mult": config.get("mult", 0),
            "h_mult": config.get("h_mult", 0),
            "h_x_mult": config.get("h_x_mult", 0),
            "h_dollars": config.get("h_dollars", 0),
            "p_dollars": config.get("p_dollars", 0),
            "t_mult": config.get("t_mult", 0),
            "t_chips": config.get("t_chips", 0),
            "x_mult": config.get("Xmult", 1),  # Xmult → x_mult
            "h_size": config.get("h_size", 0),
            "d_size": config.get("d_size", 0),
            "extra": extra,
            "extra_value": 0,
            "type": config.get("type", ""),
            "order": center.get("order"),
            "forced_selection": old_forced_selection,
            "perma_bonus": old_perma_bonus,
        }

        # Bonus accumulates (card.lua:299)
        self.ability["bonus"] = self.ability.get("bonus", 0) + config.get("bonus", 0)

        # Consumable config reference (card.lua:301-303)
        if center.get("consumeable"):
            self.ability["consumeable"] = config

        self.center_key = center.get("key", self.center_key)
        self.base_cost = center.get("cost", 1)

        # -- Special post-init fields (card.lua:308-337) --

        name = self.ability["name"]

        if name == "Invisible Joker":
            self.ability["invis_rounds"] = 0

        if name == "Caino":
            self.ability["caino_xmult"] = 1

        if name == "Yorick" and isinstance(self.ability.get("extra"), dict):
            self.ability["yorick_discards"] = self.ability["extra"].get("discards", 0)

        if name == "Loyalty Card" and isinstance(self.ability.get("extra"), dict):
            self.ability["burnt_hand"] = 0
            self.ability["loyalty_remaining"] = self.ability["extra"].get("every", 0)

        # hands_played_at_create (card.lua:337)
        self.ability["hands_played_at_create"] = hands_played

    def enhance(self, center_key: str) -> None:
        """Change enhancement while preserving base, edition, seal, perma_bonus.

        Mirrors the tarot enhancement path in card.lua:use_consumeable
        where ``set_ability`` is called but base identity, edition, seal,
        and accumulated bonuses are preserved.

        This is distinct from ``set_ability`` which is a full reset.
        """
        # Save fields that must survive enhancement change
        old_edition = self.edition
        old_seal = self.seal
        old_perma_bonus = self.ability.get("perma_bonus", 0)
        old_bonus = self.ability.get("bonus", 0)

        # Apply new center (resets ability dict)
        self.set_ability(center_key)

        # Restore preserved fields
        self.edition = old_edition
        self.seal = old_seal
        self.ability["perma_bonus"] = old_perma_bonus
        self.ability["bonus"] = old_bonus

    def change_suit(self, new_suit: str) -> None:
        """Change suit while preserving rank. Matches card.lua:547.

        Recalculates suit_nominal and suit_nominal_original. Preserves
        rank, enhancement, edition, seal.
        """
        if self.base is None:
            return
        rank_str = self.base.rank.value
        suit_letter = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
        rank_letter = {
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "10": "T",
            "Jack": "J",
            "Queen": "Q",
            "King": "K",
            "Ace": "A",
        }
        card_key = f"{suit_letter[new_suit]}_{rank_letter[rank_str]}"
        self.set_base(card_key, new_suit, rank_str)

    def change_rank(self, new_rank: str) -> None:
        """Change rank while preserving suit. Matches Strength tarot logic.

        Recalculates id, nominal, face_nominal. Preserves suit,
        enhancement, edition, seal.
        """
        if self.base is None:
            return
        suit_str = self.base.suit.value
        suit_letter = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
        rank_letter = {
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "10": "T",
            "Jack": "J",
            "Queen": "Q",
            "King": "K",
            "Ace": "A",
        }
        card_key = f"{suit_letter[suit_str]}_{rank_letter[new_rank]}"
        self.set_base(card_key, suit_str, new_rank)

    def set_edition(self, edition: dict[str, bool] | None) -> None:
        """Set the card's edition, populating scoring values.

        Matches card.lua:387 (Card:set_edition).  The source stores both
        the boolean flag (``foil=True``) and the scoring value (``chips=50``)
        on ``self.edition``.
        """
        self.edition = None
        if not edition:
            return
        self.edition = {}
        if edition.get("foil"):
            self.edition["foil"] = True
            self.edition["chips"] = 50
            self.edition["type"] = "foil"
        elif edition.get("holo"):
            self.edition["holo"] = True
            self.edition["mult"] = 10
            self.edition["type"] = "holo"
        elif edition.get("polychrome"):
            self.edition["polychrome"] = True
            self.edition["x_mult"] = 1.5
            self.edition["type"] = "polychrome"
        elif edition.get("negative"):
            self.edition["negative"] = True
            self.edition["type"] = "negative"

    def set_seal(self, seal: str | None) -> None:
        """Set the card's seal."""
        self.seal = seal

    def set_eternal(self, eternal: bool) -> None:
        self.eternal = eternal

    def set_perishable(self, perishable: bool) -> None:
        self.perishable = perishable
        if perishable:
            self.perish_tally = 5

    def set_rental(self, rental: bool) -> None:
        self.rental = rental

    def set_debuff(self, should_debuff: bool) -> None:
        self.debuff = should_debuff

    def set_cost(
        self,
        *,
        inflation: int = 0,
        discount_percent: int = 0,
        ante: int = 1,
        booster_ante_scaling: bool = False,
        has_astronomer: bool = False,
        is_couponed: bool = False,
    ) -> None:
        """Calculate cost and sell_cost, matching card.lua:369 (Card:set_cost).

        Takes game-state values as parameters rather than reading global state,
        keeping Card independent of the game loop.

        Args:
            inflation: ``G.GAME.inflation`` — cumulative price inflation.
            discount_percent: ``G.GAME.discount_percent`` — 0/25/50 from vouchers.
            ante: Current ante (for booster ante scaling).
            booster_ante_scaling: ``G.GAME.modifiers.booster_ante_scaling``.
            has_astronomer: Whether the Astronomer joker is active.
            is_couponed: Whether a tag set ``ability.couponed = true``.
        """
        import math

        # Edition surcharge
        edition_extra = 0
        if self.edition:
            edition_extra += 2 if self.edition.get("foil") else 0
            edition_extra += 3 if self.edition.get("holo") else 0
            edition_extra += 5 if self.edition.get("polychrome") else 0
            edition_extra += 5 if self.edition.get("negative") else 0

        self.extra_cost = inflation + edition_extra

        # Base formula (card.lua:375)
        self.cost = max(
            1,
            math.floor((self.base_cost + self.extra_cost + 0.5) * (100 - discount_percent) / 100),
        )

        # Booster ante scaling (card.lua:376)
        if self.ability.get("set") == "Booster" and booster_ante_scaling:
            self.cost += ante - 1

        # Astronomer: planets and celestial packs cost 0 (card.lua:380)
        if has_astronomer:
            ability_set = self.ability.get("set", "")
            ability_name = self.ability.get("name", "")
            if ability_set == "Planet" or (
                ability_set == "Booster" and "Celestial" in ability_name
            ):
                self.cost = 0

        # Rental override (card.lua:381)
        if self.ability.get("rental") or self.rental:
            self.cost = 1

        # Sell price (card.lua:382)
        extra_value = self.ability.get("extra_value", 0)
        self.sell_cost = max(1, math.floor(self.cost / 2)) + extra_value

        # Couponed by tag: cost 0 (card.lua:383)
        if is_couponed:
            self.cost = 0

    def is_face(
        self,
        *,
        from_boss: bool = False,
        pareidolia: bool = False,
    ) -> bool:
        """Check if this is a face card (J/Q/K), matching Card:is_face (card.lua:964).

        Args:
            from_boss: If True, ignore debuff (boss blinds check face status
                even on debuffed cards, e.g. The Plant).
            pareidolia: If True, ALL cards count as face cards (Pareidolia
                joker active).
        """
        if self.debuff and not from_boss:
            return False
        if self.base is None:
            return False
        if pareidolia:
            return True
        return self.base.id in (11, 12, 13)

    def is_suit(
        self,
        suit: str,
        *,
        bypass_debuff: bool = False,
        flush_calc: bool = False,
        smeared: bool = False,
    ) -> bool:
        """Check if this card matches *suit*, matching Card:is_suit (card.lua:4064).

        Args:
            suit: Target suit string (``"Spades"``, ``"Hearts"``, etc.)
                or a Suit enum value.
            bypass_debuff: If True, ignore debuff status.
            flush_calc: If True, use flush-specific rules (Stone excluded,
                Wild matches all regardless of debuff).
            smeared: If True, red suits interchangeable, black interchangeable.
        """
        if self.base is None:
            return False

        suit_str = suit.value if hasattr(suit, "value") else suit
        effect = self.ability.get("effect", "")
        card_suit = self.base.suit.value

        if flush_calc:
            if effect == "Stone Card":
                return False
            if self.ability.get("name") == "Wild Card" and not self.debuff:
                return True
            if smeared:
                target_red = suit_str in ("Hearts", "Diamonds")
                card_red = card_suit in ("Hearts", "Diamonds")
                if target_red == card_red:
                    return True
            return card_suit == suit_str
        else:
            if self.debuff and not bypass_debuff:
                return False
            if effect == "Stone Card":
                return False
            if self.ability.get("name") == "Wild Card":
                return True
            if smeared:
                target_red = suit_str in ("Hearts", "Diamonds")
                card_red = card_suit in ("Hearts", "Diamonds")
                if target_red == card_red:
                    return True
            return card_suit == suit_str

    def get_id(self) -> int:
        """Get the rank id for hand evaluation, matching Card:get_id.

        Stone Cards return a random negative number in Lua; here we
        return -1 as a deterministic placeholder (the actual randomness
        is handled at the game logic layer).
        """
        if self.ability.get("effect") == "Stone Card":
            return -1
        if self.base is None:
            return 0
        return self.base.id

    # -- scoring methods (card.lua:976-1089) --------------------------------

    def get_nominal(self, mod: str | None = None) -> float:
        """Sorting key for hand ordering, matching Card:get_nominal (card.lua:950).

        Returns a float combining base nominal, suit tiebreaker, face nominal,
        and a unique micro-tiebreaker so every card has a distinct sort value.
        """
        if self.base is None:
            return 0.0
        mult = 1.0
        if mod == "suit":
            mult = 1000.0
        if self.ability.get("effect") == "Stone Card":
            mult = -1000.0
        unique_val = 1.0 - self.sort_id / 1603301.0
        return (
            self.base.nominal
            + self.base.suit_nominal * mult
            + (self.base.suit_nominal_original or 0.0) * 0.0001 * mult
            + self.base.face_nominal
            + 0.000001 * unique_val
        )

    def get_chip_bonus(self) -> int:
        """Chip bonus when scored, matching Card:get_chip_bonus (card.lua:976).

        Stone Card: ignores base nominal, returns bonus + perma_bonus only.
        Normal card: base.nominal + bonus + perma_bonus.
        """
        if self.debuff:
            return 0
        if self.ability.get("effect") == "Stone Card":
            return self.ability.get("bonus", 0) + self.ability.get("perma_bonus", 0)
        if self.base is None:
            return 0
        return self.base.nominal + self.ability.get("bonus", 0) + self.ability.get("perma_bonus", 0)

    def get_chip_mult(
        self,
        *,
        rng: object | None = None,
        probabilities_normal: float = 1.0,
    ) -> float:
        """Additive mult when scored, matching Card:get_chip_mult (card.lua:984).

        Lucky Card: probabilistic — ``pseudorandom('lucky_mult') < normal/5``.
        If *rng* is provided, does the actual roll (simulation mode).
        If *rng* is None, returns the non-lucky value (ability.mult for
        non-Lucky, 0 for Lucky — caller handles EV separately).

        Mult Card: returns ability.mult (default 4).
        All others: returns ability.mult (typically 0).
        """
        if self.debuff:
            return 0
        if self.ability.get("set") == "Joker":
            return 0
        if self.ability.get("effect") == "Lucky Card":
            if rng is not None:
                from jackdaw.engine.rng import PseudoRandom

                if isinstance(rng, PseudoRandom):
                    roll = rng.random("lucky_mult")
                    if roll < probabilities_normal / 5:
                        self.ability["lucky_trigger"] = True
                        return self.ability.get("mult", 0)
                return 0
            return 0  # Without RNG, Lucky Card returns 0 (needs actual roll)
        return self.ability.get("mult", 0)

    def get_chip_x_mult(self) -> float:
        """Multiplicative mult when scored, matching Card:get_chip_x_mult (card.lua:999).

        Glass Card: returns ability.x_mult (default 2.0) if > 1.
        All others: returns 0 (no multiplicative effect).
        """
        if self.debuff:
            return 0
        if self.ability.get("set") == "Joker":
            return 0
        xm = self.ability.get("x_mult", 1)
        if xm <= 1:
            return 0
        return xm

    def get_chip_h_mult(self) -> float:
        """Additive mult for a card HELD in hand, matching Card:get_chip_h_mult (card.lua:1006).

        Returns ability.h_mult (typically 0 for most cards).
        No standard enhancement uses this directly — it's available for
        modded effects or future use.
        """
        if self.debuff:
            return 0
        return self.ability.get("h_mult", 0)

    def get_chip_h_x_mult(self) -> float:
        """Multiplicative mult for held card (Card:get_chip_h_x_mult, card.lua:1011).

        Steel Card: returns ability.h_x_mult (default 1.5).
        """
        if self.debuff:
            return 0
        return self.ability.get("h_x_mult", 0)

    def get_edition(self) -> dict | None:
        """Edition scoring bonuses, matching Card:get_edition (card.lua:1016).

        Returns a dict with scoring fields, or None if debuffed/no edition.
        Fields: chip_mod (Foil=50), mult_mod (Holo=10), x_mult_mod (Poly=1.5).
        """
        if self.debuff:
            return None
        if not self.edition:
            return None
        ret: dict = {"card": self}
        if self.edition.get("x_mult"):
            ret["x_mult_mod"] = self.edition["x_mult"]
        if self.edition.get("mult"):
            ret["mult_mod"] = self.edition["mult"]
        if self.edition.get("chips"):
            ret["chip_mod"] = self.edition["chips"]
        return ret

    def get_p_dollars(
        self,
        *,
        rng: object | None = None,
        probabilities_normal: float = 1.0,
    ) -> int:
        """Dollars earned when this card scores, matching Card:get_p_dollars (card.lua:1068).

        Gold Seal: +3.
        Gold Card (enhancement): ability.p_dollars (0 — wait, Gold Card uses h_dollars).
        Lucky Card: probabilistic — ``pseudorandom('lucky_money') < normal/15 → +20``.
        """
        if self.debuff:
            return 0
        ret = 0
        # Gold Seal bonus
        if self.seal == "Gold":
            ret += 3
        # Card ability dollars
        p_dollars = self.ability.get("p_dollars", 0)
        if p_dollars > 0:
            if self.ability.get("effect") == "Lucky Card":
                if rng is not None:
                    from jackdaw.engine.rng import PseudoRandom

                    if isinstance(rng, PseudoRandom):
                        roll = rng.random("lucky_money")
                        if roll < probabilities_normal / 15:
                            self.ability["lucky_trigger"] = True
                            ret += p_dollars
                # Without RNG, Lucky Card $ returns 0
            else:
                ret += p_dollars
        return ret

    def calculate_seal(self, *, repetition: bool = False) -> dict | None:
        """Seal effects, matching Card:calculate_seal (card.lua:2242).

        For the repetition context:
        - Red Seal: returns ``{'repetitions': 1}`` (one extra evaluation).
        - Other seals have non-repetition effects handled elsewhere.
        """
        if self.debuff:
            return None
        if repetition:
            if self.seal == "Red":
                return {"repetitions": 1, "card": self}
        return None

    def add_to_deck(self, game_state: dict) -> None:
        """Apply joker's passive effects when added to deck (card.lua:Card:add_to_deck).

        Mutates *game_state* in-place.  Matches card.lua:564 exactly.
        game_state keys: hand_size, discards, joker_slots, probabilities_normal,
        bankrupt_at, free_rerolls, hands_per_round, interest_amount.
        """
        name = self.ability.get("name", "")
        extra = self.ability.get("extra")

        # Top-level h_size / d_size (Juggler, Merry Andy, Drunkard, etc.)
        if self.ability.get("h_size", 0) != 0:
            game_state["hand_size"] = game_state.get("hand_size", 0) + self.ability["h_size"]
        if self.ability.get("d_size", 0) > 0:
            rr = game_state.get("round_resets")
            if rr is not None:
                rr["discards"] = rr.get("discards", 0) + self.ability["d_size"]

        if name == "Credit Card":
            amount = extra if isinstance(extra, int) else 0
            game_state["bankrupt_at"] = game_state.get("bankrupt_at", 0) - amount
        if name == "Chaos the Clown":
            game_state["free_rerolls"] = game_state.get("free_rerolls", 0) + 1
        if name == "Turtle Bean" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) + extra.get("h_size", 0)
        if name == "Oops! All 6s":
            game_state["probabilities_normal"] = game_state.get("probabilities_normal", 1) * 2
        if name == "To the Moon":
            amount = extra if isinstance(extra, int) else 0
            game_state["interest_amount"] = game_state.get("interest_amount", 0) + amount
        if name == "Troubadour" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) + extra.get("h_size", 0)
            rr = game_state.get("round_resets")
            if rr is not None:
                rr["hands"] = rr.get("hands", 0) + extra.get("h_plays", 0)
        if name == "Stuntman" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) - extra.get("h_size", 0)

        if self.edition and self.edition.get("negative"):
            game_state["joker_slots"] = game_state.get("joker_slots", 0) + 1

    def remove_from_deck(self, game_state: dict) -> None:
        """Reverse joker's passive effects when removed from deck (card.lua:Card:remove_from_deck).

        Mirrors :meth:`add_to_deck` — each effect is undone.  Matches card.lua:648.
        """
        name = self.ability.get("name", "")
        extra = self.ability.get("extra")

        if self.ability.get("h_size", 0) != 0:
            game_state["hand_size"] = game_state.get("hand_size", 0) - self.ability["h_size"]
        if self.ability.get("d_size", 0) > 0:
            rr = game_state.get("round_resets")
            if rr is not None:
                rr["discards"] = rr.get("discards", 0) - self.ability["d_size"]

        if name == "Credit Card":
            amount = extra if isinstance(extra, int) else 0
            game_state["bankrupt_at"] = game_state.get("bankrupt_at", 0) + amount
        if name == "Chaos the Clown":
            game_state["free_rerolls"] = max(0, game_state.get("free_rerolls", 0) - 1)
        if name == "Turtle Bean" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) - extra.get("h_size", 0)
        if name == "Oops! All 6s":
            game_state["probabilities_normal"] = max(
                1, game_state.get("probabilities_normal", 1) // 2
            )
        if name == "To the Moon":
            amount = extra if isinstance(extra, int) else 0
            game_state["interest_amount"] = game_state.get("interest_amount", 0) - amount
        if name == "Troubadour" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) - extra.get("h_size", 0)
            rr = game_state.get("round_resets")
            if rr is not None:
                rr["hands"] = rr.get("hands", 0) - extra.get("h_plays", 0)
        if name == "Stuntman" and isinstance(extra, dict):
            game_state["hand_size"] = game_state.get("hand_size", 0) + extra.get("h_size", 0)

        if self.edition and self.edition.get("negative"):
            game_state["joker_slots"] = game_state.get("joker_slots", 0) - 1

    def __repr__(self) -> str:
        if self.base:
            return f"Card({self.base.rank} of {self.base.suit}, center={self.center_key!r})"
        return f"Card(center={self.center_key!r}, name={self.ability.get('name', '?')!r})"
