"""Tests for card factory functions.

Verifies card creation from prototypes, control dicts, modifier
application for all card types, and resolve_create/destroy descriptors.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import (
    card_from_control,
    create_card,
    create_consumable,
    create_joker,
    create_playing_card,
    create_voucher,
    resolve_create_descriptor,
    resolve_destroy_descriptor,
)
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import JOKER_RARITY_POOLS, JOKERS, JokerProto
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _gs(**kwargs) -> dict:
    """Build a game_state dict from keyword args."""
    return dict(kwargs)


def _rng(seed: str = "RESOLVER") -> PseudoRandom:
    return PseudoRandom(seed)


# ============================================================================
# create_playing_card
# ============================================================================


class TestCreatePlayingCard:
    def test_with_all_modifiers(self):
        c = create_playing_card(
            Suit.DIAMONDS,
            Rank.JACK,
            enhancement="m_steel",
            edition={"polychrome": True},
            seal="Gold",
        )
        assert c.center_key == "m_steel"
        assert c.edition["polychrome"] is True
        assert c.seal == "Gold"
        assert c.base.rank is Rank.JACK


# ============================================================================
# create_joker
# ============================================================================


class TestCreateJoker:
    def test_basic_joker(self):
        c = create_joker("j_joker")
        assert c.ability["name"] == "Joker"
        assert c.ability["mult"] == 4
        assert c.ability["set"] == "Joker"
        assert c.center_key == "j_joker"
        assert c.base is None
        assert c.base_cost == 2


# ============================================================================
# create_consumable
# ============================================================================


class TestCreateConsumable:
    def test_tarot(self):
        c = create_consumable("c_magician")
        assert c.ability["name"] == "The Magician"
        assert c.ability["set"] == "Tarot"
        assert c.center_key == "c_magician"


# ============================================================================
# create_voucher
# ============================================================================


class TestCreateVoucher:
    def test_overstock(self):
        c = create_voucher("v_overstock_norm")
        assert c.ability["name"] == "Overstock"
        assert c.ability["set"] == "Voucher"
        assert c.center_key == "v_overstock_norm"
        assert c.base_cost == 10


# ============================================================================
# card_from_control
# ============================================================================


class TestCardFromControl:
    def test_full_control(self):
        """All fields: enhancement, edition, seal."""
        c = card_from_control(
            {
                "s": "H",
                "r": "K",
                "e": "m_gold",
                "d": "holo",
                "g": "Red",
            }
        )
        assert c.base.suit is Suit.HEARTS
        assert c.base.rank is Rank.KING
        assert c.center_key == "m_gold"
        assert c.ability["effect"] == "Gold Card"
        assert c.edition["holo"] is True
        assert c.seal == "Red"


# ============================================================================
# Deep copy isolation across factory
# ============================================================================


class TestFactoryIsolation:
    def test_two_jokers_from_same_key(self):
        c1 = create_joker("j_ice_cream")
        c2 = create_joker("j_ice_cream")
        c1.ability["extra"]["chips"] = 0
        assert c2.ability["extra"]["chips"] == 100


# ============================================================================
# create_card — common_events.lua:2082
# ============================================================================


class TestCreateCardKeyDetermination:
    def test_forced_key_bypasses_pool(self):
        rng = PseudoRandom("FORCED_CF_TEST")
        card = create_card("Joker", rng, 1, forced_key="j_joker")
        assert card.center_key == "j_joker"

    def test_known_seed_returns_specific_joker(self):
        """PseudoRandom('CF_JOKER_TEST') + forced_rarity=1 at ante=1 picks j_credit_card."""
        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.center_key == "j_credit_card"

    def test_deterministic_same_seed(self):
        k1 = create_card("Joker", PseudoRandom("DET_CF"), ante=2, forced_rarity=2).center_key
        k2 = create_card("Joker", PseudoRandom("DET_CF"), ante=2, forced_rarity=2).center_key
        assert k1 == k2

    def test_ability_set_matches_card_type(self):
        for ct, expected in (
            ("Joker", "Joker"),
            ("Tarot", "Tarot"),
            ("Planet", "Planet"),
            ("Spectral", "Spectral"),
        ):
            card = create_card(ct, PseudoRandom("SET_TEST"), 1, forced_rarity=1)
            assert card.ability["set"] == expected, (
                f"card_type={ct!r}: expected set={expected!r}, got {card.ability['set']!r}"
            )


class TestCreateCardEternalPerishable:
    """ep_roll thresholds: >0.7 eternal, >0.4 perishable (shared roll, mutually exclusive)."""

    def test_eternal_and_perishable_mutually_exclusive(self):
        for seed in ("CF_JOKER_TEST", "P0", "S15", "FOIL_TEST"):
            card = create_card(
                "Joker",
                PseudoRandom(seed),
                1,
                forced_rarity=1,
                game_state=_gs(
                    enable_eternals_in_shop=True,
                    enable_perishables_in_shop=True,
                ),
            )
            assert not (card.eternal and card.perishable), (
                f"Seed {seed!r}: eternal={card.eternal} and perishable={card.perishable} "
                "both True — should be mutually exclusive"
            )


class TestCreateCardRental:
    """Seed 'S15': ep=0.8357, r=0.8849 (>0.7)."""

    def test_rental_fires_at_stake8(self):
        card = create_card(
            "Joker",
            PseudoRandom("S15"),
            1,
            forced_rarity=1,
            game_state=_gs(enable_rentals_in_shop=True),
        )
        assert card.rental is True


class TestCreateCardEdition:
    """Seed 'CF_JOKER_TEST': edition=negative."""

    def test_negative_edition_applied(self):
        card = create_card("Joker", PseudoRandom("CF_JOKER_TEST"), 1, forced_rarity=1)
        assert card.edition is not None
        assert card.edition.get("negative") is True


class TestCreateCardGameStateFiltering:
    def test_banned_key_not_returned(self):
        """banned_keys in game_state forwarded to pool selection."""
        normal = create_card("Joker", PseudoRandom("BAN_CF"), 1, forced_rarity=1).center_key
        result = create_card(
            "Joker",
            PseudoRandom("BAN_CF"),
            1,
            forced_rarity=1,
            game_state=_gs(banned_keys={normal}),
        ).center_key
        assert result != normal


# ============================================================================
# resolve_create_descriptor
# ============================================================================


class TestResolveCreateTarot:
    def test_bare_tarot_returns_tarot_card(self):
        card = resolve_create_descriptor({"type": "Tarot"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Tarot"

    def test_tarot_forced_key_creates_specific_card(self):
        card = resolve_create_descriptor(
            {"type": "Tarot_Planet", "forced_key": "c_magician", "seed": "fool"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.center_key == "c_magician"


class TestResolveCreateJoker:
    def test_joker_descriptor_returns_joker(self):
        card = resolve_create_descriptor({"type": "Joker"}, _rng(), 1, {})
        assert card is not None
        assert card.ability.get("set") == "Joker"


class TestResolveCreatePlayingCard:
    def test_playing_card_inferred_from_rank_key(self):
        """Descriptor with 'rank' but no 'type' → PlayingCard."""
        card = resolve_create_descriptor(
            {"rank": "Ace", "suit": "Hearts", "enhancement": "m_lucky"},
            _rng(),
            1,
            {},
        )
        assert card is not None
        assert card.base is not None
        assert card.base.rank.value == "Ace"
        assert card.base.suit.value == "Hearts"
        assert card.center_key == "m_lucky"

    def test_playing_card_missing_suit_returns_none(self):
        card = resolve_create_descriptor(
            {"type": "PlayingCard", "rank": "King"},
            _rng(),
            1,
            {},
        )
        assert card is None


class TestResolveCreateUnknown:
    def test_empty_descriptor_returns_none(self):
        card = resolve_create_descriptor({}, _rng(), 1, {})
        assert card is None


# ============================================================================
# resolve_destroy_descriptor
# ============================================================================


class TestResolveDestroyRandomJoker:
    def _make_joker(self, key: str = "j_joker") -> Card:
        return create_joker(key)

    def test_eternal_joker_never_selected(self):
        """With 3 jokers (1 eternal), only the 2 non-eternal are eligible."""
        j_eternal = create_joker("j_joker", eternal=True)
        j1 = self._make_joker("j_greedy_joker")
        j2 = self._make_joker("j_lusty_joker")
        jokers = [j_eternal, j1, j2]

        selected = set()
        for i in range(20):
            result = resolve_destroy_descriptor(
                {"destroy_random_joker": True},
                jokers,
                PseudoRandom(f"MADNESS{i}"),
            )
            assert result is not None
            selected.add(id(result))

        assert id(j_eternal) not in selected, "Eternal joker must never be selected"


# ============================================================================
# Prototype data loading and typed access
# ============================================================================


class TestPrototypeCounts:
    def test_jokers(self):
        assert len(JOKERS) == 150


class TestRarityPools:
    def test_pools_sum_to_total(self):
        total = sum(len(pool) for pool in JOKER_RARITY_POOLS.values())
        assert total == 150


class TestJokerSpotChecks:
    def test_joker_basic(self):
        j = JOKERS["j_joker"]
        assert isinstance(j, JokerProto)
        assert j.name == "Joker"
        assert j.rarity == 1
        assert j.cost == 2
        assert j.order == 1
        assert j.effect == "Mult"
        assert j.config == {"mult": 4}
        assert j.blueprint_compat is True
        assert j.discovered is True
