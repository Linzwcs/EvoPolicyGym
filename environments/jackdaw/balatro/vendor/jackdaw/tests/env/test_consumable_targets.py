"""Tests for the consumable targeting system.

Covers:
- Every tarot card's targeting spec matches actual behavior
- Planet cards all return needs_card_targets=False
- Aura correctly filters out cards with editions
- Death requires exactly 2 targets
- Strength allows 1-2 targets (config: max_highlighted=2)
- Enhancement tarots accept 1-2 or exactly 1 targets
- Suit-change tarots accept 1-3 targets
- Spectrals match their documented targeting
- Seal spectrals require exactly 1 target
- No-target consumables reject card selections
- Validation: bounds, duplicates, filter checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jackdaw.env.consumable_targets import (
    get_consumable_target_spec,
    get_valid_target_cards,
    validate_card_targets,
)

# ---------------------------------------------------------------------------
# Lightweight mock Card (matches test_action_space.py pattern)
# ---------------------------------------------------------------------------


@dataclass
class MockCard:
    """Minimal mock card for testing targeting logic."""

    center_key: str = "c_base"
    ability: dict[str, Any] = field(default_factory=dict)
    edition: dict[str, bool] | None = None
    debuff: bool = False
    eternal: bool = False
    cost: int = 0
    sell_cost: int = 0
    base: Any = None
    seal: str | None = None
    sort_id: int = 0


def _make_hand(n: int = 5) -> list[MockCard]:
    """Create n hand cards (playing cards)."""
    return [MockCard(center_key="c_base", sort_id=i) for i in range(n)]


def _make_hand_with_editions(editions: list[dict | None]) -> list[MockCard]:
    """Create hand cards with specified editions (None = no edition)."""
    return [MockCard(center_key="c_base", sort_id=i, edition=ed) for i, ed in enumerate(editions)]


def _make_consumable(key: str, cfg: dict | None = None) -> MockCard:
    """Create a consumable with optional config."""
    ability: dict[str, Any] = {"set": "Tarot"}
    if cfg is not None:
        ability["consumeable"] = cfg
    return MockCard(center_key=key, ability=ability)


def _make_planet(key: str = "c_mercury") -> MockCard:
    """Create a planet consumable."""
    return MockCard(
        center_key=key,
        ability={"set": "Planet", "consumeable": {"hand_type": "Pair"}},
    )


# =========================================================================
# Tests: Enhancement tarots (1-2 targets)
# =========================================================================


class TestEnhancementTarots:
    """Enhancement tarots use max_highlighted from config."""

    @pytest.mark.parametrize(
        "key,max_h",
        [
            ("c_magician", 2),
            ("c_empress", 2),
            ("c_heirophant", 2),
        ],
    )
    def test_two_target_enhancement(self, key: str, max_h: int):
        """Enhancement tarots with max_highlighted=2 accept 1-2 cards."""
        card = _make_consumable(key, {"max_highlighted": max_h, "mod_conv": "m_lucky"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2
        assert spec.exact_targets is None  # 1 or 2 is fine
        assert spec.target_filter == "any"

    @pytest.mark.parametrize(
        "key",
        ["c_lovers", "c_chariot", "c_justice", "c_devil", "c_tower"],
    )
    def test_single_target_enhancement(self, key: str):
        """Enhancement tarots with max_highlighted=1 accept exactly 1 card."""
        card = _make_consumable(key, {"max_highlighted": 1, "mod_conv": "m_wild"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1
        assert spec.target_filter == "any"


# =========================================================================
# Tests: Suit-change tarots (1-3 targets)
# =========================================================================


class TestSuitChangeTarots:
    @pytest.mark.parametrize("key", ["c_star", "c_moon", "c_sun", "c_world"])
    def test_suit_change_up_to_three(self, key: str):
        """Suit-change tarots accept 1-3 cards."""
        card = _make_consumable(key, {"max_highlighted": 3, "suit_conv": "Diamonds"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 3
        assert spec.exact_targets is None


# =========================================================================
# Tests: Transformation tarots
# =========================================================================


class TestTransformationTarots:
    def test_strength_one_to_two(self):
        """Strength: config has max_highlighted=2, no min → 1-2 cards."""
        card = _make_consumable("c_strength", {"max_highlighted": 2, "mod_conv": "up_rank"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2

    def test_death_exactly_two(self):
        """Death: min_highlighted=2, max_highlighted=2 → exactly 2."""
        card = _make_consumable(
            "c_death", {"max_highlighted": 2, "min_highlighted": 2, "mod_conv": "card"}
        )
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 2
        assert spec.max_targets == 2
        assert spec.exact_targets == 2

    def test_hanged_man_one_to_two(self):
        """Hanged Man: max_highlighted=2, destroys 1-2 cards."""
        card = _make_consumable("c_hanged_man", {"max_highlighted": 2, "remove_card": True})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 2


# =========================================================================
# Tests: Planet cards (no targets)
# =========================================================================


class TestPlanetCards:
    @pytest.mark.parametrize(
        "key",
        [
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
        ],
    )
    def test_planet_no_targets(self, key: str):
        """All planet cards need no card targets."""
        card = _make_planet(key)
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets
        assert spec.min_targets == 0
        assert spec.max_targets == 0

    def test_black_hole_no_targets(self):
        """Black Hole (levels all hands) needs no targets."""
        card = MockCard(center_key="c_black_hole", ability={"set": "Spectral"})
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets


# =========================================================================
# Tests: No-target tarots/spectrals
# =========================================================================


class TestNoTargetConsumables:
    @pytest.mark.parametrize(
        "key",
        [
            "c_hermit",
            "c_temperance",
            "c_wheel_of_fortune",
            "c_fool",
            "c_emperor",
            "c_high_priestess",
            "c_judgement",
            "c_soul",
            "c_wraith",
            "c_ectoplasm",
            "c_hex",
            "c_ankh",
        ],
    )
    def test_no_card_targets(self, key: str):
        """These consumables don't target hand cards."""
        card = MockCard(center_key=key, ability={"set": "Tarot"})
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets
        assert spec.min_targets == 0

    @pytest.mark.parametrize(
        "key",
        ["c_familiar", "c_grim", "c_incantation", "c_immolate"],
    )
    def test_destroy_spectrals_no_selection(self, key: str):
        """Destroy/create spectrals need hand cards but no player selection."""
        card = MockCard(center_key=key, ability={"set": "Spectral"})
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets


# =========================================================================
# Tests: Aura (edition filter)
# =========================================================================


class TestAura:
    def test_aura_spec(self):
        """Aura targets exactly 1 card with no edition."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1
        assert spec.target_filter == "no_edition"

    def test_aura_filters_editions(self):
        """Aura only targets cards without editions."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand_with_editions([None, {"foil": True}, None, {"holo": True}, None])
        valid = get_valid_target_cards(card, hand)
        assert valid == [0, 2, 4]

    def test_aura_all_have_editions(self):
        """No valid targets when all hand cards have editions."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand_with_editions([{"foil": True}, {"holo": True}, {"polychrome": True}])
        valid = get_valid_target_cards(card, hand)
        assert valid == []

    def test_aura_no_editions_all_valid(self):
        """All cards valid when none have editions."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand(4)
        valid = get_valid_target_cards(card, hand)
        assert valid == [0, 1, 2, 3]


# =========================================================================
# Tests: Sigil and Ouija (all hand cards, no selection)
# =========================================================================


class TestSigilOuija:
    @pytest.mark.parametrize("key", ["c_sigil", "c_ouija"])
    def test_targets_all_no_selection(self, key: str):
        """Sigil/Ouija affect all hand cards — no player selection needed."""
        card = MockCard(center_key=key, ability={"set": "Spectral"})
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets
        assert spec.target_filter == "all_hand"

    @pytest.mark.parametrize("key", ["c_sigil", "c_ouija"])
    def test_valid_targets_empty(self, key: str):
        """get_valid_target_cards returns [] since no selection is needed."""
        card = MockCard(center_key=key, ability={"set": "Spectral"})
        hand = _make_hand(5)
        valid = get_valid_target_cards(card, hand)
        assert valid == []


# =========================================================================
# Tests: Seal spectrals (1 target)
# =========================================================================


class TestSealSpectrals:
    @pytest.mark.parametrize("key", ["c_talisman", "c_deja_vu", "c_trance", "c_medium"])
    def test_seal_spectral_one_target(self, key: str):
        """Seal spectrals target exactly 1 card (max_highlighted=1 in config)."""
        card = _make_consumable(key, {"max_highlighted": 1, "extra": "Gold"})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1


# =========================================================================
# Tests: Cryptid (1 target)
# =========================================================================


class TestCryptid:
    def test_cryptid_one_target(self):
        """Cryptid: max_highlighted=1 in config → exactly 1 card."""
        card = _make_consumable("c_cryptid", {"max_highlighted": 1, "extra": 2})
        spec = get_consumable_target_spec(card)
        assert spec.needs_card_targets
        assert spec.min_targets == 1
        assert spec.max_targets == 1
        assert spec.exact_targets == 1


# =========================================================================
# Tests: get_valid_target_cards
# =========================================================================


class TestGetValidTargetCards:
    def test_enhancement_tarot_all_valid(self):
        """Enhancement tarots accept any hand card."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(5)
        valid = get_valid_target_cards(card, hand)
        assert valid == [0, 1, 2, 3, 4]

    def test_planet_no_valid_targets(self):
        """Planets return no valid targets."""
        card = _make_planet()
        hand = _make_hand(5)
        valid = get_valid_target_cards(card, hand)
        assert valid == []

    def test_empty_hand(self):
        """Empty hand returns no valid targets."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        valid = get_valid_target_cards(card, [])
        assert valid == []


# =========================================================================
# Tests: validate_card_targets
# =========================================================================


class TestValidateCardTargets:
    def test_valid_single_target(self):
        """Single target within range for max_highlighted=1."""
        card = _make_consumable("c_chariot", {"max_highlighted": 1})
        hand = _make_hand(5)
        assert validate_card_targets(card, (2,), hand)

    def test_valid_two_targets(self):
        """Two targets for max_highlighted=2."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(5)
        assert validate_card_targets(card, (1, 3), hand)

    def test_valid_death_two_targets(self):
        """Death requires exactly 2."""
        card = _make_consumable("c_death", {"max_highlighted": 2, "min_highlighted": 2})
        hand = _make_hand(5)
        assert validate_card_targets(card, (0, 4), hand)

    def test_death_one_target_invalid(self):
        """Death with 1 target is invalid (min=2)."""
        card = _make_consumable("c_death", {"max_highlighted": 2, "min_highlighted": 2})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (0,), hand)

    def test_too_many_targets(self):
        """3 targets for max_highlighted=2 is invalid."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (0, 1, 2), hand)

    def test_zero_targets_when_required(self):
        """Empty selection for a targeting consumable is invalid."""
        card = _make_consumable("c_chariot", {"max_highlighted": 1})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (), hand)

    def test_no_target_consumable_empty_ok(self):
        """No-target consumable accepts empty selection."""
        card = _make_planet()
        hand = _make_hand(5)
        assert validate_card_targets(card, (), hand)

    def test_no_target_consumable_rejects_targets(self):
        """No-target consumable rejects non-empty selection."""
        card = _make_planet()
        hand = _make_hand(5)
        assert not validate_card_targets(card, (0,), hand)

    def test_index_out_of_bounds(self):
        """Target index beyond hand size is invalid."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(3)
        assert not validate_card_targets(card, (5,), hand)

    def test_negative_index(self):
        """Negative index is invalid."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (-1,), hand)

    def test_duplicate_indices(self):
        """Duplicate indices are invalid."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (2, 2), hand)

    def test_aura_valid_no_edition(self):
        """Aura accepts a card without edition."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand_with_editions([None, {"foil": True}, None])
        assert validate_card_targets(card, (0,), hand)
        assert validate_card_targets(card, (2,), hand)

    def test_aura_rejects_edition_card(self):
        """Aura rejects targeting a card with edition."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand_with_editions([None, {"foil": True}, None])
        assert not validate_card_targets(card, (1,), hand)

    def test_aura_rejects_two_targets(self):
        """Aura only accepts exactly 1 target."""
        card = MockCard(center_key="c_aura", ability={"set": "Spectral"})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (0, 1), hand)

    def test_suit_change_three_targets(self):
        """Suit-change tarots accept up to 3."""
        card = _make_consumable("c_star", {"max_highlighted": 3})
        hand = _make_hand(5)
        assert validate_card_targets(card, (0, 2, 4), hand)

    def test_suit_change_four_invalid(self):
        """4 targets for suit-change tarot is invalid."""
        card = _make_consumable("c_star", {"max_highlighted": 3})
        hand = _make_hand(5)
        assert not validate_card_targets(card, (0, 1, 2, 3), hand)


# =========================================================================
# Tests: Edge case — pack opening phase
# =========================================================================


class TestPackOpeningEdgeCase:
    def test_consumable_in_pack_no_hand(self):
        """Consumable from a pack with no hand cards: empty valid targets."""
        card = _make_consumable("c_magician", {"max_highlighted": 2})
        valid = get_valid_target_cards(card, [])
        assert valid == []
        # Validation also fails since there are no valid targets
        assert not validate_card_targets(card, (0,), [])

    def test_planet_in_pack(self):
        """Planet from a pack always has no targets."""
        card = _make_planet()
        spec = get_consumable_target_spec(card)
        assert not spec.needs_card_targets
        assert validate_card_targets(card, (), [])
