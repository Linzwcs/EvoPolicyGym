"""Tests for GameSpec abstraction and Balatro spec factory.

Covers:
- GameSpec construction and property computation
- GameSpec validation catches inconsistencies
- balatro_game_spec() matches current hardcoded constants
- Observation.to_game_observation() round-trips correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.env.action_space import NUM_ACTION_TYPES, ActionType
from jackdaw.env.balatro_spec import NEEDS_CARDS, NEEDS_ENTITY, NUM_ENTITY_TYPES
from jackdaw.env.game_spec import (
    ActionTypeSpec,
    EntityTypeSpec,
    GameActionMask,
    GameObservation,
    GameSpec,
)
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
    Observation,
)


class TestGameSpec:
    """Core GameSpec dataclass tests."""

    def test_num_entity_types(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(
                EntityTypeSpec("a", 4, 10, False),
                EntityTypeSpec("b", 8, 5, True, catalog_size=100),
            ),
            action_types=(),
            global_feature_dim=16,
        )
        assert spec.num_entity_types == 2
        assert spec.num_action_types == 0

    def test_needs_entity_set(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(EntityTypeSpec("a", 4, 10, False),),
            action_types=(
                ActionTypeSpec("no_target", False, False),
                ActionTypeSpec("entity_target", True, False, entity_type_index=0),
                ActionTypeSpec("card_target", False, True),
            ),
            global_feature_dim=16,
        )
        assert spec.needs_entity_set == frozenset({1})
        assert spec.needs_cards_set == frozenset({2})

    def test_entity_type_for_action(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(
                EntityTypeSpec("a", 4, 10, False),
                EntityTypeSpec("b", 8, 5, False),
            ),
            action_types=(
                ActionTypeSpec("no_target", False, False),
                ActionTypeSpec("target_b", True, False, entity_type_index=1),
            ),
            global_feature_dim=16,
        )
        assert spec.entity_type_for_action(0) == -1
        assert spec.entity_type_for_action(1) == 1

    def test_validate_missing_entity_type_index(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(EntityTypeSpec("a", 4, 10, False),),
            action_types=(
                ActionTypeSpec("bad", needs_entity_target=True, needs_card_select=False),
            ),
            global_feature_dim=16,
        )
        with pytest.raises(ValueError, match="needs_entity_target=True"):
            spec.validate()

    def test_validate_entity_type_index_out_of_range(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(EntityTypeSpec("a", 4, 10, False),),
            action_types=(ActionTypeSpec("bad", True, False, entity_type_index=5),),
            global_feature_dim=16,
        )
        with pytest.raises(ValueError, match="out of range"):
            spec.validate()

    def test_validate_catalog_size_missing(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(EntityTypeSpec("a", 4, 10, has_catalog_id=True, catalog_size=0),),
            action_types=(),
            global_feature_dim=16,
        )
        with pytest.raises(ValueError, match="has_catalog_id=True"):
            spec.validate()

    def test_frozen(self) -> None:
        spec = GameSpec(
            name="test",
            entity_types=(),
            action_types=(),
            global_feature_dim=16,
        )
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]


class TestBalatroSpec:
    """Verify balatro_game_spec() matches current hardcoded values."""

    @pytest.fixture()
    def spec(self) -> GameSpec:
        from jackdaw.env.balatro_spec import balatro_game_spec

        return balatro_game_spec()

    def test_name(self, spec: GameSpec) -> None:
        assert spec.name == "balatro"

    def test_num_entity_types_matches(self, spec: GameSpec) -> None:
        assert spec.num_entity_types == NUM_ENTITY_TYPES

    def test_num_action_types_matches(self, spec: GameSpec) -> None:
        assert spec.num_action_types == NUM_ACTION_TYPES

    def test_global_feature_dim(self, spec: GameSpec) -> None:
        assert spec.global_feature_dim == D_GLOBAL

    def test_entity_feature_dims(self, spec: GameSpec) -> None:
        dims = {et.name: et.feature_dim for et in spec.entity_types}
        assert dims["hand_card"] == D_PLAYING_CARD
        assert dims["joker"] == D_JOKER
        assert dims["consumable"] == D_CONSUMABLE
        assert dims["shop_item"] == D_SHOP
        assert dims["pack_card"] == D_PLAYING_CARD

    def test_catalog_ids(self, spec: GameSpec) -> None:
        catalog = {et.name: et.has_catalog_id for et in spec.entity_types}
        assert catalog["hand_card"] is False
        assert catalog["joker"] is True
        assert catalog["consumable"] is True
        assert catalog["shop_item"] is True
        assert catalog["pack_card"] is False

        for et in spec.entity_types:
            if et.has_catalog_id:
                assert et.catalog_size == NUM_CENTER_KEYS

    def test_needs_entity_matches_hardcoded(self, spec: GameSpec) -> None:
        assert spec.needs_entity_set == NEEDS_ENTITY

    def test_needs_cards_matches_hardcoded(self, spec: GameSpec) -> None:
        assert spec.needs_cards_set == NEEDS_CARDS

    def test_action_type_count_is_21(self, spec: GameSpec) -> None:
        assert len(spec.action_types) == 21

    def test_action_names_match_enum_order(self, spec: GameSpec) -> None:
        """Verify action type indices line up with the ActionType enum."""
        # Spot-check key action types
        assert spec.action_types[ActionType.PlayHand].name == "play_hand"
        assert spec.action_types[ActionType.Discard].name == "discard"
        assert spec.action_types[ActionType.BuyCard].name == "buy_card"
        assert spec.action_types[ActionType.UseConsumable].name == "use_consumable"
        assert spec.action_types[ActionType.PickPackCard].name == "pick_pack_card"
        assert spec.action_types[ActionType.SortHandSuit].name == "sort_hand_suit"

    def test_validates_successfully(self, spec: GameSpec) -> None:
        # Should not raise
        spec.validate()


class TestGameObservation:
    """Test GameObservation container."""

    def test_construction(self) -> None:
        obs = GameObservation(
            global_context=np.zeros(16, dtype=np.float32),
            entities={"a": np.zeros((3, 4), dtype=np.float32)},
        )
        assert obs.global_context.shape == (16,)
        assert obs.entities["a"].shape == (3, 4)


class TestObservationToGameObservation:
    """Test Observation.to_game_observation() conversion."""

    def test_round_trip(self) -> None:
        obs = Observation(
            global_context=np.ones(D_GLOBAL, dtype=np.float32),
            hand_cards=np.ones((5, D_PLAYING_CARD), dtype=np.float32),
            jokers=np.ones((2, D_JOKER), dtype=np.float32),
            consumables=np.zeros((0, D_CONSUMABLE), dtype=np.float32),
            shop_cards=np.ones((3, D_SHOP), dtype=np.float32),
            pack_cards=np.zeros((0, D_PLAYING_CARD), dtype=np.float32),
        )
        game_obs = obs.to_game_observation()

        assert isinstance(game_obs, GameObservation)
        np.testing.assert_array_equal(game_obs.global_context, obs.global_context)
        np.testing.assert_array_equal(game_obs.entities["hand_card"], obs.hand_cards)
        np.testing.assert_array_equal(game_obs.entities["joker"], obs.jokers)
        np.testing.assert_array_equal(game_obs.entities["consumable"], obs.consumables)
        np.testing.assert_array_equal(game_obs.entities["shop_item"], obs.shop_cards)
        np.testing.assert_array_equal(game_obs.entities["pack_card"], obs.pack_cards)

    def test_entity_names_match_spec(self) -> None:
        """Entity names in to_game_observation() match balatro_game_spec() entity names."""
        from jackdaw.env.balatro_spec import balatro_game_spec

        spec = balatro_game_spec()
        spec_names = {et.name for et in spec.entity_types}

        obs = Observation(
            global_context=np.zeros(D_GLOBAL, dtype=np.float32),
            hand_cards=np.zeros((0, D_PLAYING_CARD), dtype=np.float32),
            jokers=np.zeros((0, D_JOKER), dtype=np.float32),
            consumables=np.zeros((0, D_CONSUMABLE), dtype=np.float32),
            shop_cards=np.zeros((0, D_SHOP), dtype=np.float32),
            pack_cards=np.zeros((0, D_PLAYING_CARD), dtype=np.float32),
        )
        game_obs = obs.to_game_observation()

        assert set(game_obs.entities.keys()) == spec_names


class TestGameActionMask:
    """Test GameActionMask container."""

    def test_construction(self) -> None:
        mask = GameActionMask(
            type_mask=np.ones(21, dtype=np.bool_),
            card_mask=np.ones(5, dtype=np.bool_),
            entity_masks={8: np.ones(3, dtype=np.bool_)},
            min_card_select=1,
            max_card_select=5,
        )
        assert mask.type_mask.shape == (21,)
        assert mask.min_card_select == 1
