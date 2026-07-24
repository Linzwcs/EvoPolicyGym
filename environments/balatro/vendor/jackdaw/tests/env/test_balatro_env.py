"""Tests for BalatroEnvironment — the GameEnvironment implementation for Balatro.

Covers:
- BalatroEnvironment conforms to GameEnvironment protocol
- reset returns GameObservation, GameActionMask, info with Balatro objects
- step returns correct types and tracks episode stats
- Entity names in observations match balatro_game_spec
- Integration: run a short episode with random valid actions
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from jackdaw.env.action_space import ActionMask, ActionType, FactoredAction
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.game_spec import GameActionMask, GameEnvironment, GameObservation
from jackdaw.env.observation import Observation

SEED = "TEST_BALATRO_ENV_42"


@pytest.fixture()
def env() -> BalatroEnvironment:
    return BalatroEnvironment(
        adapter_factory=DirectAdapter,
        back_keys=["b_red"],
        stakes=[1],
        max_steps=500,
        seed_prefix="TEST",
    )


class TestProtocolConformance:
    """Verify BalatroEnvironment satisfies GameEnvironment protocol."""

    def test_isinstance_check(self, env: BalatroEnvironment) -> None:
        assert isinstance(env, GameEnvironment)

    def test_has_spec(self, env: BalatroEnvironment) -> None:
        spec = env.spec
        assert spec.name == "balatro"
        assert spec.num_entity_types == 5
        assert spec.num_action_types == 21


class TestReset:
    """Test reset returns correct types."""

    def test_returns_game_observation(self, env: BalatroEnvironment) -> None:
        game_obs, game_mask, info = env.reset()
        assert isinstance(game_obs, GameObservation)
        assert isinstance(game_mask, GameActionMask)

    def test_observation_entity_names(self, env: BalatroEnvironment) -> None:
        game_obs, _, _ = env.reset()
        spec = env.spec
        expected_names = {et.name for et in spec.entity_types}
        assert set(game_obs.entities.keys()) == expected_names

    def test_observation_shapes(self, env: BalatroEnvironment) -> None:
        game_obs, _, _ = env.reset()
        spec = env.spec
        assert game_obs.global_context.shape == (spec.global_feature_dim,)
        for et in spec.entity_types:
            arr = game_obs.entities[et.name]
            assert arr.ndim == 2
            assert arr.shape[1] == et.feature_dim

    def test_info_contains_balatro_objects(self, env: BalatroEnvironment) -> None:
        _, _, info = env.reset()
        assert "raw_state" in info
        assert "observation" in info
        assert "action_mask" in info
        assert "shop_splits" in info
        assert isinstance(info["observation"], Observation)
        assert isinstance(info["action_mask"], ActionMask)

    def test_mask_type_mask_shape(self, env: BalatroEnvironment) -> None:
        _, game_mask, _ = env.reset()
        assert game_mask.type_mask.shape == (21,)

    def test_episode_tracking_reset(self, env: BalatroEnvironment) -> None:
        env.reset()
        assert env.episode_length == 0
        assert env.episode_won is False


class TestStep:
    """Test step returns correct types and tracks state."""

    def test_step_returns_correct_types(self, env: BalatroEnvironment) -> None:
        game_obs, game_mask, info = env.reset()
        # Find a valid action
        action = _pick_valid_action(game_mask, info)
        result = env.step(action)
        assert len(result) == 5
        obs2, terminated, truncated, mask2, info2 = result
        assert isinstance(obs2, GameObservation)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(mask2, GameActionMask)

    def test_episode_length_increments(self, env: BalatroEnvironment) -> None:
        game_obs, game_mask, info = env.reset()
        action = _pick_valid_action(game_mask, info)
        env.step(action)
        assert env.episode_length == 1


class TestIntegration:
    """Run a short episode to verify everything works end-to-end."""

    def test_random_episode(self, env: BalatroEnvironment) -> None:
        rng = random.Random(42)
        game_obs, game_mask, info = env.reset()
        steps = 0
        max_steps = 200

        while steps < max_steps:
            action = _pick_valid_action(game_mask, info, rng)
            game_obs, terminated, truncated, game_mask, info = env.step(action)
            steps += 1

            # Verify observation consistency
            spec = env.spec
            assert game_obs.global_context.shape == (spec.global_feature_dim,)
            for et in spec.entity_types:
                arr = game_obs.entities[et.name]
                assert arr.shape[1] == et.feature_dim
                assert not np.any(np.isnan(arr))
                assert not np.any(np.isinf(arr))

            if terminated or truncated:
                break

        assert env.episode_length == steps
        assert steps > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_valid_action(
    game_mask: GameActionMask,
    info: dict[str, object],
    rng: random.Random | None = None,
) -> FactoredAction:
    """Pick a random valid action from the mask."""
    if rng is None:
        rng = random.Random(0)

    action_mask: ActionMask = info["action_mask"]  # type: ignore[assignment]
    valid_types = [i for i in range(len(game_mask.type_mask)) if game_mask.type_mask[i]]
    if not valid_types:
        return FactoredAction(action_type=0)

    at = rng.choice(valid_types)

    entity_target = None
    if at in action_mask.entity_masks:
        emask = action_mask.entity_masks[at]
        valid_entities = [i for i in range(len(emask)) if emask[i]]
        if valid_entities:
            entity_target = rng.choice(valid_entities)

    card_target = None
    if at in (ActionType.PlayHand, ActionType.Discard, ActionType.UseConsumable):
        valid_cards = [i for i in range(len(game_mask.card_mask)) if game_mask.card_mask[i]]
        if valid_cards:
            n = rng.randint(
                max(1, action_mask.min_card_select),
                min(len(valid_cards), action_mask.max_card_select),
            )
            card_target = tuple(rng.sample(valid_cards, n))

    return FactoredAction(
        action_type=at,
        card_target=card_target,
        entity_target=entity_target,
    )
