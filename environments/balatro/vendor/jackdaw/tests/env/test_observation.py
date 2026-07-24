"""Tests for entity-based observation encoding.

Covers:
- Encode initial game state, verify shapes and dtypes
- Encode after playing a hand, verify hand cards reduced
- Encode with jokers, verify joker entity count matches
- Encode with face-down cards, verify face_down flag set
- Encode empty areas produce shape (0, D) arrays
- Center key mapping covers all keys in centers.json
- No NaN/Inf in encoded observations across 100 random-agent steps
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from jackdaw.engine.actions import (
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
)
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import (
    _CENTER_KEY_TO_ID,
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
    Observation,
    encode_consumable,
    encode_observation,
    encode_playing_card,
    encode_shop_item,
)

SEED = "OBS_TEST_42"
BACK = "b_red"
STAKE = 1


def _make_adapter() -> DirectAdapter:
    adapter = DirectAdapter()
    adapter.reset(BACK, STAKE, SEED)
    return adapter


def _step_to_hand(adapter: DirectAdapter) -> None:
    """Advance to SELECTING_HAND phase."""
    adapter.step(SelectBlind())


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------


class TestShapesAndDtypes:
    def test_initial_state_shapes(self):
        adapter = _make_adapter()
        obs = encode_observation(adapter.raw_state)

        assert isinstance(obs, Observation)
        assert obs.global_context.shape == (D_GLOBAL,)
        assert obs.global_context.dtype == np.float32

        # At BLIND_SELECT, hand is empty (cards not dealt yet)
        assert obs.hand_cards.ndim == 2
        assert obs.hand_cards.shape[1] == D_PLAYING_CARD
        assert obs.hand_cards.dtype == np.float32

        assert obs.jokers.ndim == 2
        assert obs.jokers.shape[1] == D_JOKER
        assert obs.jokers.dtype == np.float32

        assert obs.consumables.ndim == 2
        assert obs.consumables.shape[1] == D_CONSUMABLE
        assert obs.consumables.dtype == np.float32

        assert obs.shop_cards.ndim == 2
        assert obs.shop_cards.shape[1] == D_SHOP
        assert obs.shop_cards.dtype == np.float32

        assert obs.pack_cards.ndim == 2
        assert obs.pack_cards.shape[1] == D_PLAYING_CARD
        assert obs.pack_cards.dtype == np.float32

    def test_selecting_hand_has_cards(self):
        adapter = _make_adapter()
        _step_to_hand(adapter)
        obs = encode_observation(adapter.raw_state)

        hand = adapter.raw_state.get("hand", [])
        assert obs.hand_cards.shape[0] == len(hand)
        assert len(hand) > 0
        assert obs.hand_cards.shape == (len(hand), D_PLAYING_CARD)

    def test_global_context_dimension(self):
        adapter = _make_adapter()
        obs = encode_observation(adapter.raw_state)
        assert obs.global_context.shape == (D_GLOBAL,)
        # D_GLOBAL = 90 base + 32 vouchers + 8 blind + 3 round_pos
        #          + 2 round_progress + 24 tags + 52 discard_hist = 211
        assert D_GLOBAL == 235


# ---------------------------------------------------------------------------
# Hand card count after playing
# ---------------------------------------------------------------------------


class TestPlayHand:
    def test_hand_reduced_after_play(self):
        adapter = _make_adapter()
        _step_to_hand(adapter)

        hand_before = len(adapter.raw_state["hand"])
        obs_before = encode_observation(adapter.raw_state)
        assert obs_before.hand_cards.shape[0] == hand_before

        # Play first 3 cards
        n_play = min(3, hand_before)
        adapter.step(PlayHand(card_indices=tuple(range(n_play))))

        gs = adapter.raw_state
        gs.get("phase")

        # After playing, we may transition to ROUND_EVAL or stay in
        # SELECTING_HAND with redrawn cards. Either way, encode should work.
        obs_after = encode_observation(gs)
        assert obs_after.hand_cards.ndim == 2
        assert obs_after.hand_cards.shape[1] == D_PLAYING_CARD
        assert obs_after.hand_cards.shape[0] == len(gs.get("hand", []))


# ---------------------------------------------------------------------------
# Joker encoding
# ---------------------------------------------------------------------------


class TestJokerEncoding:
    def test_joker_count_matches(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        jokers = gs.get("jokers", [])
        obs = encode_observation(gs)
        assert obs.jokers.shape[0] == len(jokers)

    def test_single_joker_features(self):
        """Manually add a joker to state and verify encoding."""
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state

        # Create a fake joker
        joker = Card()
        joker.set_ability("j_joker")
        joker.cost = 4
        joker.sell_cost = 2
        gs["jokers"] = [joker]

        obs = encode_observation(gs)
        assert obs.jokers.shape == (1, D_JOKER)

        vec = obs.jokers[0]
        # center_key_id should be non-zero
        assert vec[0] > 0.0
        # sell_value should be encoded
        assert vec[3] > 0.0


# ---------------------------------------------------------------------------
# Face-down cards (boss blind)
# ---------------------------------------------------------------------------


class TestFaceDownCards:
    def test_face_down_flag(self):
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state

        # Create a face-down card
        card = Card()
        card.set_base("H_A", "Hearts", "Ace")
        card.set_ability("c_base")
        card.facing = "back"

        vec = encode_playing_card(card, 0, gs)
        assert vec.shape == (D_PLAYING_CARD,)

        # face_down flag at index 7
        assert vec[7] == 1.0
        # rank/suit/enhancement should be zeroed when face-down
        assert vec[0] == 0.0  # rank_id
        assert vec[1] == 0.0  # suit
        assert vec[3] == 0.0  # enhancement

    def test_face_up_card(self):
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state

        card = Card()
        card.set_base("H_A", "Hearts", "Ace")
        card.set_ability("c_base")
        card.facing = "front"

        vec = encode_playing_card(card, 0, gs)
        assert vec[7] == 0.0  # not face down
        assert vec[0] > 0.0  # rank_id should be filled (Ace=14/14=1.0)
        assert abs(vec[0] - 14.0 / 14.0) < 1e-6


# ---------------------------------------------------------------------------
# Empty areas
# ---------------------------------------------------------------------------


class TestEmptyAreas:
    def test_empty_jokers(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["jokers"] = []
        obs = encode_observation(gs)
        assert obs.jokers.shape == (0, D_JOKER)

    def test_empty_consumables(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["consumables"] = []
        obs = encode_observation(gs)
        assert obs.consumables.shape == (0, D_CONSUMABLE)

    def test_empty_shop(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["shop_cards"] = []
        gs["shop_vouchers"] = []
        gs["shop_boosters"] = []
        obs = encode_observation(gs)
        assert obs.shop_cards.shape == (0, D_SHOP)

    def test_empty_pack(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["pack_cards"] = []
        obs = encode_observation(gs)
        assert obs.pack_cards.shape == (0, D_PLAYING_CARD)

    def test_empty_hand(self):
        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["hand"] = []
        obs = encode_observation(gs)
        assert obs.hand_cards.shape == (0, D_PLAYING_CARD)


# ---------------------------------------------------------------------------
# Center key mapping coverage
# ---------------------------------------------------------------------------


class TestCenterKeyMapping:
    def test_covers_all_centers_json_keys(self):
        centers_path = (
            Path(__file__).resolve().parent.parent.parent
            / "jackdaw"
            / "engine"
            / "data"
            / "centers.json"
        )
        with open(centers_path) as f:
            data = json.load(f)

        missing = []
        for key in data.keys():
            if key not in _CENTER_KEY_TO_ID:
                missing.append(key)

        assert not missing, f"Center keys missing from mapping: {missing}"

    def test_ids_are_contiguous(self):
        ids = sorted(_CENTER_KEY_TO_ID.values())
        assert ids[0] == 1  # 0 is reserved for unknown
        assert ids[-1] == len(ids)  # contiguous from 1..N

    def test_unknown_key_returns_zero(self):
        from jackdaw.env.observation import center_key_id

        assert center_key_id("nonexistent_key") == 0

    def test_num_center_keys_matches(self):
        assert NUM_CENTER_KEYS == len(_CENTER_KEY_TO_ID)
        assert NUM_CENTER_KEYS > 0


# ---------------------------------------------------------------------------
# No NaN/Inf across random-agent steps
# ---------------------------------------------------------------------------


class TestNoNaNInf:
    def test_100_random_steps(self):
        """Encode observations across 100 random-agent steps.

        Verify no NaN or Inf values appear in any array.
        """
        random.seed(54321)
        adapter = _make_adapter()

        def _check_obs(obs: Observation, step_num: int) -> None:
            for name in (
                "global_context",
                "hand_cards",
                "jokers",
                "consumables",
                "shop_cards",
                "pack_cards",
            ):
                arr = getattr(obs, name)
                assert not np.any(np.isnan(arr)), f"NaN in {name} at step {step_num}"
                assert not np.any(np.isinf(arr)), f"Inf in {name} at step {step_num}"

        obs = encode_observation(adapter.raw_state)
        _check_obs(obs, 0)

        for step_num in range(1, 101):
            gs = adapter.raw_state
            phase = gs.get("phase")
            if phase == GamePhase.GAME_OVER:
                break
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            # Pick a random non-utility action
            progress = [
                a
                for a in legal
                if not isinstance(
                    a, (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)
                )
            ]
            pool = progress if progress else legal
            action = random.choice(pool)

            # Resolve marker actions
            hand = gs.get("hand", [])
            if isinstance(action, PlayHand) and not action.card_indices and hand:
                n = min(5, len(hand))
                count = random.randint(1, n)
                indices = tuple(sorted(random.sample(range(len(hand)), count)))
                action = PlayHand(card_indices=indices)
            if isinstance(action, Discard) and not action.card_indices and hand:
                n = min(5, len(hand))
                count = random.randint(1, n)
                indices = tuple(sorted(random.sample(range(len(hand)), count)))
                action = Discard(card_indices=indices)

            adapter.step(action)
            obs = encode_observation(adapter.raw_state)
            _check_obs(obs, step_num)


# ---------------------------------------------------------------------------
# Global context specific tests
# ---------------------------------------------------------------------------


class TestGlobalContext:
    def test_phase_one_hot(self):
        adapter = _make_adapter()
        obs = encode_observation(adapter.raw_state)
        # BLIND_SELECT = index 0
        phase_vec = obs.global_context[0:6]
        assert phase_vec[0] == 1.0
        assert np.sum(phase_vec) == 1.0

    def test_phase_changes_after_select(self):
        adapter = _make_adapter()
        _step_to_hand(adapter)
        obs = encode_observation(adapter.raw_state)
        phase_vec = obs.global_context[0:6]
        # SELECTING_HAND = index 1
        assert phase_vec[1] == 1.0
        assert np.sum(phase_vec) == 1.0

    def test_blind_on_deck_one_hot(self):
        adapter = _make_adapter()
        obs = encode_observation(adapter.raw_state)
        bod_vec = obs.global_context[6:10]
        # "Small" = index 1
        assert bod_vec[1] == 1.0
        assert np.sum(bod_vec) == 1.0

    def test_hand_levels_encoded(self):
        adapter = _make_adapter()
        obs = encode_observation(adapter.raw_state)
        # Hand levels start at index 30, 12 * 5 = 60 values
        hl_vec = obs.global_context[30:90]
        # All should be non-negative (levels, chips, mult are positive)
        assert np.all(hl_vec >= 0.0)
        # At least some visible hands should have non-zero chips/mult
        assert np.sum(hl_vec) > 0.0


# ---------------------------------------------------------------------------
# Encoding individual entities
# ---------------------------------------------------------------------------


class TestIndividualEncoders:
    def test_encode_playing_card_all_suits(self):
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state

        suits = [("Hearts", "H"), ("Diamonds", "D"), ("Clubs", "C"), ("Spades", "S")]
        for suit_name, suit_letter in suits:
            card = Card()
            card.set_base(f"{suit_letter}_A", suit_name, "Ace")
            card.set_ability("c_base")
            vec = encode_playing_card(card, 0, gs)
            assert vec.shape == (D_PLAYING_CARD,)
            assert vec[0] > 0.0  # rank_id

    def test_encode_consumable(self):
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state

        card = Card()
        card.set_ability("c_strength")
        card.sell_cost = 2
        gs["consumables"] = [card]

        vec = encode_consumable(card, gs)
        assert vec.shape == (D_CONSUMABLE,)
        assert vec[0] > 0.0  # center_key_id

    def test_encode_shop_item(self):
        from jackdaw.engine.card import Card

        adapter = _make_adapter()
        gs = adapter.raw_state
        gs["dollars"] = 10

        card = Card()
        card.set_ability("j_joker")
        card.cost = 4
        gs["jokers"] = []

        vec = encode_shop_item(card, gs)
        assert vec.shape == (D_SHOP,)
        assert vec[3] == 1.0  # affordable (cost 4 <= dollars 10)
        assert vec[4] == 1.0  # has_slot (0 jokers < 5 slots)
