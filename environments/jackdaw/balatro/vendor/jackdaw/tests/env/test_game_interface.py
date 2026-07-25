"""Tests for the unified game interface layer.

Covers:
- DirectAdapter reset/step/legal_actions cycle
- DirectAdapter runs to completion with random_agent
- BridgeAdapter(SimBackend) produces identical behavior to DirectAdapter
- done/won flags set correctly on game over and win
- GameAdapter protocol compliance
"""

from __future__ import annotations

import random

import pytest

from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
)
from jackdaw.env.game_interface import (
    BridgeAdapter,
    DirectAdapter,
    GameAdapter,
    GameState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = "TEST_INTERFACE_42"
BACK = "b_red"
STAKE = 1


def _random_agent_step(adapter: GameAdapter) -> None:
    """Pick a random legal action, resolving marker actions."""
    legal = adapter.get_legal_actions()
    assert legal, "No legal actions available"

    hand = adapter.raw_state.get("hand", [])

    # Separate progress-making from utility
    progress = [
        a
        for a in legal
        if not isinstance(
            a, (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)
        )
    ]
    pool = progress if progress else legal

    action = random.choice(pool)

    # Resolve marker PlayHand/Discard
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


def _run_to_completion(adapter: GameAdapter, max_actions: int = 5000) -> int:
    """Run the adapter to completion using random actions. Returns action count."""
    actions = 0
    while actions < max_actions:
        if adapter.done:
            break
        phase = adapter.raw_state.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break
        legal = adapter.get_legal_actions()
        if not legal:
            break
        _random_agent_step(adapter)
        actions += 1
    return actions


# ---------------------------------------------------------------------------
# GameState dataclass tests
# ---------------------------------------------------------------------------


class TestGameState:
    def test_frozen(self):
        gs = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            round=0,
            dollars=4,
            hands_left=4,
            discards_left=3,
            hand_size=8,
            joker_slots=5,
            consumable_slots=2,
            blind_on_deck="Small",
            blind_chips=0,
            chips=0,
            won=False,
            done=False,
        )
        with pytest.raises(AttributeError):
            gs.dollars = 10  # type: ignore[misc]

    def test_fields(self):
        gs = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            round=3,
            dollars=15,
            hands_left=3,
            discards_left=2,
            hand_size=8,
            joker_slots=5,
            consumable_slots=2,
            blind_on_deck="Boss",
            blind_chips=450,
            chips=100,
            won=False,
            done=False,
        )
        assert gs.phase == GamePhase.SELECTING_HAND
        assert gs.ante == 2
        assert gs.dollars == 15
        assert gs.blind_chips == 450
        assert not gs.done


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_direct_adapter_satisfies_protocol(self):
        assert isinstance(DirectAdapter(), GameAdapter)

    def test_bridge_adapter_satisfies_protocol(self):
        from jackdaw.bridge.backend import SimBackend

        assert isinstance(BridgeAdapter(SimBackend()), GameAdapter)


# ---------------------------------------------------------------------------
# DirectAdapter tests
# ---------------------------------------------------------------------------


class TestDirectAdapter:
    def test_reset_returns_game_state(self):
        adapter = DirectAdapter()
        state = adapter.reset(BACK, STAKE, SEED)
        assert isinstance(state, GameState)
        assert state.phase == GamePhase.BLIND_SELECT
        assert state.blind_on_deck == "Small"
        assert state.dollars > 0 or state.dollars == 0  # valid int
        assert not state.done
        assert not state.won

    def test_reset_sets_raw_state(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        gs = adapter.raw_state
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["blind_on_deck"] == "Small"
        assert "hand_levels" in gs
        assert "rng" in gs

    def test_step_select_blind(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        state = adapter.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND
        assert state.hands_left > 0

    def test_legal_actions_blind_select(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, SelectBlind) for a in legal)

    def test_legal_actions_selecting_hand(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, PlayHand) for a in legal)

    def test_done_false_initially(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        assert not adapter.done
        assert not adapter.won

    def test_raw_state_is_zero_copy(self):
        """raw_state must return the same dict object (no copies)."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        gs1 = adapter.raw_state
        gs2 = adapter.raw_state
        assert gs1 is gs2

    def test_run_to_completion_random(self):
        """DirectAdapter completes a full run with random agent."""
        random.seed(12345)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(adapter, max_actions=5000)
        assert actions > 0
        # Game should have ended
        assert adapter.done or adapter.won

    def test_reset_clears_previous_run(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        # Reset should start fresh
        state = adapter.reset(BACK, STAKE, "DIFFERENT_SEED")
        assert state.phase == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# BridgeAdapter(SimBackend) tests
# ---------------------------------------------------------------------------


class TestBridgeAdapterSim:
    def _make_adapter(self) -> BridgeAdapter:
        from jackdaw.bridge.backend import SimBackend

        return BridgeAdapter(SimBackend())

    def test_reset_returns_game_state(self):
        adapter = self._make_adapter()
        state = adapter.reset(BACK, STAKE, SEED)
        assert isinstance(state, GameState)
        assert state.phase == GamePhase.BLIND_SELECT
        assert not state.done

    def test_step_select_blind(self):
        adapter = self._make_adapter()
        adapter.reset(BACK, STAKE, SEED)
        state = adapter.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND

    def test_legal_actions(self):
        adapter = self._make_adapter()
        adapter.reset(BACK, STAKE, SEED)
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, SelectBlind) for a in legal)

    def test_raw_state_is_sim_engine_dict(self):
        """For SimBackend, raw_state should be the engine's live dict."""
        from jackdaw.bridge.backend import SimBackend

        backend = SimBackend()
        adapter = BridgeAdapter(backend)
        adapter.reset(BACK, STAKE, SEED)
        assert adapter.raw_state is backend._gs

    def test_done_won_flags(self):
        adapter = self._make_adapter()
        adapter.reset(BACK, STAKE, SEED)
        assert not adapter.done
        assert not adapter.won

    def test_run_to_completion(self):
        """BridgeAdapter(SimBackend) completes a run with random agent."""
        random.seed(12345)
        adapter = self._make_adapter()
        adapter.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(adapter, max_actions=5000)
        assert actions > 0
        assert adapter.done or adapter.won


# ---------------------------------------------------------------------------
# DirectAdapter vs BridgeAdapter(SimBackend) equivalence
# ---------------------------------------------------------------------------


class TestEquivalence:
    """DirectAdapter and BridgeAdapter(SimBackend) must produce identical
    behavior for the same seed when driven by the same deterministic agent.
    """

    def _greedy_step(self, adapter: GameAdapter) -> str | None:
        """Deterministic greedy agent step. Returns action type name or None."""
        legal = adapter.get_legal_actions()
        if not legal:
            return None

        hand = adapter.raw_state.get("hand", [])

        for a in legal:
            if isinstance(a, SelectBlind):
                adapter.step(a)
                return "SelectBlind"
            if isinstance(a, CashOut):
                adapter.step(a)
                return "CashOut"
            if isinstance(a, NextRound):
                adapter.step(a)
                return "NextRound"
            if isinstance(a, SkipPack):
                adapter.step(a)
                return "SkipPack"

        # Play hand if possible
        for a in legal:
            if isinstance(a, PlayHand) and hand:
                n = min(5, len(hand))
                action = PlayHand(card_indices=tuple(range(n)))
                adapter.step(action)
                return "PlayHand"

        # Fallback
        adapter.step(legal[0])
        return type(legal[0]).__name__

    def test_identical_sequence(self):
        """Both adapters produce the same state sequence with greedy agent."""
        from jackdaw.bridge.backend import SimBackend

        direct = DirectAdapter()
        bridge = BridgeAdapter(SimBackend())

        ds = direct.reset(BACK, STAKE, SEED)
        bs = bridge.reset(BACK, STAKE, SEED)

        assert ds.phase == bs.phase
        assert ds.dollars == bs.dollars
        assert ds.ante == bs.ante

        max_steps = 200
        for i in range(max_steps):
            if direct.done or bridge.done:
                break

            d_phase = direct.raw_state.get("phase")
            b_phase = bridge.raw_state.get("phase")
            if direct.won and d_phase == GamePhase.SHOP:
                break
            if bridge.won and b_phase == GamePhase.SHOP:
                break

            # Take same greedy action on both
            d_legal = direct.get_legal_actions()
            b_legal = bridge.get_legal_actions()

            if not d_legal or not b_legal:
                break

            # Use direct's legal actions to pick action, apply to both
            hand = direct.raw_state.get("hand", [])

            action = None
            for a in d_legal:
                if isinstance(a, SelectBlind):
                    action = a
                    break
                if isinstance(a, CashOut):
                    action = a
                    break
                if isinstance(a, NextRound):
                    action = a
                    break
                if isinstance(a, SkipPack):
                    action = a
                    break

            if action is None:
                for a in d_legal:
                    if isinstance(a, PlayHand) and hand:
                        n = min(5, len(hand))
                        action = PlayHand(card_indices=tuple(range(n)))
                        break

            if action is None:
                action = d_legal[0]

            ds = direct.step(action)
            bs = bridge.step(action)

            # Core state fields must match
            assert ds.phase == bs.phase, f"Step {i}: phase mismatch {ds.phase} != {bs.phase}"
            assert ds.dollars == bs.dollars, (
                f"Step {i}: dollars mismatch {ds.dollars} != {bs.dollars}"
            )
            assert ds.ante == bs.ante, f"Step {i}: ante mismatch"
            assert ds.round == bs.round, f"Step {i}: round mismatch"
            assert ds.chips == bs.chips, f"Step {i}: chips mismatch"
            assert ds.hands_left == bs.hands_left, f"Step {i}: hands_left mismatch"
            assert ds.discards_left == bs.discards_left, f"Step {i}: discards_left mismatch"

        # Both must end in the same terminal state
        assert direct.done == bridge.done
        assert direct.won == bridge.won


# ---------------------------------------------------------------------------
# Game over / win flag tests
# ---------------------------------------------------------------------------


class TestTerminalFlags:
    def test_game_over_sets_done(self):
        """When the game reaches GAME_OVER, done must be True."""
        random.seed(99999)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        if adapter.raw_state.get("phase") == GamePhase.GAME_OVER:
            assert adapter.done is True

    def test_won_flag_on_win(self):
        """If the run is won, adapter.won must be True."""
        # Use a deterministic greedy agent that may or may not win —
        # we just verify the flag is consistent with raw state.
        random.seed(77777)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        assert adapter.won == adapter.raw_state.get("won", False)

    def test_done_consistent_with_phase(self):
        """adapter.done must match phase == GAME_OVER."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        # Not done at start
        assert not adapter.done
        assert adapter.raw_state["phase"] != GamePhase.GAME_OVER

    def test_bridge_done_flag(self):
        """BridgeAdapter done/won flags match underlying engine."""
        from jackdaw.bridge.backend import SimBackend

        random.seed(88888)
        adapter = BridgeAdapter(SimBackend())
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        # Flags must be consistent
        raw_won = adapter.raw_state.get("won", False)
        assert adapter.won == raw_won
