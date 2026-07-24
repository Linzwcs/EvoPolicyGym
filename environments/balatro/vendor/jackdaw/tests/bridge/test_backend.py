"""Tests for jackdaw.bridge.backend — SimBackend and LiveBackend."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import (
    BAD_REQUEST,
    INVALID_STATE,
    NOT_ALLOWED,
    LiveBackend,
    RPCError,
    SimBackend,
)

# ============================================================================
# SimBackend
# ============================================================================


@pytest.fixture()
def sim() -> SimBackend:
    return SimBackend()


def _start_run(sim: SimBackend, seed: str = "TEST") -> dict:
    return sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})


class TestSimHealth:
    def test_health(self, sim):
        result = sim.handle("health", None)
        assert result == {"status": "ok"}


class TestSimStartAndGamestate:
    def test_start_returns_gamestate(self, sim):
        result = _start_run(sim)

        assert result["state"] == "BLIND_SELECT"
        assert result["deck"] == "RED"
        assert result["stake"] == "WHITE"
        assert result["seed"] == "TEST"
        assert result["ante_num"] == 1
        assert result["money"] == 4
        assert result["won"] is False
        assert isinstance(result["blinds"], dict)
        assert isinstance(result["hands"], dict)

    def test_gamestate_after_start(self, sim):
        _start_run(sim)
        result = sim.handle("gamestate", None)

        assert result["state"] == "BLIND_SELECT"
        assert result["ante_num"] == 1

    def test_start_different_deck_and_stake(self, sim):
        result = sim.handle("start", {"deck": "BLUE", "stake": "RED", "seed": "X"})

        assert result["deck"] == "BLUE"
        assert result["stake"] == "RED"


class TestSimFullRound:
    def test_select_play_cashout_nextround(self, sim):
        _start_run(sim, seed="ROUND_TEST")

        # Select blind
        result = sim.handle("select", {})
        assert result["state"] == "SELECTING_HAND"
        assert result["hand"]["count"] > 0
        assert result["round"]["hands_left"] > 0

        # Make the blind trivially beatable
        sim._gs["blind"].chips = 1

        # Play first 5 cards
        hand_count = result["hand"]["count"]
        n = min(5, hand_count)
        result = sim.handle("play", {"cards": list(range(n))})

        # Should have won the blind
        assert result["state"] == "ROUND_EVAL"

        # Cash out
        result = sim.handle("cash_out", {})
        assert result["state"] == "SHOP"

        # Next round
        result = sim.handle("next_round", {})
        assert result["state"] == "BLIND_SELECT"


class TestSimNoActiveRun:
    def test_gamestate_before_start(self, sim):
        with pytest.raises(RPCError) as exc_info:
            sim.handle("gamestate", None)
        assert exc_info.value.code == INVALID_STATE

    def test_action_before_start(self, sim):
        with pytest.raises(RPCError) as exc_info:
            sim.handle("select", {})
        assert exc_info.value.code == INVALID_STATE


class TestSimIllegalAction:
    def test_play_during_blind_select(self, sim):
        _start_run(sim)
        # In BLIND_SELECT, playing cards is illegal
        with pytest.raises(RPCError) as exc_info:
            sim.handle("play", {"cards": [0, 1, 2]})
        assert exc_info.value.code == NOT_ALLOWED

    def test_cash_out_during_blind_select(self, sim):
        _start_run(sim)
        with pytest.raises(RPCError) as exc_info:
            sim.handle("cash_out", {})
        assert exc_info.value.code == NOT_ALLOWED


class TestSimBadParams:
    def test_buy_empty_params(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("buy", {})
        assert exc_info.value.code == BAD_REQUEST

    def test_sell_empty_params(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("sell", {})
        assert exc_info.value.code == BAD_REQUEST

    def test_unknown_method(self, sim):
        with pytest.raises(RPCError) as exc_info:
            sim.handle("nonexistent", {})
        assert exc_info.value.code == BAD_REQUEST


class TestSimMenu:
    def test_menu_resets_run(self, sim):
        _start_run(sim)
        result = sim.handle("menu", {})
        assert result == {"state": "MENU"}

        # Gamestate should fail after menu
        with pytest.raises(RPCError) as exc_info:
            sim.handle("gamestate", None)
        assert exc_info.value.code == INVALID_STATE

    def test_menu_without_run(self, sim):
        result = sim.handle("menu", {})
        assert result == {"state": "MENU"}


# ============================================================================
# RPCError
# ============================================================================


class TestRPCError:
    def test_fields(self):
        err = RPCError(code=-32001, message="test error", data={"key": "val"})
        assert err.code == -32001
        assert err.message == "test error"
        assert err.data == {"key": "val"}
        assert str(err) == "test error"

    def test_default_data(self):
        err = RPCError(code=-32000, message="oops")
        assert err.data == {}


# ============================================================================
# LiveBackend — requires a running balatrobot instance
# ============================================================================


@pytest.mark.live
class TestLiveBackendHealth:
    def test_health(self):
        backend = LiveBackend()
        result = backend.handle("health", None)
        assert result.get("status") == "ok"


@pytest.mark.live
class TestLiveBackendGamestate:
    def test_start_and_gamestate(self):
        backend = LiveBackend()
        backend.handle("menu", None)
        backend.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "LIVE_TEST"})
        result = backend.handle("gamestate", None)
        assert "state" in result
