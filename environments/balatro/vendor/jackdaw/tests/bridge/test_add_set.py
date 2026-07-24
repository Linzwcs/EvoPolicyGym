"""Tests for SimBackend add/set debug methods."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import RPCError, SimBackend


@pytest.fixture
def backend() -> SimBackend:
    """A SimBackend with an active game in SELECTING_HAND."""
    b = SimBackend()
    b.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "ADD_SET_TEST"})
    b.handle("select", {})  # advance to SELECTING_HAND
    return b


class TestSet:
    def test_set_money(self, backend: SimBackend) -> None:
        backend.handle("set", {"money": 100})
        gs = backend._gs
        assert gs is not None
        assert gs["dollars"] == 100

    def test_set_hands(self, backend: SimBackend) -> None:
        backend.handle("set", {"hands": 10})
        gs = backend._gs
        assert gs is not None
        assert gs["current_round"]["hands_left"] == 10

    def test_set_discards(self, backend: SimBackend) -> None:
        backend.handle("set", {"discards": 7})
        gs = backend._gs
        assert gs is not None
        assert gs["current_round"]["discards_left"] == 7

    def test_set_ante(self, backend: SimBackend) -> None:
        backend.handle("set", {"ante": 5})
        gs = backend._gs
        assert gs is not None
        assert gs["round_resets"]["ante"] == 5

    def test_set_round(self, backend: SimBackend) -> None:
        backend.handle("set", {"round": 3})
        gs = backend._gs
        assert gs is not None
        assert gs["round"] == 3

    def test_set_chips(self, backend: SimBackend) -> None:
        backend.handle("set", {"chips": 500})
        gs = backend._gs
        assert gs is not None
        assert gs["chips"] == 500

    def test_set_multiple(self, backend: SimBackend) -> None:
        backend.handle("set", {"money": 50, "ante": 3, "round": 2})
        gs = backend._gs
        assert gs is not None
        assert gs["dollars"] == 50
        assert gs["round_resets"]["ante"] == 3
        assert gs["round"] == 2

    def test_set_no_active_run(self) -> None:
        b = SimBackend()
        with pytest.raises(RPCError, match="No active run"):
            b.handle("set", {"money": 100})


class TestAdd:
    def test_add_joker(self, backend: SimBackend) -> None:
        before = len(backend._gs["jokers"])
        backend.handle("add", {"key": "j_joker"})
        assert len(backend._gs["jokers"]) == before + 1
        added = backend._gs["jokers"][-1]
        assert added.center_key == "j_joker"

    def test_add_joker_with_edition(self, backend: SimBackend) -> None:
        backend.handle("add", {"key": "j_joker", "edition": "foil"})
        added = backend._gs["jokers"][-1]
        assert added.center_key == "j_joker"
        edition = added.get_edition()
        # Foil adds 50 chips — verify the edition effect is present
        assert edition.get("chip_mod") == 50

    def test_add_joker_eternal(self, backend: SimBackend) -> None:
        backend.handle("add", {"key": "j_joker", "eternal": True})
        added = backend._gs["jokers"][-1]
        assert added.eternal is True

    def test_add_consumable(self, backend: SimBackend) -> None:
        before = len(backend._gs["consumables"])
        backend.handle("add", {"key": "c_magician"})
        assert len(backend._gs["consumables"]) == before + 1
        added = backend._gs["consumables"][-1]
        assert added.center_key == "c_magician"

    def test_add_playing_card_to_hand(self, backend: SimBackend) -> None:
        # In SELECTING_HAND, cards go to hand
        before = len(backend._gs["hand"])
        backend.handle("add", {"key": "H_A"})
        assert len(backend._gs["hand"]) == before + 1

    def test_add_playing_card_with_enhancement(self, backend: SimBackend) -> None:
        backend.handle("add", {"key": "H_A", "enhancement": "m_glass"})
        added = backend._gs["hand"][-1]
        assert added.ability.get("effect") == "Glass Card"

    def test_add_playing_card_with_seal(self, backend: SimBackend) -> None:
        backend.handle("add", {"key": "H_A", "seal": "Gold"})
        added = backend._gs["hand"][-1]
        assert added.seal == "Gold"

    def test_add_playing_card_with_edition(self, backend: SimBackend) -> None:
        backend.handle("add", {"key": "H_A", "edition": "polychrome"})
        added = backend._gs["hand"][-1]
        edition = added.get_edition()
        # Polychrome gives x1.5 mult
        assert edition.get("x_mult_mod") == 1.5

    def test_add_invalid_key(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Unrecognised"):
            backend.handle("add", {"key": "z_invalid"})

    def test_add_no_key(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="requires"):
            backend.handle("add", {})

    def test_add_no_active_run(self) -> None:
        b = SimBackend()
        with pytest.raises(RPCError, match="No active run"):
            b.handle("add", {"key": "j_joker"})

    def test_add_invalid_playing_card(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Invalid playing card"):
            backend.handle("add", {"key": "X_Z"})
