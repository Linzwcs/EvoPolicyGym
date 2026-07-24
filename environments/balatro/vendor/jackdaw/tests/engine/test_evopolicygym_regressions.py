"""Regressions discovered while exercising Jackdaw through EvoPolicyGym."""

from __future__ import annotations

from jackdaw.engine.actions import CashOut, GamePhase, SelectBlind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def _playing_cards(game_state):
    return [
        *game_state.get("deck", []),
        *game_state.get("hand", []),
        *game_state.get("discard_pile", []),
    ]


class TestMarbleJokerRegression:
    def test_setting_blind_creates_a_stone_card_with_a_front(self):
        game_state = initialize_run("b_red", 1, "EPG-MARBLE")
        game_state["jokers"].append(create_joker("j_marble"))
        before = {id(card) for card in _playing_cards(game_state)}

        step(game_state, SelectBlind())

        created = [
            card for card in _playing_cards(game_state) if id(card) not in before
        ]
        assert len(created) == 1
        assert created[0].base is not None
        assert created[0].ability.get("effect") == "Stone Card"

    def test_cash_out_target_roll_ignores_a_frontless_legacy_card(self):
        game_state = initialize_run("b_red", 1, "EPG-MARBLE-LEGACY")
        for card in _playing_cards(game_state):
            card.ability["effect"] = "Stone Card"

        malformed = Card()
        malformed.ability = {"effect": "m_stone", "set": "Enhanced"}
        game_state["deck"].append(malformed)
        game_state["phase"] = GamePhase.ROUND_EVAL

        step(game_state, CashOut())

        assert game_state["phase"] == GamePhase.SHOP
        assert game_state["current_round"]["mail_card"] == {
            "rank": "Ace",
            "id": 14,
        }
