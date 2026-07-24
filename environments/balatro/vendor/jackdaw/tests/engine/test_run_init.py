"""Tests for jackdaw.engine.run_init — merged unit + integration (trimmed).

Coverage: get_starting_params defaults, init_game_object keys,
Red/Abandoned/Zodiac deck integration tests.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.challenges import CHALLENGES
from jackdaw.engine.data.prototypes import BLINDS, TAGS
from jackdaw.engine.pools import UNAVAILABLE, get_current_pool
from jackdaw.engine.profile import (
    fresh_profile,
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import get_starting_params, init_game_object, initialize_run
from jackdaw.engine.stakes import DEFAULT_STARTING_PARAMS, apply_stake_modifiers

# ---------------------------------------------------------------------------
# get_starting_params
# ---------------------------------------------------------------------------


class TestGetStartingParams:
    def test_defaults(self):
        sp = get_starting_params()
        assert sp["dollars"] == 4
        assert sp["hand_size"] == 8
        assert sp["discards"] == 3
        assert sp["hands"] == 4
        assert sp["reroll_cost"] == 5
        assert sp["joker_slots"] == 5
        assert sp["ante_scaling"] == 1
        assert sp["consumable_slots"] == 2
        assert sp["no_faces"] is False
        assert sp["erratic_suits_and_ranks"] is False


# ---------------------------------------------------------------------------
# init_game_object
# ---------------------------------------------------------------------------


class TestInitGameObject:
    def test_top_level_keys(self):
        gs = init_game_object()
        expected = {
            "won",
            "round_scores",
            "joker_usage",
            "consumeable_usage",
            "hand_usage",
            "last_tarot_planet",
            "win_ante",
            "stake",
            "modifiers",
            "starting_params",
            "banned_keys",
            "round",
            "probabilities",
            "bosses_used",
            "pseudorandom",
            "starting_deck_size",
            "ecto_minus",
            "pack_size",
            "skips",
            "STOP_USE",
            "edition_rate",
            "joker_rate",
            "tarot_rate",
            "planet_rate",
            "spectral_rate",
            "playing_card_rate",
            "consumeable_buffer",
            "joker_buffer",
            "discount_percent",
            "interest_cap",
            "interest_amount",
            "inflation",
            "hands_played",
            "unused_discards",
            "perishable_rounds",
            "rental_rate",
            "blind",
            "chips",
            "dollars",
            "max_jokers",
            "bankrupt_at",
            "current_boss_streak",
            "base_reroll_cost",
            "blind_on_deck",
            "sort",
            "previous_round",
            "tags",
            "tag_tally",
            "pool_flags",
            "used_jokers",
            "used_vouchers",
            "current_round",
            "round_resets",
            "round_bonus",
            "shop",
        }
        assert expected.issubset(gs.keys()), f"Missing keys: {expected - gs.keys()}"


# ---------------------------------------------------------------------------
# Integration: Red Deck, stake 1
# ---------------------------------------------------------------------------


class TestRedDeckStake1:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_red", 1, self.SEED)

    def test_deck_size_52(self):
        assert len(self._gs()["deck"]) == 52

    def test_discards_4(self):
        assert self._gs()["starting_params"]["discards"] == 4  # 3 + 1

    def test_boss_blind_selected(self):
        gs = self._gs()
        boss = gs["round_resets"]["blind_choices"]["Boss"]
        assert boss == "bl_head"
        assert boss in BLINDS

    def test_tags_generated(self):
        gs = self._gs()
        tags = gs["round_resets"]["blind_tags"]
        assert tags == {"Small": "tag_economy", "Big": "tag_investment"}
        assert tags["Small"] in TAGS
        assert tags["Big"] in TAGS


# ---------------------------------------------------------------------------
# Integration: Abandoned Deck, stake 5
# ---------------------------------------------------------------------------


class TestAbandonedDeckStake5:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_abandoned", 5, self.SEED)

    def test_deck_size_40(self):
        """Abandoned Deck removes J/Q/K -> 52 - 12 = 40 cards."""
        assert len(self._gs()["deck"]) == 40

    def test_no_face_cards_in_deck(self):
        gs = self._gs()
        face_ranks = {"Jack", "Queen", "King"}
        for card in gs["deck"]:
            assert card.base.rank.value not in face_ranks


# ---------------------------------------------------------------------------
# Integration: Zodiac Deck
# ---------------------------------------------------------------------------


class TestZodiacDeck:
    SEED = "TESTSEED"

    def _gs(self):
        return initialize_run("b_zodiac", 1, self.SEED)

    def test_three_vouchers_applied(self):
        uv = self._gs()["used_vouchers"]
        assert uv.get("v_tarot_merchant") is True
        assert uv.get("v_planet_merchant") is True
        assert uv.get("v_overstock_norm") is True

    def test_tarot_rate(self):
        assert self._gs()["tarot_rate"] == 9.6


# ============================================================================
# Profile (merged from test_profile.py)
# ============================================================================


class TestFreshProfile:
    def test_105_jokers_unlocked(self):
        p = fresh_profile()
        from jackdaw.engine.data.prototypes import JOKERS

        unlocked_jokers = {k for k in JOKERS if k in p.unlocked}
        assert len(unlocked_jokers) == 105


class TestPoolFilteringWithProfile:
    def test_locked_joker_available_all_unlocked(self):
        """Sim assumes fully-unlocked profile: locked jokers are available."""
        # j_hanging_chad is a locked Common joker (unlocked=false in centers.json)
        # with no enhancement_gate or pool_flag restrictions.
        # The engine skips unlock checks, so it should appear in the pool.
        locked_joker = "j_hanging_chad"

        rng = PseudoRandom("PROFILE_TEST")
        pool, _ = get_current_pool("Joker", rng, 1, rarity=1)
        assert locked_joker in pool and pool[pool.index(locked_joker)] != UNAVAILABLE

    def test_discovery_gated_tags_available_by_default(self):
        """Tags with 'requires' fields are available when discovered is not set."""
        # tag_foil requires e_foil to be discovered.  The engine defaults to
        # everything discovered, so it should appear in the Tag pool at ante 1.
        rng = PseudoRandom("TAG_DISC_TEST")
        pool, _ = get_current_pool("Tag", rng, 1)
        assert "tag_foil" in pool and pool[pool.index("tag_foil")] != UNAVAILABLE


# ============================================================================
# Stakes and Challenges (merged from test_run_config.py)
# ============================================================================


def _stake_gs(extra_params: dict | None = None) -> dict[str, Any]:
    """Return a fresh game_state with default starting_params."""
    params = dict(DEFAULT_STARTING_PARAMS)
    if extra_params:
        params.update(extra_params)
    return {"starting_params": params}


def _apply(stake: int, extra_params: dict | None = None) -> dict[str, Any]:
    """Apply stake modifiers and return the game_state for inspection."""
    gs = _stake_gs(extra_params)
    apply_stake_modifiers(stake, gs)
    return gs


class TestStake5:
    def test_discards_decremented(self):
        gs = _apply(5)
        assert gs["starting_params"]["discards"] == 2  # 3 - 1

    def test_cumulative_modifiers(self):
        gs = _apply(5)
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True  # from stake 2
        assert gs["modifiers"]["scaling"] == 2  # from stake 3
        assert gs["modifiers"]["enable_eternals_in_shop"] is True  # from stake 4


class TestStake8:
    def test_all_modifiers(self):
        gs = _apply(8)
        mods = gs["modifiers"]
        assert mods["no_blind_reward"]["Small"] is True
        assert mods["scaling"] == 3
        assert mods["enable_eternals_in_shop"] is True
        assert mods["enable_perishables_in_shop"] is True
        assert mods["enable_rentals_in_shop"] is True
        assert gs["starting_params"]["discards"] == 2  # 3 - 1


class TestChallengesData:
    def test_exactly_20_challenges(self):
        assert len(CHALLENGES) == 20


class TestIntegrationWithInitializeRun:
    def test_omelette_through_initialize_run(self):
        gs = initialize_run("b_red", 1, "OMELETTE_TEST", challenge=CHALLENGES["c_omelette_1"])
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True
        assert gs["banned_keys"].get("j_golden") is True
        assert len(gs["challenge_jokers"]) == 5
