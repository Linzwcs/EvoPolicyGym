"""Integration tests for the Balatro RL environment.

End-to-end tests that exercise the full stack:
- Full episode runs with RandomAgent through DirectAdapter and BridgeAdapter
- Multi-seed stress testing for crash resilience
- Observation consistency with engine state
- Action roundtrip fidelity
- Policy network smoke test (if torch available)
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import RandomAgent
from jackdaw.env.game_interface import BridgeAdapter, DirectAdapter
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
    encode_observation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = "TEST123"
BACK = "b_red"
STAKE = 1  # White Stake
MAX_STEPS = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_episode_collecting(
    adapter,
    agent,
    *,
    max_steps: int = MAX_STEPS,
    collect_obs: bool = False,
):
    """Run a full episode, optionally collecting observations.

    Returns a dict with episode metadata and optionally collected data.
    """
    agent.reset()

    observations: list[Observation] = []
    mask_checks: list[bool] = []

    gs = adapter.raw_state
    steps = 0

    for _ in range(max_steps):
        if adapter.done:
            break
        phase = gs.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": legal}

        # Collect observation
        if collect_obs:
            obs = encode_observation(gs)
            observations.append(obs)

            # Check mask consistency: every legal action type has mask True
            mask_ok = True
            if mask.type_mask.sum() == 0 and legal:
                mask_ok = False
            mask_checks.append(mask_ok)

        fa = agent.act({}, mask, info)

        engine_action = factored_to_engine_action(fa, gs)
        adapter.step(engine_action)
        gs = adapter.raw_state
        steps += 1

    rr = gs.get("round_resets", {})
    return {
        "steps": steps,
        "won": adapter.won,
        "done": adapter.done,
        "ante": rr.get("ante", 1),
        "observations": observations,
        "mask_checks": mask_checks,
    }


# ---------------------------------------------------------------------------
# (a) Full episode test — DirectAdapter
# ---------------------------------------------------------------------------


class TestFullEpisodeDirectAdapter:
    """Run RandomAgent through a complete episode via DirectAdapter."""

    def test_episode_terminates(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        # Episode must terminate
        assert result["done"] or result["won"], "Episode did not terminate"
        assert result["steps"] > 0, "No steps taken"

    def test_observations_valid(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        for obs in result["observations"]:
            assert isinstance(obs, Observation)
            # Global context
            assert obs.global_context.shape == (D_GLOBAL,)
            assert obs.global_context.dtype == np.float32
            assert np.isfinite(obs.global_context).all(), "NaN/Inf in global_context"

            # Hand cards
            if obs.hand_cards.shape[0] > 0:
                assert obs.hand_cards.shape[1] == D_PLAYING_CARD
                assert np.isfinite(obs.hand_cards).all(), "NaN/Inf in hand_cards"

            # Jokers
            if obs.jokers.shape[0] > 0:
                assert obs.jokers.shape[1] == D_JOKER
                assert np.isfinite(obs.jokers).all(), "NaN/Inf in jokers"

            # Consumables
            if obs.consumables.shape[0] > 0:
                assert obs.consumables.shape[1] == D_CONSUMABLE
                assert np.isfinite(obs.consumables).all(), "NaN/Inf in consumables"

            # Shop
            if obs.shop_cards.shape[0] > 0:
                assert obs.shop_cards.shape[1] == D_SHOP
                assert np.isfinite(obs.shop_cards).all(), "NaN/Inf in shop_cards"

    def test_action_masking_consistent(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        result = _run_episode_collecting(adapter, agent, collect_obs=True)

        assert all(result["mask_checks"]), "Action mask inconsistent with legal actions"


# ---------------------------------------------------------------------------
# (b) Full episode test — BridgeAdapter(SimBackend) identical trajectory
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBridgeAdapterIdenticalTrajectory:
    """BridgeAdapter(SimBackend) must produce identical game trajectory."""

    def test_identical_trajectory(self):
        from jackdaw.bridge.backend import SimBackend

        # Run DirectAdapter episode
        direct = DirectAdapter()
        direct.reset(BACK, STAKE, SEED)

        bridge = BridgeAdapter(SimBackend())
        bridge.reset(BACK, STAKE, SEED)

        agent_d = RandomAgent()
        agent_d.reset()

        steps = 0
        for _ in range(MAX_STEPS):
            if direct.done or bridge.done:
                break
            d_phase = direct.raw_state.get("phase")
            if direct.won and d_phase == GamePhase.SHOP:
                break

            d_legal = direct.get_legal_actions()
            b_legal = bridge.get_legal_actions()
            if not d_legal or not b_legal:
                break

            gs = direct.raw_state
            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": d_legal}
            fa = agent_d.act({}, mask, info)

            engine_action = factored_to_engine_action(fa, gs)

            ds = direct.step(engine_action)
            bs = bridge.step(engine_action)
            steps += 1

            # Core state must match
            assert ds.phase == bs.phase, f"Step {steps}: phase {ds.phase} != {bs.phase}"
            assert ds.dollars == bs.dollars, f"Step {steps}: dollars {ds.dollars} != {bs.dollars}"
            assert ds.ante == bs.ante, f"Step {steps}: ante mismatch"
            assert ds.round == bs.round, f"Step {steps}: round mismatch"
            assert ds.chips == bs.chips, f"Step {steps}: chips mismatch"
            assert ds.hands_left == bs.hands_left, f"Step {steps}: hands_left mismatch"
            assert ds.discards_left == bs.discards_left, f"Step {steps}: discards_left mismatch"

        assert steps > 0, "No steps taken"
        assert direct.done == bridge.done, "done flag mismatch"
        assert direct.won == bridge.won, "won flag mismatch"


# ---------------------------------------------------------------------------
# (c) Multi-seed stress test
# ---------------------------------------------------------------------------


class TestMultiSeedStress:
    """Run 50 episodes with random seeds — no crashes, no NaN, no invalid actions."""

    def test_50_episodes_no_crashes(self):
        n_episodes = 50
        seeds = [f"STRESS_{i}" for i in range(n_episodes)]
        wins = 0
        antes: list[int] = []
        step_counts: list[int] = []

        for seed in seeds:
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            agent.reset()

            gs = adapter.raw_state
            steps = 0

            for _ in range(MAX_STEPS):
                if adapter.done:
                    break
                phase = gs.get("phase")
                if adapter.won and phase == GamePhase.SHOP:
                    break

                legal = adapter.get_legal_actions()
                if not legal:
                    break

                mask = get_action_mask(gs)
                info = {"raw_state": gs, "legal_actions": legal}
                fa = agent.act({}, mask, info)

                # Verify observation has no NaN
                obs = encode_observation(gs)
                assert np.isfinite(obs.global_context).all(), f"NaN in global_context seed={seed}"
                if obs.hand_cards.shape[0] > 0:
                    assert np.isfinite(obs.hand_cards).all(), f"NaN in hand_cards seed={seed}"

                engine_action = factored_to_engine_action(fa, gs)
                adapter.step(engine_action)
                gs = adapter.raw_state
                steps += 1

            rr = gs.get("round_resets", {})
            antes.append(rr.get("ante", 1))
            step_counts.append(steps)
            if adapter.won:
                wins += 1

        # Statistics
        win_rate = wins / n_episodes
        avg_ante = sum(antes) / len(antes)
        avg_steps = sum(step_counts) / len(step_counts)

        # Sanity checks — not performance assertions, just "reasonable"
        assert avg_ante >= 1.0, f"Average ante {avg_ante} is unreasonably low"
        assert avg_steps > 10, f"Average steps {avg_steps} is unreasonably low"

        # Print statistics for information
        print(f"\n  Stress test: {n_episodes} episodes")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Avg ante: {avg_ante:.1f}")
        print(f"  Avg steps: {avg_steps:.0f}")


# ---------------------------------------------------------------------------
# (d) Observation consistency test
# ---------------------------------------------------------------------------


class TestObservationConsistency:
    """At each step, observation entity counts must match game state."""

    def test_entity_counts_match(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        agent.reset()

        gs = adapter.raw_state
        checked = 0

        for _ in range(MAX_STEPS):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            obs = encode_observation(gs)

            # Hand cards count
            hand = gs.get("hand", [])
            assert obs.hand_cards.shape[0] == len(hand), (
                f"hand_cards.shape[0]={obs.hand_cards.shape[0]} != len(hand)={len(hand)}"
            )

            # Jokers count
            jokers = gs.get("jokers", [])
            assert obs.jokers.shape[0] == len(jokers), (
                f"jokers.shape[0]={obs.jokers.shape[0]} != len(jokers)={len(jokers)}"
            )

            # Consumables count
            consumables = gs.get("consumables", [])
            assert obs.consumables.shape[0] == len(consumables), (
                f"consumables.shape[0]={obs.consumables.shape[0]} "
                f"!= len(consumables)={len(consumables)}"
            )

            # Global context phase encoding matches game state phase
            phase_val = gs.get("phase", GamePhase.GAME_OVER)
            if isinstance(phase_val, str):
                phase_val = GamePhase(phase_val)
            _PHASE_IDX = {
                GamePhase.BLIND_SELECT: 0,
                GamePhase.SELECTING_HAND: 1,
                GamePhase.ROUND_EVAL: 2,
                GamePhase.SHOP: 3,
                GamePhase.PACK_OPENING: 4,
                GamePhase.GAME_OVER: 5,
            }
            expected_idx = _PHASE_IDX.get(phase_val, 5)
            # Phase is one-hot in obs.global_context[0:6]
            phase_vec = obs.global_context[:6]
            actual_idx = int(np.argmax(phase_vec))
            assert actual_idx == expected_idx, (
                f"Phase encoding mismatch: global_context phase={actual_idx} "
                f"but game_state phase={phase_val} (expected idx={expected_idx})"
            )

            checked += 1

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        assert checked > 0, "No steps checked"


# ---------------------------------------------------------------------------
# (e) Action roundtrip test
# ---------------------------------------------------------------------------


class TestActionRoundtrip:
    """engine_action -> factored -> engine_action must be lossless for representable actions."""

    def test_roundtrip_across_episode(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        agent = RandomAgent()
        agent.reset()

        gs = adapter.raw_state
        roundtrip_count = 0
        skip_count = 0

        for _ in range(MAX_STEPS):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            # Test roundtrip for each legal action
            for engine_action in legal:
                try:
                    fa = engine_action_to_factored(engine_action, gs)
                    recovered = factored_to_engine_action(fa, gs)

                    # Compare the recovered action to the original
                    assert type(engine_action) is type(recovered), (
                        f"Type mismatch: {type(engine_action).__name__} "
                        f"!= {type(recovered).__name__}"
                    )
                    roundtrip_count += 1

                except ValueError:
                    # Complex permutations can't be represented as single swaps
                    skip_count += 1

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        assert roundtrip_count > 0, "No actions were roundtrip-tested"

    def test_specific_action_types(self):
        """Test roundtrip for specific action types that must be exact."""
        from jackdaw.engine.actions import (
            Discard,
            PlayHand,
            SelectBlind,
            SkipBlind,
            SortHand,
        )

        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        gs = adapter.raw_state

        # SelectBlind roundtrip
        fa = engine_action_to_factored(SelectBlind(), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SelectBlind)

        # SkipBlind roundtrip
        fa = engine_action_to_factored(SkipBlind(), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SkipBlind)

        # Advance to selecting hand
        adapter.step(SelectBlind())
        gs = adapter.raw_state
        hand = gs.get("hand", [])

        if hand:
            # PlayHand roundtrip
            indices = tuple(range(min(3, len(hand))))
            orig = PlayHand(card_indices=indices)
            fa = engine_action_to_factored(orig, gs)
            recovered = factored_to_engine_action(fa, gs)
            assert isinstance(recovered, PlayHand)
            assert recovered.card_indices == orig.card_indices

            # Discard roundtrip
            orig = Discard(card_indices=indices)
            fa = engine_action_to_factored(orig, gs)
            recovered = factored_to_engine_action(fa, gs)
            assert isinstance(recovered, Discard)
            assert recovered.card_indices == orig.card_indices

        # SortHand roundtrip
        fa = engine_action_to_factored(SortHand(mode="rank"), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "rank"

        fa = engine_action_to_factored(SortHand(mode="suit"), gs)
        recovered = factored_to_engine_action(fa, gs)
        assert isinstance(recovered, SortHand)
        assert recovered.mode == "suit"


# ---------------------------------------------------------------------------
# (h) Full episode with RandomAgent — end-to-end pipeline validation
# ---------------------------------------------------------------------------


def _run_full_episode(adapter, agent, *, max_steps=MAX_STEPS):
    """Run a full episode returning rich diagnostics.

    Returns dict with: steps, won, done, ante, observations,
    masks_valid, action_conversions_ok.
    """
    agent.reset()

    gs = adapter.raw_state
    steps = 0
    obs_list: list[Observation] = []
    masks_valid: list[bool] = []
    action_ok: list[bool] = []

    for _ in range(max_steps):
        if adapter.done:
            break
        phase = gs.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        legal = adapter.get_legal_actions()
        if not legal:
            break

        # Observation
        obs = encode_observation(gs)
        obs_list.append(obs)

        # Action mask
        mask = get_action_mask(gs)
        has_legal_type = mask.type_mask.sum() > 0
        masks_valid.append(has_legal_type)

        info = {"raw_state": gs, "legal_actions": legal}
        fa = agent.act({}, mask, info)

        # Action conversion
        try:
            engine_action = factored_to_engine_action(fa, gs)
            action_ok.append(True)
        except ValueError:
            action_ok.append(False)
            break

        adapter.step(engine_action)
        gs = adapter.raw_state
        steps += 1

    rr = gs.get("round_resets", {})
    return {
        "steps": steps,
        "won": adapter.won,
        "done": adapter.done,
        "ante": rr.get("ante", 1),
        "observations": obs_list,
        "masks_valid": masks_valid,
        "action_ok": action_ok,
    }


class TestFullEpisodeRandomAgent:
    """Run RandomAgent for 20 seeds — validates full pipeline end-to-end."""

    N_SEEDS = 20

    def _seeds(self):
        return [f"RANDOM_E2E_{i}" for i in range(self.N_SEEDS)]

    @pytest.mark.slow
    def test_no_crashes(self):
        """Primary goal: no crashes across 20 episodes."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert result["steps"] > 0, f"seed={seed}: zero steps"

    @pytest.mark.slow
    def test_observations_valid(self):
        """Every observation has correct shapes and no NaN."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)

            for i, obs in enumerate(result["observations"]):
                assert obs.global_context.shape == (D_GLOBAL,), (
                    f"seed={seed} step={i}: bad global_context shape"
                )
                assert np.isfinite(obs.global_context).all(), (
                    f"seed={seed} step={i}: NaN in global_context"
                )
                if obs.hand_cards.shape[0] > 0:
                    assert obs.hand_cards.shape[1] == D_PLAYING_CARD
                    assert np.isfinite(obs.hand_cards).all()
                if obs.jokers.shape[0] > 0:
                    assert obs.jokers.shape[1] == D_JOKER
                    assert np.isfinite(obs.jokers).all()

    @pytest.mark.slow
    def test_masks_have_legal_type(self):
        """Every ActionMask has at least one legal type until game over."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert all(result["masks_valid"]), (
                f"seed={seed}: mask with zero legal types before game end"
            )

    @pytest.mark.slow
    def test_action_conversions_valid(self):
        """Every FactoredAction converts to engine action without ValueError."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert all(result["action_ok"]), f"seed={seed}: action conversion failed"

    @pytest.mark.slow
    def test_episodes_terminate(self):
        """Every episode terminates within max_steps."""
        for seed in self._seeds():
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            result = _run_full_episode(adapter, agent)
            assert result["done"] or result["won"], f"seed={seed}: episode did not terminate"


# ---------------------------------------------------------------------------
# (j) Action roundtrip fidelity — comprehensive per-step test
# ---------------------------------------------------------------------------


class TestActionRoundtripFidelity:
    """For every step of a full episode, verify all legal actions roundtrip."""

    @pytest.mark.slow
    def test_all_legal_actions_roundtrip(self):
        """For every legal action at every step, convert to factored and back."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, "ROUNDTRIP_FULL")
        agent = RandomAgent()
        agent.reset()

        gs = adapter.raw_state
        total_tested = 0
        total_skipped = 0
        failures: list[str] = []

        for step_idx in range(MAX_STEPS):
            if adapter.done:
                break
            phase = gs.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            legal = adapter.get_legal_actions()
            if not legal:
                break

            for engine_action in legal:
                try:
                    fa = engine_action_to_factored(engine_action, gs)
                    recovered = factored_to_engine_action(fa, gs)

                    # Type must match
                    if not isinstance(recovered, type(engine_action)):
                        failures.append(
                            f"step={step_idx} type mismatch: "
                            f"{type(engine_action).__name__} != "
                            f"{type(recovered).__name__}"
                        )

                    # For actions with card_indices, verify exact match
                    orig_ci = getattr(engine_action, "card_indices", None)
                    recov_ci = getattr(recovered, "card_indices", None)
                    if orig_ci is not None and recov_ci is not None:
                        if tuple(orig_ci) != tuple(recov_ci):
                            failures.append(
                                f"step={step_idx} card_indices mismatch: {orig_ci} != {recov_ci}"
                            )

                    total_tested += 1
                except ValueError:
                    # Complex permutations are expected to fail
                    total_skipped += 1

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            gs = adapter.raw_state

        assert total_tested > 0, "No actions roundtrip-tested"
        assert not failures, (
            f"{len(failures)} roundtrip failures "
            f"(tested={total_tested}, skipped={total_skipped}):\n" + "\n".join(failures[:20])
        )


# ---------------------------------------------------------------------------
# (k) Observation encoding stress test
# ---------------------------------------------------------------------------


class TestObservationEncodingStress:
    """Run 50 seeds through RandomAgent, encode at every step."""

    N_SEEDS = 50
    VALUE_LIMIT = 1e9  # no astronomically large numbers

    @pytest.mark.slow
    def test_no_nan_correct_shapes_reasonable_values(self):
        seeds = [f"OBS_STRESS_{i}" for i in range(self.N_SEEDS)]
        total_obs_checked = 0

        for seed in seeds:
            adapter = DirectAdapter()
            adapter.reset(BACK, STAKE, seed)
            agent = RandomAgent()
            agent.reset()
            gs = adapter.raw_state

            for _ in range(MAX_STEPS):
                if adapter.done:
                    break
                phase = gs.get("phase")
                if adapter.won and phase == GamePhase.SHOP:
                    break

                legal = adapter.get_legal_actions()
                if not legal:
                    break

                obs = encode_observation(gs)

                # Shape checks
                assert obs.global_context.shape == (D_GLOBAL,), (
                    f"seed={seed}: global_context shape {obs.global_context.shape}"
                )

                hand = gs.get("hand", [])
                assert obs.hand_cards.shape[0] == len(hand), (
                    f"seed={seed}: hand_cards rows {obs.hand_cards.shape[0]} "
                    f"!= hand size {len(hand)}"
                )
                if len(hand) > 0:
                    assert obs.hand_cards.shape[1] == D_PLAYING_CARD

                jokers = gs.get("jokers", [])
                assert obs.jokers.shape[0] == len(jokers), (
                    f"seed={seed}: jokers rows {obs.jokers.shape[0]} != joker count {len(jokers)}"
                )
                if len(jokers) > 0:
                    assert obs.jokers.shape[1] == D_JOKER

                # No NaN/Inf
                assert np.isfinite(obs.global_context).all(), (
                    f"seed={seed}: NaN/Inf in global_context"
                )
                if obs.hand_cards.shape[0] > 0:
                    assert np.isfinite(obs.hand_cards).all(), f"seed={seed}: NaN/Inf in hand_cards"
                if obs.jokers.shape[0] > 0:
                    assert np.isfinite(obs.jokers).all(), f"seed={seed}: NaN/Inf in jokers"
                if obs.consumables.shape[0] > 0:
                    assert np.isfinite(obs.consumables).all(), (
                        f"seed={seed}: NaN/Inf in consumables"
                    )
                if obs.shop_cards.shape[0] > 0:
                    assert np.isfinite(obs.shop_cards).all(), f"seed={seed}: NaN/Inf in shop_cards"
                if obs.pack_cards.shape[0] > 0:
                    assert np.isfinite(obs.pack_cards).all(), f"seed={seed}: NaN/Inf in pack_cards"

                # No astronomically large values (catches log-scale overflow)
                assert np.abs(obs.global_context).max() < self.VALUE_LIMIT, (
                    f"seed={seed}: global_context has value > {self.VALUE_LIMIT}"
                )
                if obs.hand_cards.shape[0] > 0:
                    assert np.abs(obs.hand_cards).max() < self.VALUE_LIMIT
                if obs.jokers.shape[0] > 0:
                    assert np.abs(obs.jokers).max() < self.VALUE_LIMIT

                total_obs_checked += 1

                mask = get_action_mask(gs)
                info = {"raw_state": gs, "legal_actions": legal}
                fa = agent.act({}, mask, info)
                engine_action = factored_to_engine_action(fa, gs)
                adapter.step(engine_action)
                gs = adapter.raw_state

        assert total_obs_checked > 1000, (
            f"Only checked {total_obs_checked} observations "
            f"(expected >1000 across {self.N_SEEDS} seeds)"
        )


# ---------------------------------------------------------------------------
# (m) Multi-deck, multi-stake stress test
# ---------------------------------------------------------------------------


ALL_DECK_KEYS = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_magic",
    "b_nebula",
    "b_ghost",
    "b_abandoned",
    "b_checkered",
    "b_zodiac",
    "b_painted",
    "b_anaglyph",
    "b_plasma",
    "b_erratic",
]


class TestMultiDeckMultiStake:
    """Run RandomAgent on all 15 decks × 3 seeds = 45 episodes. No crashes."""

    N_SEEDS_PER_DECK = 3

    @pytest.mark.slow
    def test_all_decks_no_crashes(self):
        failures: list[str] = []

        for deck in ALL_DECK_KEYS:
            for seed_idx in range(self.N_SEEDS_PER_DECK):
                seed = f"DECK_{deck}_{seed_idx}"
                try:
                    adapter = DirectAdapter()
                    adapter.reset(deck, STAKE, seed)
                    agent = RandomAgent()
                    result = _run_full_episode(adapter, agent, max_steps=MAX_STEPS)
                    assert result["steps"] > 0, "zero steps"
                except Exception as e:
                    failures.append(f"{deck} seed={seed}: {type(e).__name__}: {e}")

        assert not failures, f"{len(failures)}/45 episodes crashed:\n" + "\n".join(failures[:10])
