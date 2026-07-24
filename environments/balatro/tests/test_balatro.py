from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue
from jackdaw.engine import initialize_run
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.data.prototypes import (
    BLINDS,
    BOOSTERS,
    CENTER_POOLS,
    ENHANCEMENTS,
    JOKERS,
    PLANETS,
    SPECTRALS,
    TAGS,
    TAROTS,
    VOUCHERS,
)
from jackdaw.engine.pools import UNAVAILABLE, get_current_pool
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import assign_ante_blinds

from balatro import BalatroBenchmark, baseline_program
from balatro.environment import (
    EXCLUDED_TAG_KEYS,
    EXCLUDED_VOUCHER_KEYS,
    BalatroEnvironment,
)
from balatro.observation import encode_observation
from balatro.rules import blind_rule, visible_card_rule, visible_tag_rule

_SCENARIO: dict[str, PolicyValue] = {"back": "b_red", "stake": 1}


class BalatroBenchmarkTests(unittest.TestCase):
    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = BalatroBenchmark()

        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(benchmark.episodes("validation", seed=7, count=10))

        self.assertEqual(train, repeated)
        self.assertEqual(len({item.environment_seed for item in train}), 10)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in validation
            )
        )
        self.assertTrue(all(item.scenario == _SCENARIO for item in train))

    def test_semantic_observation_is_deterministic(self) -> None:
        benchmark = BalatroBenchmark()
        episode = EpisodeSpec(environment_seed=123, scenario=_SCENARIO)
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=episode,
                    actions=(
                        {"kind": "select_blind"},
                        {
                            "kind": "play_hand",
                            "card_indices": [0],
                        },
                    ),
                ),
            ),
        )

        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(episode)
        try:
            observation = environment.reset()
        finally:
            environment.close()
        self.assertIsInstance(observation, dict)
        assert isinstance(observation, dict)
        self.assertEqual(
            observation["schema"],
            "evopolicygym-balatro/observation-v1",
        )
        self.assertEqual(observation["phase"], "blind_select")
        self.assertNotIn("seed", observation)
        self.assertIn("legal_actions", observation)
        blind = observation["blind"]
        assert isinstance(blind, dict)
        self.assertEqual(blind["rule"], "No special rule.")
        self.assertEqual(blind["dollar_reward"], 3)
        self.assertNotIn("reward", blind)
        skip_tag = blind["skip_tag"]
        assert isinstance(skip_tag, dict)
        self.assertIn("rule", skip_tag)

        guide = benchmark.spec.metadata["policy_guide"]
        assert isinstance(guide, dict)
        scoring = guide["scoring"]
        assert isinstance(scoring, dict)
        selected_vs_scoring = scoring["selected_vs_scoring"]
        assert isinstance(selected_vs_scoring, str)
        self.assertIn("kickers do not contribute", selected_vs_scoring)
        objective = guide["objective"]
        phases = guide["phases"]
        actions = guide["actions"]
        assert isinstance(objective, dict)
        assert isinstance(phases, dict)
        assert isinstance(actions, dict)
        game_money = objective["game_money"]
        round_eval = phases["round_eval"]
        strictness = actions["strictness"]
        assert isinstance(game_money, str)
        assert isinstance(round_eval, str)
        assert isinstance(strictness, str)
        self.assertIn("blind.dollar_reward", game_money)
        self.assertIn("cash_out", round_eval)
        self.assertIn("exactly the required fields", strictness)
        self.assertEqual(
            benchmark.spec.id,
            "jackdaw/Balatro/red-deck-white-stake/run-score-v2",
        )
        excluded = benchmark.spec.metadata["excluded_content"]
        assert isinstance(excluded, dict)
        self.assertEqual(excluded["tags"], list(EXCLUDED_TAG_KEYS))
        self.assertEqual(excluded["vouchers"], list(EXCLUDED_VOUCHER_KEYS))
        self.assertNotIn(skip_tag["key"], EXCLUDED_TAG_KEYS)

    def test_every_joker_has_a_public_rule_and_parameters(self) -> None:
        for key in BLINDS:
            self.assertTrue(blind_rule(key), key)
        for key in JOKERS:
            rule = visible_card_rule(
                key=key,
                card_set="Joker",
                parameters={},
            )
            assert isinstance(rule, dict)
            self.assertTrue(rule["summary"], key)

        game_state = initialize_run("b_red", 1, "EPG_PUBLIC_JOKER_RULE")
        game_state["jokers"].append(create_joker("j_jolly"))
        game_state["jokers"].append(create_joker("j_ancient"))
        game_state["used_vouchers"]["v_grabber"] = True
        observation = encode_observation(game_state, step_count=0)
        jokers = observation["jokers"]
        assert isinstance(jokers, list)
        self.assertEqual(len(jokers), 2)
        joker = jokers[0]
        assert isinstance(joker, dict)
        rule = joker["rule"]
        assert isinstance(rule, dict)
        parameters = rule["parameters"]
        rarity = rule["rarity"]
        summary = rule["summary"]
        assert isinstance(parameters, dict)
        assert isinstance(rarity, dict)
        assert isinstance(summary, str)
        self.assertIn("Pair", summary)
        self.assertEqual(parameters["t_mult"], 8)
        self.assertEqual(parameters["type"], "Pair")
        self.assertEqual(rarity["name"], "Common")

        ancient = jokers[1]
        assert isinstance(ancient, dict)
        ancient_rule = ancient["rule"]
        assert isinstance(ancient_rule, dict)
        visible_state = ancient_rule["visible_state"]
        assert isinstance(visible_state, dict)
        self.assertEqual(
            visible_state["target_suit"],
            game_state["current_round"]["ancient_card"]["suit"],
        )
        vouchers = observation["vouchers"]
        assert isinstance(vouchers, list)
        voucher = vouchers[0]
        assert isinstance(voucher, dict)
        self.assertEqual(voucher["name"], "Grabber")
        voucher_rule = voucher["rule"]
        assert isinstance(voucher_rule, dict)
        voucher_summary = voucher_rule["summary"]
        assert isinstance(voucher_summary, str)
        self.assertIn("1 hand per round", voucher_summary)

    def test_every_visible_non_joker_card_has_an_implemented_rule(self) -> None:
        pools = {
            "Enhanced": ENHANCEMENTS,
            "Tarot": TAROTS,
            "Planet": PLANETS,
            "Spectral": SPECTRALS,
            "Voucher": VOUCHERS,
            "Booster": BOOSTERS,
        }
        for card_set, prototypes in pools.items():
            for key in prototypes:
                rule = visible_card_rule(
                    key=key,
                    card_set=card_set,
                    parameters={},
                )
                assert isinstance(rule, dict)
                self.assertTrue(rule["summary"], key)
                self.assertIsInstance(rule["parameters"], dict)

        empress = visible_card_rule(
            key="c_empress",
            card_set="Tarot",
            parameters={"effect": "Enhance"},
        )
        assert isinstance(empress, dict)
        empress_summary = empress["summary"]
        assert isinstance(empress_summary, str)
        self.assertIn("up to 2 selected cards", empress_summary)
        parameters = empress["parameters"]
        assert isinstance(parameters, dict)
        self.assertEqual(parameters["max_highlighted"], 2)

        mercury = visible_card_rule(
            key="c_mercury",
            card_set="Planet",
            parameters={},
        )
        assert isinstance(mercury, dict)
        mercury_summary = mercury["summary"]
        assert isinstance(mercury_summary, str)
        self.assertIn("Pair", mercury_summary)
        self.assertIn("15 base Chips", mercury_summary)

        for key in TAGS:
            tag = visible_tag_rule(key)
            self.assertEqual(tag["key"], key)
            rule = tag["rule"]
            assert isinstance(rule, dict)
            self.assertTrue(rule["summary"], key)

    def test_inactive_content_is_filtered_before_ante_pool_selection(self) -> None:
        tag_rng = PseudoRandom("EPG_TAG_EXCLUSIONS")
        for key in EXCLUDED_TAG_KEYS:
            available, _ = get_current_pool(
                "Tag",
                tag_rng,
                8,
                discovered={"all"},
            )
            excluded, _ = get_current_pool(
                "Tag",
                tag_rng,
                8,
                banned_keys={key},
                discovered={"all"},
            )
            index = sorted(TAGS, key=lambda item: TAGS[item].order).index(key)
            self.assertEqual(available[index], key)
            self.assertEqual(excluded[index], UNAVAILABLE)

        voucher_rng = PseudoRandom("EPG_VOUCHER_EXCLUSIONS")
        voucher_order = list(CENTER_POOLS["Voucher"])
        for key in EXCLUDED_VOUCHER_KEYS:
            prerequisites = set(VOUCHERS[key].requires)
            available, _ = get_current_pool(
                "Voucher",
                voucher_rng,
                8,
                used_vouchers=prerequisites,
            )
            excluded, _ = get_current_pool(
                "Voucher",
                voucher_rng,
                8,
                used_vouchers=prerequisites,
                banned_keys={key},
            )
            index = voucher_order.index(key)
            self.assertEqual(available[index], key)
            self.assertEqual(excluded[index], UNAVAILABLE)

        benchmark = BalatroBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        environment = BalatroEnvironment(episode)
        try:
            environment.reset()
            state = environment._state
            assert state is not None
            self.assertEqual(
                {key for key, value in state["banned_keys"].items() if value},
                set(EXCLUDED_TAG_KEYS) | set(EXCLUDED_VOUCHER_KEYS),
            )
            for ante in range(1, 17):
                generated = assign_ante_blinds(ante, state["rng"], state)
                self.assertNotIn(generated["voucher"], EXCLUDED_VOUCHER_KEYS)
                self.assertNotIn(
                    generated["blind_tags"]["Small"],
                    EXCLUDED_TAG_KEYS,
                )
                self.assertNotIn(
                    generated["blind_tags"]["Big"],
                    EXCLUDED_TAG_KEYS,
                )
        finally:
            environment.close()

    def test_invalid_action_does_not_advance_or_repair_state(self) -> None:
        benchmark = BalatroBenchmark()
        episode = EpisodeSpec(environment_seed=456, scenario=_SCENARIO)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step({"kind": "next_round"})
            actual = environment.step({"kind": "select_blind"})
        finally:
            environment.close()

        repeated = benchmark.make_environment(episode)
        try:
            self.assertEqual(initial, repeated.reset())
            expected = repeated.step({"kind": "select_blind"})
        finally:
            repeated.close()

        self.assertEqual(actual, expected)

    def test_last_hand_projects_the_actual_score_result(self) -> None:
        benchmark = BalatroBenchmark()
        episode = EpisodeSpec(environment_seed=123, scenario=_SCENARIO)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            selected = environment.step({"kind": "select_blind"})
            scored = environment.step(
                {
                    "kind": "play_hand",
                    "card_indices": [0],
                }
            )
        finally:
            environment.close()

        assert isinstance(initial, dict)
        initial_last_hand = initial["last_hand"]
        assert isinstance(initial_last_hand, dict)
        self.assertEqual(initial_last_hand["total"], 0)
        assert isinstance(selected.observation, dict)
        selected_resources = selected.observation["resources"]
        assert isinstance(selected_resources, dict)
        chips_before = selected_resources["chips"]
        assert isinstance(chips_before, int) and not isinstance(chips_before, bool)
        assert isinstance(scored.observation, dict)
        scored_resources = scored.observation["resources"]
        assert isinstance(scored_resources, dict)
        chips_after = scored_resources["chips"]
        last_hand = scored.observation["last_hand"]
        assert isinstance(chips_after, int) and not isinstance(chips_after, bool)
        assert isinstance(last_hand, dict)

        self.assertEqual(last_hand["handname"], "High Card")
        self.assertEqual(last_hand["hand_type"], "High Card")
        self.assertEqual(last_hand["hand_level"], "Level 1")
        self.assertEqual(last_hand["level"], 1)
        last_chips = last_hand["chips"]
        last_mult = last_hand["mult"]
        total = last_hand["total"]
        scoring_cards = last_hand["scoring_cards"]
        breakdown = last_hand["breakdown"]
        assert isinstance(last_chips, (int, float)) and not isinstance(last_chips, bool)
        assert isinstance(last_mult, (int, float)) and not isinstance(last_mult, bool)
        assert isinstance(total, int) and not isinstance(total, bool)
        assert isinstance(scoring_cards, list)
        assert isinstance(breakdown, list)
        self.assertGreater(last_chips, 0)
        self.assertGreater(last_mult, 0)
        self.assertGreater(total, 0)
        self.assertEqual(chips_after - chips_before, total)
        self.assertEqual(len(scoring_cards), 1)
        self.assertTrue(breakdown)

    def test_feedback_penalizes_failure_and_keeps_seeds_private(self) -> None:
        benchmark = BalatroBenchmark()
        episode = EpisodeSpec(environment_seed=789, scenario=_SCENARIO)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            step = environment.step({"kind": "select_blind"})
        finally:
            environment.close()
        failed = EpisodeRecord(
            episode=episode,
            policy_seed=321,
            initial_observation=initial,
            transitions=(Transition(action={"kind": "select_blind"}, step=step),),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(feedback.artifacts[0].name, "replay.jsonl")
        content = feedback.artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", content)
        self.assertNotIn(b"policy_seed", content)
        self.assertNotIn(b"EPG", content)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertNotIn("episode_diagnostics", feedback.content)
        self.assertNotIn("action_counts", feedback.content)

    def test_baseline_completes_and_publishes_replay(self) -> None:
        result = evaluate(
            baseline_program(),
            BalatroBenchmark(),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=4,
                seed=5,
                episode_timeout_seconds=30,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "jackdaw/Balatro/red-deck-white-stake/run-score-v2",
        )
        self.assertGreater(result.feedback.score, 0.0)
        self.assertEqual(result.episodes[0].status, "completed")
        replay = result.feedback.artifacts[0]
        documents = tuple(json.loads(line) for line in replay.read_bytes().splitlines())
        transitions = tuple(document for document in documents if document["type"] == "transition")
        episodes = tuple(document for document in documents if document["type"] == "episode")
        self.assertTrue(transitions)
        self.assertEqual(len(episodes), 4)
        self.assertEqual(
            [document["episode_index"] for document in episodes],
            [0, 1, 2, 3],
        )
        self.assertTrue(all("selection" not in document for document in episodes))
        self.assertEqual(documents[0]["type"], "episode")
        self.assertIn("hand", transitions[0]["state"])
        self.assertNotIn("legal_actions", transitions[0]["state"])
        played = next(
            document
            for document in transitions
            if document["action"]["kind"] == "play_hand"
        )
        self.assertGreater(played["state"]["last_hand"]["total"], 0)
        self.assertTrue(played["state"]["last_hand"]["scoring_cards"])
        cleared = next(
            document
            for document in transitions
            if document["reward"] == 1.0
        )
        self.assertEqual(cleared["state"]["phase"], "round_eval")
        earnings = cleared["state"]["round_earnings"]
        self.assertGreater(earnings["blind_dollar_reward"], 0)
        self.assertEqual(
            earnings["total_dollars"],
            earnings["blind_dollar_reward"]
            + earnings["unused_hands_bonus"]
            + earnings["unused_discards_bonus"]
            + earnings["joker_dollars"]
            + earnings["interest"]
            - earnings["rental_cost"],
        )


if __name__ == "__main__":
    unittest.main()
