from __future__ import annotations

import io
import json
import unittest
from typing import Any

import jax
import jax.numpy as jnp
import numpy
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from jumanji_benchmarks import (
    JUMANJI_PROFILES,
    JumanjiBenchmark,
    JumanjiConfig,
    baseline_program,
)


class JumanjiBenchmarkTests(unittest.TestCase):
    def test_all_profiles_run_real_multistep_feedback(self) -> None:
        self.assertEqual(len(JUMANJI_PROFILES), 18)
        for profile in JUMANJI_PROFILES:
            with self.subTest(profile=profile):
                config = JumanjiConfig(profile=profile)
                environment = JumanjiBenchmark(config).make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    initial_observation = environment.reset()
                    self.assertIsInstance(initial_observation, dict)
                    schema = _dictionary(
                        JumanjiBenchmark(config).spec.observation_space
                    )
                    schema_fields = _dictionary(schema["fields"])
                    self.assertEqual(
                        _policy_carriers(initial_observation),
                        {
                            field: _dictionary(specification)["policy_carrier"]
                            for field, specification in schema_fields.items()
                        },
                    )
                    observation = initial_observation
                    transitions: list[Transition] = []
                    interaction_limit = (
                        config.max_episode_steps if profile == "job-shop" else 4
                    )
                    for _ in range(interaction_limit):
                        action = _first_valid_action(observation, config=config)
                        step = environment.step(action)
                        transitions.append(Transition(action=action, step=step))
                        if step.done:
                            break
                        observation = step.observation
                    self.assertIsInstance(step.reward, float)
                    feedback = JumanjiBenchmark(config).feedback(
                        (
                            EpisodeRecord(
                                episode=EpisodeSpec(environment_seed=123),
                                policy_seed=456,
                                initial_observation=initial_observation,
                                transitions=tuple(transitions),
                            ),
                        )
                    )
                    self.assertEqual(len(feedback.artifacts), 2)
                    feedback_content = _dictionary(feedback.content)
                    manifest = _dictionary(
                        _sequence(feedback_content["observation_artifacts"])[0]
                    )
                    self.assertTrue(
                        all(
                            "policy_carrier" in field and "policy_path" in field
                            for field in manifest["fields"]
                        )
                    )
                    if profile == "job-shop":
                        summary = _dictionary(
                            _sequence(feedback_content["episode_summaries"])[0]
                        )
                        self.assertLessEqual(len(summary["actions"]), 16)
                        self.assertGreater(summary["distinct_actions"], 16)
                        self.assertGreater(summary["action_summaries_omitted"], 0)
                    trace = [
                        json.loads(line)
                        for line in feedback.artifacts[0].content.splitlines()
                    ]
                    self.assertEqual(trace[0]["profile"], profile)
                    semantics = trace[1]["decision_observation"]["semantics"]
                    self.assertEqual(semantics["kind"], profile)
                    self.assertIn("progress", semantics)
                    self.assertTrue(
                        all(
                            "policy_path" in field
                            for field in semantics["field_summaries"]
                        )
                    )
                    self.assertTrue(
                        all(
                            "policy_path" in field
                            for field in trace[1]["decision_observation"]["fields"]
                        )
                    )
                    if profile == "sliding-tile-puzzle":
                        progress = semantics["progress"]
                        self.assertEqual(
                            progress["solved"],
                            progress["correctly_positioned_tiles"]
                            == progress["total_tiles"],
                        )
                    if profile == "flat-pack":
                        progress = semantics["progress"]
                        self.assertEqual(
                            progress["solved"],
                            progress["occupied_grid_cells"] == progress["grid_cells"],
                        )
                    if profile == "knapsack":
                        progress = semantics["progress"]
                        self.assertAlmostEqual(
                            progress["remaining_capacity"],
                            progress["total_capacity"] - progress["packed_weight"],
                        )
                    if profile == "tetris":
                        progress = trace[1]["result_observation"]["semantics"][
                            "progress"
                        ]
                        self.assertEqual(progress["step"], 1)
                        self.assertEqual(len(progress["column_heights"]), 10)
                        self.assertEqual(
                            progress["aggregate_height"],
                            sum(progress["column_heights"]),
                        )
                        self.assertIn(
                            "lines_cleared", _dictionary(transitions[0].step.metrics)
                        )
                    self.assertEqual(
                        semantics["action_mask"] is not None,
                        config.has_action_mask,
                    )
                    with numpy.load(
                        io.BytesIO(feedback.artifacts[1].content),
                        allow_pickle=False,
                    ) as archive:
                        step_indices = archive["step_indices"].tolist()
                        if len(transitions) <= 48:
                            self.assertEqual(step_indices, list(range(len(transitions))))
                        else:
                            self.assertEqual(len(step_indices), 48)
                            self.assertEqual(step_indices[0], 0)
                            self.assertEqual(step_indices[-1], len(transitions) - 1)
                        self.assertTrue(
                            any(name.startswith("initial__") for name in archive.files)
                        )
                        if profile.startswith("sudoku"):
                            progress = trace[0]["initial_observation"]["semantics"][
                                "progress"
                            ]
                            self.assertEqual(
                                progress["filled_cells"],
                                int(numpy.count_nonzero(archive["initial__board"] >= 0)),
                            )
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        maze = JumanjiBenchmark()
        tetris = JumanjiBenchmark(JumanjiConfig(profile="tetris"))
        self.assertNotEqual(maze.spec.environment_digest, tetris.spec.environment_digest)
        self.assertEqual(tetris.spec.environment_parameters["profile"], "tetris")
        self.assertEqual(tetris.spec.max_episode_steps, 400)
        maze_fields = _dictionary(_dictionary(maze.spec.observation_space)["fields"])
        self.assertEqual(
            _dictionary(maze_fields["agent_position.row"])["policy_carrier"], "int"
        )
        self.assertEqual(_dictionary(maze_fields["walls"])["policy_carrier"], "TensorValue")
        easy_rubik = JumanjiBenchmark(
            JumanjiConfig(profile="rubiks-cube-partly-scrambled")
        )
        self.assertEqual(easy_rubik.spec.environment_parameters["initial_scramble_moves"], 7)
        self.assertEqual(
            _dictionary(
                _dictionary(_dictionary(easy_rubik.spec.observation_space)["fields"])["cube"]
            )["face_order"],
            ["up", "front", "right", "back", "left", "down"],
        )

    def test_rubik_specs_and_real_scramble_solutions(self) -> None:
        profiles = (
            ("rubiks-cube", 100, 200),
            ("rubiks-cube-partly-scrambled", 7, 20),
        )
        for profile, scramble_moves, time_limit in profiles:
            with self.subTest(profile=profile):
                config = JumanjiConfig(profile=profile)
                benchmark = JumanjiBenchmark(config)
                spec = benchmark.spec
                self.assertEqual(
                    spec.environment_parameters["initial_scramble_moves"],
                    scramble_moves,
                )
                self.assertEqual(spec.max_episode_steps, time_limit)
                self.assertIn(
                    f"exactly {scramble_moves}",
                    spec.description,
                )
                self.assertIn("All 18 actions are always legal", spec.description)
                self.assertIn("produces six uniform faces rewards 1", spec.description)
                cube_spec = _dictionary(
                    _dictionary(_dictionary(spec.observation_space)["fields"])["cube"]
                )
                self.assertIn("sticker color id", cube_spec["meaning"])
                sticker_colors = _dictionary(cube_spec["sticker_value_colors"])
                orientations = _dictionary(cube_spec["face_view_orientations"])
                self.assertEqual(sticker_colors["0"], "white")
                self.assertEqual(sticker_colors["5"], "yellow")
                self.assertIn(
                    "back face points up",
                    orientations["up"],
                )
                self.assertFalse(
                    spec.environment_parameters[
                        "misplaced_stickers_is_solution_distance"
                    ]
                )

                episode = EpisodeSpec(environment_seed=123)
                environment = benchmark.make_environment(episode)
                try:
                    initial = environment.reset()
                    transitions: list[Transition] = []
                    for action in _inverse_rubik_scramble(123, scramble_moves):
                        step = environment.step(action)
                        transitions.append(Transition(action=action, step=step))
                        if step.done:
                            break
                finally:
                    environment.close()

                self.assertEqual(len(transitions), scramble_moves)
                self.assertTrue(transitions[-1].step.terminated)
                self.assertFalse(transitions[-1].step.truncated)
                self.assertEqual(transitions[-1].step.reward, 1.0)
                self.assertEqual(
                    _dictionary(transitions[-1].step.metrics)["terminal_reason"], "solved"
                )
                feedback = benchmark.feedback(
                    (
                        EpisodeRecord(
                            episode=episode,
                            policy_seed=456,
                            initial_observation=initial,
                            transitions=tuple(transitions),
                        ),
                    )
                )
                trace = [
                    json.loads(line)
                    for line in feedback.artifacts[0].content.splitlines()
                ]
                final_transition = [
                    line for line in trace if line["type"] == "transition"
                ][-1]
                progress = final_transition["result_observation"]["semantics"][
                    "progress"
                ]
                self.assertTrue(progress["solved"])
                self.assertEqual(progress["uniform_faces"], 6)
                self.assertEqual(progress["correct_stickers"], 54)
                self.assertEqual(progress["misplaced_stickers"], 0)
                self.assertEqual(progress["sticker_counts_by_color"], [9] * 6)
                self.assertTrue(progress["valid_sticker_inventory"])

    def test_standard_rubik_real_time_limit(self) -> None:
        config = JumanjiConfig(profile="rubiks-cube")
        environment = JumanjiBenchmark(config).make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            for index in range(config.max_episode_steps):
                step = environment.step([0, 0, 0])
                self.assertEqual(step.done, index == config.max_episode_steps - 1)
        finally:
            environment.close()
        self.assertEqual(step.reward, 0.0)
        self.assertTrue(step.terminated)
        self.assertFalse(step.truncated)
        self.assertEqual(_dictionary(step.metrics)["terminal_reason"], "time_limit")
        self.assertEqual(_dictionary(step.observation)["step_count"], 200)

    def test_graph_coloring_spec_explains_objective_and_fields(self) -> None:
        spec = JumanjiBenchmark(JumanjiConfig(profile="graph-coloring")).spec
        self.assertIn("minimizing the number of distinct colors", spec.description)
        self.assertIn("successful terminal reward", spec.description)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn(
            "nodes u and v are adjacent", _dictionary(fields["adj_matrix"])["meaning"]
        )
        self.assertIn(
            "-1 until node i is assigned", _dictionary(fields["colors"])["meaning"]
        )
        self.assertIn(
            "node colored by the next action",
            _dictionary(fields["current_node_index"])["meaning"],
        )
        self.assertIn(
            "assigning color c",
            _dictionary(fields["action_mask"])["meaning"],
        )

    def test_minesweeper_spec_explains_objective_and_fields(self) -> None:
        spec = JumanjiBenchmark(JumanjiConfig(profile="minesweeper")).spec
        self.assertIn("Reveal all 90 non-mine cells", spec.description)
        self.assertIn("Each safe reveal rewards 1", spec.description)
        self.assertIn("a mine is revealed", spec.description)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        board = _dictionary(fields["board"])
        action_mask = _dictionary(fields["action_mask"])
        self.assertIn("-1 for an unexplored cell", board["meaning"])
        self.assertIn("up to eight neighbors", board["meaning"])
        self.assertIn(
            "still unexplored and may be selected",
            action_mask["meaning"],
        )
        self.assertIn("Total number of mines", _dictionary(fields["num_mines"])["meaning"])
        self.assertIn(
            "Number of cells selected", _dictionary(fields["step_count"])["meaning"]
        )

    def test_sliding_tile_spec_explains_goal_reward_and_fields(self) -> None:
        spec = JumanjiBenchmark(JumanjiConfig(profile="sliding-tile-puzzle")).spec
        self.assertIn("[1, 2, ..., 24, 0]", spec.description)
        self.assertIn("applies 200 random legal moves", spec.description)
        self.assertIn("newly correct positions", spec.description)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn(
            "solved flattened ordering", _dictionary(fields["puzzle"])["meaning"]
        )
        self.assertIn(
            "[row, column] coordinates",
            _dictionary(fields["empty_tile_position"])["meaning"],
        )
        self.assertIn("empty tile can move", _dictionary(fields["action_mask"])["meaning"])
        self.assertIn(
            "empty-tile moves taken", _dictionary(fields["step_count"])["meaning"]
        )

    def test_bin_pack_spec_explains_objective_nested_paths_and_fields(self) -> None:
        spec = JumanjiBenchmark(JumanjiConfig(profile="bin-pack")).spec
        self.assertIn("Maximize volume utilization", spec.description)
        self.assertIn("fixed-orientation item", spec.description)
        self.assertIn("return equals final volume utilization", spec.description)
        observation_space = _dictionary(spec.observation_space)
        self.assertIn("never observation['ems.x1']", observation_space["policy_path_rule"])
        fields = _dictionary(observation_space["fields"])
        ems_x1 = _dictionary(fields["ems.x1"])
        action_mask = _dictionary(fields["action_mask"])
        self.assertEqual(ems_x1["policy_path"], ["ems", "x1"])
        self.assertEqual(action_mask["policy_path"], ["action_mask"])
        self.assertIn("Normalized lower x", ems_x1["meaning"])
        self.assertIn("unpacked item i fits", action_mask["meaning"])
        self.assertIn("already packed", _dictionary(fields["items_placed"])["meaning"])

    def test_flat_pack_spec_and_mask_dead_end_are_explicit(self) -> None:
        config = JumanjiConfig(profile="flat-pack")
        spec = JumanjiBenchmark(config).spec
        self.assertIn("generated instances admit a complete tiling", spec.description)
        self.assertIn("clockwise rotation", spec.description)
        self.assertIn("return equals final grid occupancy", spec.description)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("zero cells are transparent", _dictionary(fields["blocks"])["meaning"])
        self.assertIn("without overlap", _dictionary(fields["action_mask"])["meaning"])

        environment = JumanjiBenchmark(config).make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            for _ in range(config.max_episode_steps):
                action = _first_valid_action(observation, config=config)
                step = environment.step(action)
                if step.done:
                    break
                observation = step.observation
            self.assertTrue(step.terminated)
            self.assertFalse(step.truncated)
            self.assertIs(_dictionary(step.metrics).get("no_legal_actions"), True)
            final_observation = step.observation
            self.assertIsInstance(final_observation, dict)
            if type(final_observation) is not dict:
                self.fail("expected an object observation")
            final_mask = final_observation["action_mask"]
            self.assertIsInstance(final_mask, TensorValue)
            if type(final_mask) is not TensorValue:
                self.fail("expected a TensorValue action mask")
            self.assertFalse(any(final_mask.data))
        finally:
            environment.close()

    def test_knapsack_spec_explains_capacity_reward_and_fields(self) -> None:
        spec = JumanjiBenchmark(JumanjiConfig(profile="knapsack")).spec
        self.assertIn("total weight at most 12.5", spec.description)
        self.assertIn("dense reward is that item's value", spec.description)
        self.assertIn("return equals the packed subset's total value", spec.description)
        self.assertEqual(spec.environment_parameters["total_capacity"], 12.5)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn(
            "12.5 minus the total weight", _dictionary(fields["action_mask"])["meaning"]
        )
        self.assertIn("already been packed", _dictionary(fields["packed_items"])["meaning"])
        self.assertIn("immediate reward", _dictionary(fields["values"])["meaning"])
        self.assertIn("contribution to capacity", _dictionary(fields["weights"])["meaning"])

    def test_tetris_spec_explains_dynamics_reward_and_upstream_step_fix(self) -> None:
        config = JumanjiConfig(profile="tetris")
        spec = JumanjiBenchmark(config).spec
        self.assertIn("lets the piece fall", spec.description)
        self.assertIn("crop the current 4x4 shape", spec.description)
        self.assertIn("0/40/100/300/1200", spec.description)
        self.assertIn("after 400 placed pieces", spec.description)
        meanings = _dictionary(_dictionary(spec.action_space)["component_value_meanings"])
        self.assertEqual(_dictionary(meanings["rotation"])["1"], "90_degrees_clockwise")
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("row 0 is the top", _dictionary(fields["grid"])["meaning"])
        self.assertIn("reports constant zero", _dictionary(fields["step_count"])["meaning"])
        self.assertIn("crop all-zero outer", _dictionary(fields["tetromino"])["meaning"])

        environment = JumanjiBenchmark(config).make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            if type(observation) is not dict:
                self.fail("expected an object observation")
            self.assertEqual(observation["step_count"], 0)
            action = _first_valid_action(observation, config=config)
            step = environment.step(action)
            self.assertIsInstance(step.observation, dict)
            if type(step.observation) is not dict:
                self.fail("expected an object observation")
            self.assertEqual(step.observation["step_count"], 1)
            self.assertIn(step.reward, (0.0, 40.0, 100.0, 300.0, 1_200.0))
            self.assertEqual(
                _dictionary(step.metrics)["lines_cleared"],
                (0.0, 40.0, 100.0, 300.0, 1_200.0).index(step.reward),
            )
        finally:
            environment.close()

    def test_cvrp_spec_and_feedback_explain_normalized_routing_state(self) -> None:
        config = JumanjiConfig(profile="cvrp")
        benchmark = JumanjiBenchmark(config)
        spec = benchmark.spec
        self.assertIn("customers 1-20 from depot 0", spec.description)
        self.assertIn("raw capacity 30", spec.description)
        self.assertIn("negative distance", spec.description)
        self.assertIn("returns to depot 0", spec.description)
        action_meanings = _dictionary(_dictionary(spec.action_space)["meaning"])
        self.assertEqual(action_meanings["0"], "depot")
        self.assertEqual(action_meanings["20"], "customer_20")
        self.assertEqual(spec.environment_parameters["raw_vehicle_capacity"], 30)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("depot resets this scalar", _dictionary(fields["capacity"])["meaning"])
        self.assertIn("Unused suffix slots", _dictionary(fields["trajectory"])["meaning"])
        self.assertIn(
            "not that it is an unvisited customer",
            _dictionary(fields["unvisited_nodes"])["meaning"],
        )

        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            self.assertIsInstance(initial, dict)
            if type(initial) is not dict:
                self.fail("expected an object observation")
            customer = _first_valid_action(initial, config=config)
            self.assertNotEqual(customer, 0)
            customer_step = environment.step(customer)
            depot_step = environment.step(0)
            feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=episode,
                        policy_seed=456,
                        initial_observation=initial,
                        transitions=(
                            Transition(action=customer, step=customer_step),
                            Transition(action=0, step=depot_step),
                        ),
                    ),
                )
            )
            trace = [
                json.loads(line) for line in feedback.artifacts[0].content.splitlines()
            ]
            initial_progress = trace[0]["initial_observation"]["semantics"]["progress"]
            customer_progress = trace[1]["result_observation"]["semantics"][
                "progress"
            ]
            depot_progress = trace[2]["result_observation"]["semantics"]["progress"]
            self.assertEqual(initial_progress["route"], [0])
            self.assertEqual(initial_progress["unvisited_customers"], 20)
            self.assertEqual(customer_progress["route"], [0, customer])
            self.assertEqual(customer_progress["visited_customers"], 1)
            self.assertFalse(customer_progress["at_depot"])
            self.assertEqual(depot_progress["route"], [0, customer, 0])
            self.assertEqual(depot_progress["depot_returns"], 1)
            self.assertEqual(depot_progress["remaining_capacity"], 1.0)
            self.assertAlmostEqual(
                depot_progress["distance_traveled"],
                2.0 * customer_progress["distance_traveled"],
            )
        finally:
            environment.close()

        full_episode = EpisodeSpec(environment_seed=321)
        environment = benchmark.make_environment(full_episode)
        try:
            initial = environment.reset()
            observation: PolicyValue = initial
            transitions: list[Transition] = []
            for _ in range(config.max_episode_steps):
                action = _first_valid_action(observation, config=config)
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
                observation = step.observation
            self.assertTrue(step.terminated)
            self.assertFalse(step.truncated)
            self.assertEqual(len(transitions), 40)

            feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=full_episode,
                        policy_seed=654,
                        initial_observation=initial,
                        transitions=tuple(transitions),
                    ),
                )
            )
            trace = [
                json.loads(line) for line in feedback.artifacts[0].content.splitlines()
            ]
            final_progress = trace[-1]["result_observation"]["semantics"]["progress"]
            self.assertEqual(final_progress["visited_customers"], 20)
            self.assertEqual(final_progress["unvisited_customers"], 0)
            self.assertEqual(final_progress["route_visits"], 40)
            self.assertEqual(final_progress["depot_returns"], 20)
            self.assertEqual(len(final_progress["route"]), 41)
            self.assertEqual(final_progress["route"][0], 0)
            self.assertEqual(final_progress["route"][-1], 0)
        finally:
            environment.close()

    def test_tsp_spec_and_feedback_explain_closed_tour_distance(self) -> None:
        config = JumanjiConfig(profile="tsp")
        benchmark = JumanjiBenchmark(config)
        spec = benchmark.spec
        self.assertIn("20 cities", spec.description)
        self.assertIn("first selected city starts the tour and rewards 0", spec.description)
        self.assertIn("distance back to the first city", spec.description)
        action_meanings = _dictionary(_dictionary(spec.action_space)["meaning"])
        self.assertEqual(action_meanings["0"], "city_0")
        self.assertEqual(action_meanings["19"], "city_19")
        self.assertEqual(spec.environment_parameters["city_count"], 20)
        self.assertIs(spec.environment_parameters["tour_returns_to_first_city"], True)
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("not yet been selected", _dictionary(fields["action_mask"])["meaning"])
        self.assertIn(
            "-1 before the first selection", _dictionary(fields["position"])["meaning"]
        )
        self.assertIn(
            "Unused suffix entries are -1", _dictionary(fields["trajectory"])["meaning"]
        )

        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            self.assertIsInstance(initial, dict)
            if type(initial) is not dict:
                self.fail("expected an object observation")
            coordinates_value = initial["coordinates"]
            self.assertIsInstance(coordinates_value, TensorValue)
            if type(coordinates_value) is not TensorValue:
                self.fail("expected TensorValue coordinates")
            coordinates = numpy.frombuffer(
                coordinates_value.data,
                dtype="<f4",
            ).reshape(20, 2)
            self.assertEqual(initial["position"], -1)

            observation: PolicyValue = initial
            transitions: list[Transition] = []
            for _ in range(config.max_episode_steps):
                action = _first_valid_action(observation, config=config)
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
                observation = step.observation
            self.assertEqual(transitions[0].step.reward, 0.0)
            self.assertTrue(step.terminated)
            self.assertFalse(step.truncated)
            self.assertEqual(len(transitions), 20)

            route = [_integer(transition.action) for transition in transitions]
            closed_route = numpy.asarray((*route, route[0]), dtype=numpy.int64)
            tour_length = float(
                numpy.linalg.norm(
                    coordinates[closed_route[1:]]
                    - coordinates[closed_route[:-1]],
                    axis=1,
                ).sum()
            )
            total_reward = sum(transition.step.reward for transition in transitions)
            self.assertAlmostEqual(total_reward, -tour_length, places=5)

            feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=episode,
                        policy_seed=456,
                        initial_observation=initial,
                        transitions=tuple(transitions),
                    ),
                )
            )
            trace = [
                json.loads(line) for line in feedback.artifacts[0].content.splitlines()
            ]
            initial_progress = trace[0]["initial_observation"]["semantics"]["progress"]
            final_progress = trace[-1]["result_observation"]["semantics"]["progress"]
            self.assertEqual(initial_progress["current_node"], -1)
            self.assertEqual(initial_progress["route"], [])
            self.assertEqual(initial_progress["unvisited_nodes"], 20)
            self.assertEqual(final_progress["route"], route)
            self.assertEqual(final_progress["visited_nodes"], 20)
            self.assertEqual(final_progress["unvisited_nodes"], 0)
            self.assertTrue(final_progress["tour_complete"])
            self.assertAlmostEqual(
                final_progress["rewarded_distance"],
                tour_length,
                places=5,
            )
            self.assertAlmostEqual(
                final_progress["tour_length_if_closed_now"],
                tour_length,
                places=5,
            )
        finally:
            environment.close()

    def test_snake_spec_and_feedback_explain_channels_growth_and_positions(self) -> None:
        config = JumanjiConfig(profile="snake")
        benchmark = JumanjiBenchmark(config)
        spec = benchmark.spec
        self.assertIn("12x12 grid", spec.description)
        self.assertIn("Episode return equals", spec.description)
        self.assertIn("fruits eaten", spec.description)
        self.assertIn("current tail cell", spec.description)
        self.assertEqual(
            _dictionary(spec.action_space)["meaning"],
            {"0": "up", "1": "right", "2": "down", "3": "left"},
        )
        self.assertEqual(spec.environment_parameters["grid_cells"], 144)
        self.assertEqual(spec.environment_parameters["time_limit"], 4_000)
        self.assertEqual(
            spec.environment_parameters["grid_channels"],
            ["body", "head", "tail", "fruit", "normalized_body_order"],
        )
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("tail advances", _dictionary(fields["action_mask"])["meaning"])
        grid_spec = _dictionary(fields["grid"])
        self.assertIn("Channels 0/1/2/3", grid_spec["meaning"])
        self.assertIn("1/length at the tail", grid_spec["meaning"])

        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            observation = initial
            transitions: list[Transition] = []
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            for _ in range(24):
                self.assertIsInstance(observation, dict)
                if type(observation) is not dict:
                    self.fail("expected an object observation")
                grid_value = observation["grid"]
                mask_value = observation["action_mask"]
                self.assertIsInstance(grid_value, TensorValue)
                self.assertIsInstance(mask_value, TensorValue)
                if type(grid_value) is not TensorValue:
                    self.fail("expected TensorValue grid")
                if type(mask_value) is not TensorValue:
                    self.fail("expected TensorValue action mask")
                grid = numpy.frombuffer(grid_value.data, dtype="<f4").reshape(
                    grid_value.shape
                )
                head = numpy.argwhere(grid[..., 1] > 0.5)[0]
                fruit = numpy.argwhere(grid[..., 3] > 0.5)[0]
                legal_actions = [
                    action for action, legal in enumerate(mask_value.data) if legal
                ]
                action = min(
                    legal_actions,
                    key=lambda candidate: abs(
                        int(head[0]) + moves[candidate][0] - int(fruit[0])
                    )
                    + abs(int(head[1]) + moves[candidate][1] - int(fruit[1])),
                )
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.reward == 1.0:
                    break
                self.assertFalse(step.done)
                observation = step.observation
            self.assertEqual(step.reward, 1.0)
            self.assertFalse(step.done)

            feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=episode,
                        policy_seed=456,
                        initial_observation=initial,
                        transitions=tuple(transitions),
                    ),
                )
            )
            trace = [
                json.loads(line) for line in feedback.artifacts[0].content.splitlines()
            ]
            initial_progress = trace[0]["initial_observation"]["semantics"]["progress"]
            final_progress = trace[-1]["result_observation"]["semantics"]["progress"]
            self.assertEqual(initial_progress["snake_length"], 1)
            self.assertEqual(initial_progress["fruits_eaten"], 0)
            self.assertEqual(
                initial_progress["head_position"],
                initial_progress["tail_position"],
            )
            self.assertEqual(final_progress["snake_length"], 2)
            self.assertEqual(final_progress["fruits_eaten"], 1)
            self.assertEqual(final_progress["free_cells"], 142)
            self.assertEqual(len(final_progress["body_path_head_to_tail"]), 2)
            self.assertEqual(
                final_progress["body_path_head_to_tail"][0],
                final_progress["head_position"],
            )
            self.assertEqual(
                final_progress["body_path_head_to_tail"][-1],
                final_progress["tail_position"],
            )
        finally:
            environment.close()

    def test_pacman_spec_matches_real_actions_rewards_coordinates_and_terminal_reason(
        self,
    ) -> None:
        config = JumanjiConfig(profile="pacman")
        benchmark = JumanjiBenchmark(config)
        spec = benchmark.spec
        self.assertIn("actions 0/1/2/3 to up/left/down/right", spec.description)
        self.assertIn("for 60 total", spec.description)
        self.assertIn("at most once per ghost", spec.description)
        self.assertEqual(
            _dictionary(spec.action_space)["meaning"],
            {
                "0": "up",
                "1": "left",
                "2": "down",
                "3": "right",
                "4": "no_op",
            },
        )
        self.assertEqual(spec.environment_parameters["initial_pellets"], 318)
        self.assertEqual(spec.environment_parameters["power_up_extra_reward"], 50)
        self.assertEqual(spec.environment_parameters["maze_layout"], "fixed")
        fields = _dictionary(_dictionary(spec.observation_space)["fields"])
        self.assertIn("Entry 4", _dictionary(fields["action_mask"])["meaning"])
        self.assertIn(
            "grid row index", _dictionary(fields["player_locations.x"])["meaning"]
        )
        self.assertIn("[column, row]", _dictionary(fields["pellet_locations"])["meaning"])
        self.assertIn(
            "continues below zero",
            _dictionary(fields["frightened_state_time"])["meaning"],
        )

        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            self.assertIsInstance(initial, dict)
            if type(initial) is not dict:
                self.fail("expected an object observation")
            player = initial["player_locations"]
            self.assertIsInstance(player, dict)
            if type(player) is not dict:
                self.fail("expected an object player position")
            self.assertEqual([player["x"], player["y"]], [23, 13])
            mask = initial["action_mask"]
            self.assertIsInstance(mask, TensorValue)
            if type(mask) is not TensorValue:
                self.fail("expected TensorValue action mask")
            self.assertEqual(list(mask.data), [0, 1, 0, 1, 0])

            left_step = environment.step(1)
            left_observation = _dictionary(left_step.observation)
            left_player = left_observation["player_locations"]
            self.assertIsInstance(left_player, dict)
            if type(left_player) is not dict:
                self.fail("expected an object player position")
            self.assertEqual([left_player["x"], left_player["y"]], [23, 12])
            self.assertEqual(left_step.reward, 10.0)
            self.assertEqual(left_observation["score"], 10)
            self.assertEqual(left_observation["frightened_state_time"], -1)

            feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=episode,
                        policy_seed=456,
                        initial_observation=initial,
                        transitions=(Transition(action=1, step=left_step),),
                    ),
                )
            )
            trace = [
                json.loads(line) for line in feedback.artifacts[0].content.splitlines()
            ]
            initial_progress = trace[0]["initial_observation"]["semantics"]["progress"]
            result_progress = trace[1]["result_observation"]["semantics"]["progress"]
            self.assertEqual(initial_progress["player_position_row_column"], [23, 13])
            self.assertEqual(initial_progress["remaining_pellets"], 318)
            self.assertEqual(initial_progress["remaining_power_ups"], 4)
            self.assertEqual(result_progress["player_position_row_column"], [23, 12])
            self.assertEqual(result_progress["pellets_collected"], 1)
            self.assertEqual(result_progress["score"], 10)
            self.assertEqual(result_progress["frightened_timer_raw"], -1)
            self.assertEqual(result_progress["frightened_steps_remaining"], 0)
            self.assertFalse(result_progress["ghosts_edible"])
            self.assertEqual(len(result_progress["ghost_positions_row_column"]), 4)
        finally:
            environment.close()

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            right_step = environment.step(3)
            right_player = _dictionary(right_step.observation)["player_locations"]
            self.assertIsInstance(right_player, dict)
            if type(right_player) is not dict:
                self.fail("expected an object player position")
            self.assertEqual([right_player["x"], right_player["y"]], [23, 14])
        finally:
            environment.close()

        terminal_episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(terminal_episode)
        try:
            terminal_initial = environment.reset()
            observation = terminal_initial
            terminal_transitions: list[Transition] = []
            for _ in range(config.max_episode_steps):
                action = _first_valid_action(observation, config=config)
                step = environment.step(action)
                terminal_transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
                observation = step.observation
            self.assertTrue(step.terminated)
            self.assertEqual(_dictionary(step.metrics)["terminal_reason"], "ghost_collision")
            terminal_feedback = benchmark.feedback(
                (
                    EpisodeRecord(
                        episode=terminal_episode,
                        policy_seed=789,
                        initial_observation=terminal_initial,
                        transitions=tuple(terminal_transitions),
                    ),
                )
            )
            summary = _dictionary(
                _sequence(_dictionary(terminal_feedback.content)["episode_summaries"])[0]
            )
            summary_metrics = _dictionary(summary["metrics"])
            terminal_reason = _dictionary(summary_metrics["terminal_reason"])
            self.assertEqual(
                terminal_reason["final"],
                "ghost_collision",
            )
        finally:
            environment.close()

    def test_both_sudoku_profiles_publish_database_semantics_and_solve_real_puzzles(
        self,
    ) -> None:
        variants = (
            ("sudoku", 10_000, 25, 77),
            ("sudoku-very-easy", 1_000, 46, 80),
        )
        for profile, database_size, minimum_clues, maximum_clues in variants:
            with self.subTest(profile=profile):
                config = JumanjiConfig(profile=profile)
                benchmark = JumanjiBenchmark(config)
                spec = benchmark.spec
                self.assertIn(f"fixed {database_size:,}-puzzle", spec.description)
                self.assertIn("internal value 0-8 means human symbol 1-9", spec.description)
                self.assertIn("Episode return is binary", spec.description)
                self.assertEqual(
                    spec.environment_parameters["puzzle_database_size"],
                    database_size,
                )
                self.assertEqual(
                    spec.environment_parameters["minimum_initial_clues"],
                    minimum_clues,
                )
                self.assertEqual(
                    spec.environment_parameters["maximum_initial_clues"],
                    maximum_clues,
                )
                value_meanings = _dictionary(
                    _dictionary(spec.action_space)["component_value_meanings"]
                )
                value_labels = _dictionary(value_meanings["value"])
                self.assertEqual(value_labels["0"], "human_symbol_1")
                self.assertEqual(value_labels["8"], "human_symbol_9")
                fields = _dictionary(_dictionary(spec.observation_space)["fields"])
                self.assertIn(
                    "human Sudoku symbols 1-9", _dictionary(fields["board"])["meaning"]
                )
                self.assertIn("3x3 box", _dictionary(fields["action_mask"])["meaning"])

                episode = EpisodeSpec(environment_seed=123)
                environment = benchmark.make_environment(episode)
                try:
                    initial = environment.reset()
                    self.assertIsInstance(initial, dict)
                    if type(initial) is not dict:
                        self.fail("expected an object observation")
                    board_value = initial["board"]
                    self.assertIsInstance(board_value, TensorValue)
                    if type(board_value) is not TensorValue:
                        self.fail("expected TensorValue Sudoku board")
                    board = numpy.frombuffer(board_value.data, dtype="<i4").reshape(9, 9)
                    clue_count = int(numpy.count_nonzero(board >= 0))
                    self.assertGreaterEqual(clue_count, minimum_clues)
                    self.assertLessEqual(clue_count, maximum_clues)
                    solution = _solve_sudoku(board)

                    transitions: list[Transition] = []
                    for row, column in numpy.argwhere(board < 0):
                        action: PolicyValue = [
                            int(row),
                            int(column),
                            int(solution[int(row), int(column)]),
                        ]
                        step = environment.step(action)
                        transitions.append(Transition(action=action, step=step))
                    self.assertTrue(step.terminated)
                    self.assertFalse(step.truncated)
                    self.assertEqual(step.reward, 1.0)
                    self.assertTrue(
                        all(
                            transition.step.reward == 0.0
                            for transition in transitions[:-1]
                        )
                    )
                    self.assertEqual(len(transitions), 81 - clue_count)

                    feedback = benchmark.feedback(
                        (
                            EpisodeRecord(
                                episode=episode,
                                policy_seed=456,
                                initial_observation=initial,
                                transitions=tuple(transitions),
                            ),
                        )
                    )
                    trace = [
                        json.loads(line)
                        for line in feedback.artifacts[0].content.splitlines()
                    ]
                    initial_progress = trace[0]["initial_observation"]["semantics"][
                        "progress"
                    ]
                    final_transition = next(
                        line
                        for line in reversed(trace)
                        if line["type"] == "transition"
                    )
                    final_progress = final_transition["result_observation"]["semantics"][
                        "progress"
                    ]
                    self.assertEqual(initial_progress["filled_cells"], clue_count)
                    self.assertEqual(len(initial_progress["candidate_counts"]), 9)
                    self.assertEqual(final_progress["filled_cells"], 81)
                    self.assertEqual(final_progress["empty_cells"], 0)
                    self.assertEqual(final_progress["legal_assignments"], 0)
                    self.assertEqual(final_progress["zero_candidate_empty_cells"], 0)
                    self.assertTrue(final_progress["solved"])
                    self.assertIn(
                        "human_symbol=",
                        final_transition["action_meaning"],
                    )
                    self.assertEqual(feedback.score, 1.0)
                finally:
                    environment.close()

    def test_invalid_profile_and_actions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JumanjiConfig(profile="unknown")
        with self.assertRaises(TypeError):
            JumanjiConfig(profile=1)  # type: ignore[arg-type]

        environment = JumanjiBenchmark().make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = JumanjiBenchmark(
            JumanjiConfig(profile="minesweeper")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([10, 0])
        finally:
            environment.close()

    def test_masked_action_is_rejected_before_upstream_step(self) -> None:
        config = JumanjiConfig(profile="minesweeper")
        environment = JumanjiBenchmark(config).make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            action = _first_valid_action(observation, config=config)
            step = environment.step(action)
            if not step.done:
                with self.assertRaises(InvalidAction):
                    environment.step(action)
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            JumanjiBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"profile": "tetris"})
            )

    def test_baseline_is_packaged(self) -> None:
        program = baseline_program()
        self.assertIn("policy.py", program.files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            JumanjiBenchmark(JumanjiConfig(profile="rubiks-cube-partly-scrambled")),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    ([0, 0, 0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_feedback_bounds_long_trace_and_keeps_reward_events(self) -> None:
        config = JumanjiConfig(profile="maze")
        initial = _maze_observation(0)
        transitions = tuple(
            Transition(
                action=step_index % 2,
                step=Step(
                    observation=_maze_observation(step_index + 1),
                    reward=1.0 if step_index in {20, 40} else 0.0,
                    terminated=False,
                    truncated=step_index == 59,
                    metrics={"step_count": step_index + 1},
                ),
            )
            for step_index in range(60)
        )
        feedback = JumanjiBenchmark(config).feedback(
            (
                EpisodeRecord(
                    episode=EpisodeSpec(environment_seed=123),
                    policy_seed=456,
                    initial_observation=initial,
                    transitions=transitions,
                ),
            )
        )
        feedback_content = _dictionary(feedback.content)
        self.assertEqual(feedback_content["traced_steps"], 48)
        self.assertEqual(feedback_content["trace_steps_omitted"], 12)
        lines = [
            json.loads(line) for line in feedback.artifacts[0].content.splitlines()
        ]
        traced = [line for line in lines if line["type"] == "transition"]
        indices = [line["step_index"] for line in traced]
        self.assertEqual(len(indices), 48)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 59)
        self.assertIn(20, indices)
        self.assertIn(40, indices)
        self.assertTrue(all(line["action_was_legal"] for line in traced))
        mask = traced[0]["decision_observation"]["semantics"]["action_mask"]
        self.assertEqual(mask["layout"], "discrete")
        self.assertEqual(mask["legal_action_samples"], [0, 1])
        with numpy.load(
            io.BytesIO(feedback.artifacts[1].content),
            allow_pickle=False,
        ) as archive:
            self.assertEqual(len(archive["step_indices"]), 48)
            self.assertEqual(archive["decision__walls"].shape, (48, 10, 10))


def _maze_observation(step: int) -> PolicyValue:
    return {
        "agent_position": {"row": step % 10, "col": (step // 10) % 10},
        "target_position": {"row": 9, "col": 9},
        "walls": TensorValue(dtype="bool", shape=(10, 10), data=bytes(100)),
        "action_mask": TensorValue(
            dtype="bool",
            shape=(4,),
            data=bytes((1, 1, 0, 0)),
        ),
        "step_count": step,
    }


def _dictionary(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _sequence(value: object) -> list[Any]:
    assert isinstance(value, list)
    return value


def _integer(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _solve_sudoku(board: NDArray[numpy.int32]) -> NDArray[numpy.int32]:
    solution = board.copy()

    def search() -> bool:
        best: tuple[int, int, set[int]] | None = None
        for row, column in numpy.argwhere(solution < 0):
            row_index = int(row)
            column_index = int(column)
            box_row = (row_index // 3) * 3
            box_column = (column_index // 3) * 3
            used = {
                int(value)
                for value in (
                    *solution[row_index, :],
                    *solution[:, column_index],
                    *solution[
                        box_row : box_row + 3,
                        box_column : box_column + 3,
                    ].reshape(-1),
                )
                if value >= 0
            }
            candidates = set(range(9)) - used
            if not candidates:
                return False
            if best is None or len(candidates) < len(best[2]):
                best = (row_index, column_index, candidates)
        if best is None:
            return True
        row_index, column_index, candidates = best
        for value in sorted(candidates):
            solution[row_index, column_index] = value
            if search():
                return True
        solution[row_index, column_index] = -1
        return False

    if not search():
        raise AssertionError("packaged Sudoku puzzle has no solution")
    return solution


def _inverse_rubik_scramble(seed: int, scramble_moves: int) -> list[PolicyValue]:
    key = jax.random.PRNGKey(seed & 0xFFFF_FFFF)
    key = jax.random.fold_in(key, (seed >> 32) & 0xFFFF_FFFF)
    _, scramble_key = jax.random.split(key)
    flattened = numpy.asarray(
        jax.random.randint(
            key=scramble_key,
            minval=0,
            maxval=18,
            shape=(scramble_moves,),
            dtype=jnp.int32,
        )
    )
    inverse_rotation = (1, 0, 2)
    actions: list[PolicyValue] = []
    for flattened_action in reversed(flattened.tolist()):
        face, rotation = divmod(int(flattened_action), 3)
        actions.append([face, 0, inverse_rotation[rotation]])
    return actions


def _policy_carriers(value: PolicyValue, *, path: str = "") -> dict[str, str]:
    if type(value) is dict:
        carriers: dict[str, str] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            carriers.update(_policy_carriers(item, path=child_path))
        return carriers
    if type(value) is TensorValue:
        return {path: "TensorValue"}
    if type(value) is bool:
        return {path: "bool"}
    if type(value) is int:
        return {path: "int"}
    if type(value) is float:
        return {path: "float"}
    raise AssertionError(f"unexpected Policy carrier at {path}: {type(value).__name__}")


def _first_valid_action(observation: PolicyValue, *, config: JumanjiConfig) -> PolicyValue:
    if type(observation) is not dict:
        raise AssertionError("expected an object observation")
    mask = observation.get("action_mask")
    if config.action_kind == "discrete":
        if type(mask) is not TensorValue or mask.dtype != "bool":
            if not config.has_action_mask:
                return 0
            raise AssertionError("expected a boolean action mask")
        return _first(mask.data)
    if not config.has_action_mask:
        return [0] * len(config.action_num_values)
    if type(mask) is not TensorValue or mask.dtype != "bool":
        raise AssertionError("expected a boolean action mask")
    if mask.shape == config.action_num_values:
        flat_index = _first(mask.data)
        return _unravel(flat_index, mask.shape)
    if (
        len(set(config.action_num_values)) == 1
        and mask.shape == (len(config.action_num_values), config.action_num_values[0])
    ):
        width = config.action_num_values[0]
        return [
            _first(mask.data[index * width : (index + 1) * width])
            for index in range(len(config.action_num_values))
        ]
    raise AssertionError(f"unexpected action mask shape: {mask.shape}")


def _first(values: bytes) -> int:
    for index, valid in enumerate(values):
        if valid:
            return index
    raise AssertionError("action mask has no valid action")


def _unravel(flat_index: int, shape: tuple[int, ...]) -> PolicyValue:
    result: list[PolicyValue] = [0 for _ in shape]
    for index in range(len(shape) - 1, -1, -1):
        flat_index, coordinate = divmod(flat_index, shape[index])
        result[index] = coordinate
    return result


if __name__ == "__main__":
    unittest.main()
