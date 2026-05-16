from __future__ import annotations

import unittest

from hlbench.core.scenario import find_scenario_dir, list_scenarios
from hlbench.core.validate import validate_scenario
from hlbench.envs.registry import get_backend
from hlbench.harness.evaluator import compile_policy


class ScenarioCatalogTest(unittest.TestCase):
    def test_builtin_scenarios_validate_and_compile(self) -> None:
        scenarios = list_scenarios()
        self.assertIn("acrobot_swingup", scenarios)
        self.assertIn("bipedal_walker", scenarios)
        self.assertIn("car_racing", scenarios)
        self.assertIn("cartpole_balance", scenarios)
        self.assertIn("lunar_lander", scenarios)
        self.assertIn("lunar_lander_continuous", scenarios)
        self.assertIn("minigrid_doorkey_16x16", scenarios)
        self.assertIn("minigrid_keycorridor_s6r3", scenarios)
        self.assertIn("minigrid_lavacrossing_s11n5", scenarios)
        self.assertIn("minigrid_obstructedmaze_2dlhb", scenarios)
        self.assertIn("mountain_car", scenarios)
        self.assertIn("pendulum_swingup", scenarios)

        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                validation = validate_scenario(scenario)
                self.assertTrue(validation.ok, validation.to_record())

                policy_path = find_scenario_dir(scenario) / "policy.py"
                compile_result = compile_policy(policy_path)
                self.assertTrue(compile_result.ok, compile_result.to_record())

    def test_mountain_car_uses_classic_control_telemetry_observation(self) -> None:
        from hlbench.core.scenario import load_scenario

        scenario = load_scenario("mountain_car")
        contract = get_backend(scenario.env_backend).describe(scenario)
        self.assertEqual(scenario.observation_type, "state")
        self.assertEqual(scenario.to_record()["observation_type"], "state")
        self.assertEqual(scenario.observation_mode, "jsonable")
        self.assertEqual(contract.observation_schema["type"], "box")
        self.assertEqual(contract.observation_schema["shape"], [2])
        self.assertEqual([item["name"] for item in contract.observation_schema["dimensions"]], ["position", "velocity"])

    def test_car_racing_uses_image_artifact_observation(self) -> None:
        from hlbench.core.scenario import load_scenario

        scenario = load_scenario("car_racing")
        contract = get_backend(scenario.env_backend).describe(scenario)
        self.assertEqual(scenario.observation_type, "image")
        self.assertEqual(scenario.observation_mode, "image_artifact")
        self.assertEqual(contract.observation_schema["type"], "dict")
        self.assertEqual(contract.observation_schema["mode"], "image_artifact")
        self.assertEqual(contract.observation_schema["image_schema"]["shape"], [96, 96, 3])

    def test_minigrid_uses_symbolic_public_observation(self) -> None:
        from hlbench.core.scenario import load_scenario

        scenario = load_scenario("minigrid_doorkey_16x16")
        contract = get_backend(scenario.env_backend).describe(scenario)
        self.assertEqual(scenario.observation_type, "symbolic")
        self.assertEqual(scenario.observation_mode, "minigrid_public")
        self.assertEqual(contract.observation_schema["type"], "dict")
        self.assertEqual(contract.observation_schema["mode"], "minigrid_public")
        self.assertEqual(sorted(contract.observation_schema["fields"]), ["action_count", "direction", "image", "mission"])
        self.assertEqual(contract.observation_schema["fields"]["image"]["shape"], [7, 7, 3])
        self.assertEqual(contract.action_schema["n"], 7)


if __name__ == "__main__":
    unittest.main()
