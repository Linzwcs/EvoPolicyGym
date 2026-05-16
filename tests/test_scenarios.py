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
        self.assertIn("cartpole_balance", scenarios)
        self.assertIn("lunar_lander", scenarios)
        self.assertIn("lunar_lander_continuous", scenarios)
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
        self.assertEqual(scenario.observation_mode, "jsonable")
        self.assertEqual(contract.observation_schema["type"], "box")
        self.assertEqual(contract.observation_schema["shape"], [2])
        self.assertEqual([item["name"] for item in contract.observation_schema["dimensions"]], ["position", "velocity"])


if __name__ == "__main__":
    unittest.main()
