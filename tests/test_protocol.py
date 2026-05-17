from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from hlbench.harness.agents.command import CommandAgent
from hlbench.harness.agents.config import resolve_agent_config
from hlbench.harness.epoch_runner import run_epoch
from hlbench.harness.loop_runner import run_loop
from hlbench.rollout.cli import main as rollout_main
from hlbench.workspace.create import create_workspace


class HarnessProtocolTest(unittest.TestCase):
    def test_epoch_zero_uses_submission_evaluation_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_epoch(
                scenario_name="cartpole_balance",
                run_id="protocol-layout",
                workspace_root=root / "workspace",
                epoch_dir=root / "epoch_000",
                reset_workspace=True,
                train_episodes=1,
                timeout_seconds=30,
            )

            epoch_dir = root / "epoch_000"
            self.assertTrue((epoch_dir / "input" / "policy.py").is_file())
            self.assertTrue((epoch_dir / "submission" / "policy.py").is_file())
            self.assertTrue((epoch_dir / "submission" / "agent.json").is_file())
            self.assertTrue((epoch_dir / "submission" / "stdout.txt").is_file())
            self.assertTrue((epoch_dir / "submission" / "stderr.txt").is_file())
            self.assertTrue((epoch_dir / "submission" / "compile.json").is_file())
            self.assertTrue((epoch_dir / "evaluation" / "train").is_dir())
            self.assertTrue((epoch_dir / "evaluation" / "validation").is_dir())
            self.assertTrue((epoch_dir / "evaluation" / "heldout").is_dir())
            self.assertFalse((epoch_dir / "evaluator").exists())

            transition = _read_json(epoch_dir / "transition.json")
            self.assertEqual(transition["input"]["feedback_source"]["kind"], "none")
            self.assertNotIn("before", transition)
            self.assertNotIn("after", transition)
            self.assertIn("submission", transition)
            self.assertIn("evaluation", transition)
            self.assertIn("comparison", transition)
            agent = transition["submission"]["agent"]
            self.assertEqual(agent["backend"], "command")
            self.assertEqual(agent["name"], "none")
            self.assertEqual(agent["stdout_path"], "stdout.txt")
            self.assertEqual(agent["stderr_path"], "stderr.txt")
            self.assertNotIn("stdout", agent)
            self.assertNotIn("stderr", agent)
            self.assertEqual((epoch_dir / "submission" / "stdout.txt").read_text(), "")
            self.assertEqual((epoch_dir / "submission" / "stderr.txt").read_text(), "agent command skipped")

    def test_workspace_feedback_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_epoch(
                scenario_name="cartpole_balance",
                run_id="feedback-boundary",
                workspace_root=root / "workspace",
                epoch_dir=root / "epoch_000",
                reset_workspace=True,
                train_episodes=1,
                timeout_seconds=30,
            )

            feedback_dir = root / "workspace" / "feedback"
            self.assertTrue((feedback_dir / "current" / "summary.json").is_file())
            self.assertTrue((feedback_dir / "current" / "episodes.jsonl").is_file())
            self.assertTrue((feedback_dir / "current" / "replays").is_dir())
            self.assertTrue((feedback_dir / "history" / "epoch_000" / "train" / "summary.json").is_file())
            self.assertTrue((feedback_dir / "history" / "epoch_000" / "validation_summary.json").is_file())

            first_episode = _read_jsonl(feedback_dir / "current" / "episodes.jsonl")[0]
            self.assertEqual(first_episode["replay_path"], "replays/episode_0000.jsonl")

            for path in feedback_dir.rglob("*"):
                relative = str(path.relative_to(feedback_dir))
                self.assertNotIn("heldout", relative.lower())
                if path.is_file():
                    content = path.read_text()
                    self.assertNotIn("heldout", content.lower())
                    self.assertNotIn(str(root), content)
                    self.assertNotIn("evaluation", content.lower())
                    self.assertNotIn("private", content.lower())

            for split in ("validation", "heldout"):
                split_dir = root / "epoch_000" / "evaluation" / split
                self.assertTrue((split_dir / "summary.json").is_file())
                self.assertTrue((split_dir / "manifest.json").is_file())
                self.assertFalse((split_dir / "episodes.jsonl").exists())
                self.assertFalse((split_dir / "failures.jsonl").exists())
                self.assertFalse((split_dir / "replays").exists())

    def test_workspace_rollout_rejects_private_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = create_workspace(
                scenario_name="cartpole_balance",
                output_dir=Path(tmp) / "workspace",
                overwrite=True,
            )

            with self.assertRaises(SystemExit) as raised:
                rollout_main(["--workspace", str(workspace.root), "--split", "validation"])
            self.assertIn("workspace rollouts may only use --split train", str(raised.exception))

    def test_task_markdown_is_self_contained_for_environment_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = create_workspace(
                scenario_name="minigrid_doorkey_16x16",
                output_dir=Path(tmp) / "workspace",
                overwrite=True,
            )

            task_md = (workspace.root / "task.md").read_text()
            self.assertIn("## Environment Loop", task_md)
            self.assertIn("## Observation", task_md)
            self.assertIn("## Actions", task_md)
            self.assertIn("## Termination", task_md)
            self.assertIn("image_cell_encoding", task_md)
            self.assertIn('"door": 4', task_md)
            self.assertIn('"key": 5', task_md)
            self.assertIn("`0 / turn_left`", task_md)
            self.assertIn("Success threshold used for `success_rate`", task_md)

    def test_workspace_agents_md_explains_tools_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = create_workspace(
                scenario_name="cartpole_balance",
                output_dir=Path(tmp) / "workspace",
                overwrite=True,
            )

            agents_md = (workspace.root / "AGENTS.md").read_text()
            self.assertIn("Use `tools/` for reusable analysis helpers", agents_md)
            self.assertIn("Write helper outputs, scratch files, notes", agents_md)
            self.assertIn("Do not use `tools/` or `experiments/` to modify the evaluator", agents_md)

    def test_agent_command_streams_are_written_as_submission_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_epoch(
                scenario_name="cartpole_balance",
                run_id="agent-streams",
                agent_preset="none",
                agent_command=[
                    sys.executable,
                    "-c",
                    "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr)",
                ],
                workspace_root=root / "workspace",
                epoch_dir=root / "epoch_000",
                reset_workspace=True,
                train_episodes=1,
                timeout_seconds=30,
            )

            submission_dir = root / "epoch_000" / "submission"
            agent = result.to_record()["submission"]["agent"]
            self.assertEqual(agent["name"], "custom")
            self.assertEqual(agent["command"][0], sys.executable)
            self.assertEqual(agent["stdout_path"], "stdout.txt")
            self.assertEqual(agent["stderr_path"], "stderr.txt")
            self.assertNotIn("stdout", agent)
            self.assertNotIn("stderr", agent)
            self.assertEqual((submission_dir / "stdout.txt").read_text(), "hello stdout\n")
            self.assertEqual((submission_dir / "stderr.txt").read_text(), "hello stderr\n")

    def test_agent_presets_resolve_to_command_backend(self) -> None:
        none = resolve_agent_config(preset="none")
        self.assertEqual(none.backend, "command")
        self.assertEqual(none.command, ())

        codex = resolve_agent_config(preset="codex")
        self.assertEqual(codex.command, ("codex", "exec"))

        custom = resolve_agent_config(preset="claude", command=["echo", "ok"])
        self.assertEqual(custom.name, "claude")
        self.assertEqual(custom.command, ("echo", "ok"))

    def test_command_agent_timeout_outputs_are_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CommandAgent(
                command=[sys.executable, "-c", "import time; time.sleep(2)"],
                timeout_seconds=1,
            ).run(Path(tmp))
            self.assertTrue(result.timed_out)
            self.assertIsInstance(result.stdout, str)
            self.assertIsInstance(result.stderr, str)

    def test_loop_writes_report_artifacts_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_loop(
                scenario_name="cartpole_balance",
                run_id="report-artifacts",
                model_name="test-model",
                epochs=2,
                train_episodes=1,
                timeout_seconds=30,
                base_root=root / "run",
            )

            run_dir = root / "run"
            self.assertEqual(result.run_dir, run_dir)
            self.assertTrue((run_dir / "transitions.jsonl").is_file())
            self.assertTrue((run_dir / "learning_curve.json").is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertTrue((run_dir / "report" / "index.html").is_file())
            self.assertTrue((run_dir / "report" / "metrics.json").is_file())
            self.assertTrue((run_dir / "report" / "learning_curve.json").is_file())
            self.assertTrue((run_dir / "report" / "learning_curve.svg").is_file())
            self.assertTrue((run_dir / "report" / "transitions.html").is_file())

            transitions = _read_jsonl(run_dir / "transitions.jsonl")
            curve = json.loads((run_dir / "learning_curve.json").read_text())
            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(len(transitions), 2)
            self.assertEqual(len(curve), 2)
            self.assertEqual(metrics["epochs"], 2)
            self.assertIn("heldout_return_auc", metrics["primary"])
            self.assertEqual(curve[0]["transition"], "epochs/epoch_000/transition.json")
            self.assertEqual(curve[1]["transition"], "epochs/epoch_001/transition.json")

            workspace = run_dir / "workspace"
            self.assertFalse((workspace / "report").exists())
            self.assertFalse((workspace / "metrics.json").exists())
            self.assertFalse((workspace / "learning_curve.json").exists())

    def test_compile_error_receives_minimum_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_epoch(
                scenario_name="cartpole_balance",
                run_id="compile-error",
                command=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('system/policy.py').write_text('def broken(:\\n')",
                ],
                workspace_root=root / "workspace",
                epoch_dir=root / "epoch_000",
                reset_workspace=True,
                train_episodes=1,
                timeout_seconds=30,
            )

            record = result.to_record()
            self.assertFalse(record["submission"]["compile"]["ok"])
            self.assertTrue(record["comparison"]["reward"]["invalid"])
            self.assertEqual(record["comparison"]["reward"]["reward"], 0.0)
            for split in ("train", "validation", "heldout"):
                self.assertEqual(record["evaluation"][split]["summary"]["mean_score"], 0.0)
                self.assertEqual(
                    record["evaluation"][split]["summary"]["minimum_score_episodes"],
                    record["evaluation"][split]["summary"]["episodes"],
                )

    def test_policy_runtime_error_marks_transition_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_epoch(
                scenario_name="cartpole_balance",
                run_id="runtime-error",
                command=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('system/policy.py').write_text('class Policy:\\n    def act(self, observation):\\n        return 0\\n')",
                ],
                workspace_root=root / "workspace",
                epoch_dir=root / "epoch_000",
                reset_workspace=True,
                train_episodes=1,
                timeout_seconds=30,
            )

            record = result.to_record()
            self.assertTrue(record["submission"]["compile"]["ok"])
            self.assertTrue(record["comparison"]["reward"]["invalid"])
            self.assertTrue(record["comparison"]["reward"]["minimum_score_applied"])
            self.assertEqual(record["comparison"]["reward"]["reward"], 0.0)
            for split in ("train", "validation", "heldout"):
                self.assertEqual(record["evaluation"][split]["summary"]["mean_score"], 0.0)
                self.assertEqual(
                    record["evaluation"][split]["summary"]["minimum_score_episodes"],
                    record["evaluation"][split]["summary"]["episodes"],
                )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
