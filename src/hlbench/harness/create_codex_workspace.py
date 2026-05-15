"""Create an isolated Codex dry-run workspace for the Minigrid pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = REPO_ROOT / "hlbench" / "scenarios"
RUNS_ROOT = REPO_ROOT / "runs"
PROMPT_TEMPLATE = REPO_ROOT / "prompts" / "codex_harness_step.md"


def create_workspace(run_id: str | None = None, scenario: str = "minigrid_doorkey") -> Path:
    if run_id is None:
        run_id = "codex-dryrun-" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / run_id
    workspace = run_dir / "workspace"
    if workspace.exists():
        raise FileExistsError(workspace)
    workspace.mkdir(parents=True)

    scenario_dir = SCENARIO_ROOT / scenario
    shutil.copy2(scenario_dir / "policy.py", workspace / "policy.py")
    (workspace / "policy_memory.json").write_text("{}\n")
    (workspace / "notes.md").write_text("# Notes\n\n")
    (workspace / "task_spec.md").write_text(_dryrun_task_spec())
    tools_dir = workspace / "tools"
    tools_dir.mkdir()
    (tools_dir / "run_rollout.py").write_text(_fixture_rollout_tool())

    prompt = (
        PROMPT_TEMPLATE.read_text()
        + "\n\n## Dry-Run Workspace\n\n"
        + "You are already inside a generated workspace containing `task_spec.md`, "
        + "`policy.py`, `policy_memory.json`, `notes.md`, and `tools/run_rollout.py`.\n"
        + "Use the fixture train rollout to validate one policy edit. Do not edit files outside this workspace.\n"
    )
    (run_dir / "prompt.md").write_text(prompt)
    (run_dir / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # Codex Dry-Run Workspace

            Run:

            ```text
            cd {workspace}
            codex exec < ../prompt.md
            ```

            The fixture rollout has no Gymnasium dependency. It checks the policy
            contract and produces train feedback artifacts under `rollout/`.
            """
        )
    )
    print(workspace)
    return workspace


def _dryrun_task_spec() -> str:
    return textwrap.dedent(
        """\
        # Minigrid DoorKey Dry-Run Task

        This workspace is a dependency-free dry run of the HL-Bench Codex loop.
        It simulates train feedback for `MiniGrid-DoorKey-8x8-v0` without running
        the real environment.

        ## Objective

        Improve `policy.py` from the trivial `S_0` policy into a valid first
        heuristic. Make one targeted edit.

        ## Policy Contract

        ```python
        class Policy:
            def reset(self, seed: int, task_config: dict) -> None:
                ...

            def act(self, observation: dict, info: dict) -> int:
                ...
        ```

        `act` must return an integer action in `[0, action_count)`.

        ## Simulated Public Observation

        The fixture uses an observation shaped like Minigrid's public observation:

        ```json
        {
          "image": [[[...]]],
          "direction": 0,
          "mission": "use the key to open the door and then get to the goal",
          "action_count": 7
        }
        ```

        ## Allowed Files

        ```text
        policy.py
        policy_memory.json
        notes.md
        ```

        ## Protected Files

        ```text
        task_spec.md
        tools/run_rollout.py
        ```

        ## Allowed Commands

        ```text
        python tools/run_rollout.py --split train
        python -m compileall policy.py
        ```

        Do not run validation or held-out evaluation in this dry run.
        """
    )


def _fixture_rollout_tool() -> str:
    return textwrap.dedent(
        """\
        import argparse
        import importlib.util
        import json
        from pathlib import Path


        def load_policy():
            spec = importlib.util.spec_from_file_location("policy", "policy.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Policy()


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--split", default="train")
            args = parser.parse_args()

            policy = load_policy()
            observation = {
                "image": [[[0, 0, 0], [1, 0, 0], [2, 5, 0]]],
                "direction": 0,
                "mission": "use the key to open the door and then get to the goal",
                "action_count": 7,
            }
            policy.reset(1, {"scenario_id": "minigrid_doorkey_8x8_v0"})
            actions = []
            for _ in range(12):
                action = policy.act(observation, {"action_count": 7})
                if not isinstance(action, int) or not 0 <= action < 7:
                    raise SystemExit(f"invalid action: {action!r}")
                actions.append(action)

            unique_actions = sorted(set(actions))
            if len(unique_actions) <= 1:
                failure_modes = [
                    {
                        "failure_id": "failure_000000",
                        "cluster": "repeats_one_action_until_timeout",
                        "count": 2,
                        "summary": "Policy repeats one action and would time out.",
                    }
                ]
                success_rate = 0.0
                mean_return = 0.0
            else:
                failure_modes = [
                    {
                        "failure_id": "failure_000000",
                        "cluster": "not_yet_validated_in_real_env",
                        "count": 1,
                        "summary": "Policy is syntactically valid and uses multiple actions; real rollout still required.",
                    }
                ]
                success_rate = 0.0
                mean_return = 0.0

            rollout = Path("rollout")
            rollout.mkdir(exist_ok=True)
            summary = {
                "split": args.split,
                "episodes": 2,
                "success_rate": success_rate,
                "mean_return": mean_return,
                "mean_steps": 200.0,
                "action_sequence_sample": actions,
                "failure_modes": failure_modes,
            }
            (rollout / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
            with (rollout / "failures.jsonl").open("w") as handle:
                for failure in failure_modes:
                    handle.write(json.dumps(failure, sort_keys=True) + "\\n")
            print(json.dumps(summary, indent=2))


        if __name__ == "__main__":
            main()
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenario", default="minigrid_doorkey")
    args = parser.parse_args()
    create_workspace(args.run_id, args.scenario)


if __name__ == "__main__":
    main()

