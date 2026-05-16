"""Create a real Minigrid Codex workspace backed by the repository rollout runner."""

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


def create_workspace(
    run_id: str | None = None,
    scenario: str = "minigrid_doorkey",
    train_episodes: int | None = None,
) -> Path:
    if run_id is None:
        run_id = "codex-real-" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / run_id
    workspace = run_dir / "workspace"
    if workspace.exists():
        raise FileExistsError(workspace)
    workspace.mkdir(parents=True)

    scenario_dir = SCENARIO_ROOT / scenario
    shutil.copy2(scenario_dir / "policy.py", workspace / "policy.py")
    (workspace / "policy_memory.json").write_text("{}\n")
    (workspace / "notes.md").write_text("# Notes\n\n")
    (workspace / "task_spec.md").write_text(_workspace_task_spec(scenario, train_episodes))
    tools_dir = workspace / "tools"
    tools_dir.mkdir()
    (tools_dir / "run_rollout.py").write_text(_rollout_wrapper(scenario, train_episodes))

    prompt = (
        PROMPT_TEMPLATE.read_text()
        + "\n\n## Real Minigrid Workspace\n\n"
        + "You are inside an isolated workspace. Edit only `policy.py`, "
        + "`policy_memory.json`, and `notes.md`. Use `python tools/run_rollout.py --split train` "
        + "for the configured train feedback batch. Do not run validation or held-out rollouts.\n"
    )
    (run_dir / "prompt.md").write_text(prompt)
    print(workspace)
    return workspace


def _workspace_task_spec(scenario: str, train_episodes: int | None) -> str:
    scenario_dir = SCENARIO_ROOT / scenario
    scenario_config = json.loads((scenario_dir / "scenario.json").read_text())
    env_id = str(scenario_config["env_id"])
    scenario_id = str(scenario_config["scenario_id"])
    configured_train = "all train seeds" if train_episodes is None else str(train_episodes)
    task_overview = textwrap.indent(_task_overview(scenario_config), "        ")
    return textwrap.dedent(
        f"""\
        # Minigrid Real Rollout Task

        This workspace evaluates candidate `policy.py` against the configured
        MiniGrid scenario through the repository rollout runner.

        Scenario: `{scenario}`

        Scenario id: `{scenario_id}`

        Environment id: `{env_id}`

        Configured train episodes: `{configured_train}`

        ## Task Overview

{task_overview}

        ## Objective

        Improve the current `policy.py` heuristic. Make one targeted edit based
        on train feedback.

        ## Policy Contract

        ```python
        class Policy:
            def reset(self, seed: int, task_config: dict) -> None:
                ...

            def act(self, observation: dict, info: dict) -> int:
                ...
        ```

        `act` must return an integer action in `[0, action_count)`.

        ## Public Observation

        The policy receives only public Minigrid observation fields:

        - `image`
        - `direction`
        - `mission`
        - `action_count`

        Do not use hidden grid internals, object coordinates, evaluator files,
        held-out seeds, or private rollout artifacts.

        ## Train Feedback Artifacts

        Train rollout artifacts are written under:

        ```text
        rollout/
          summary.json
          failures.jsonl
          trials.jsonl
          replays/
            trial_*.jsonl
        ```

        Replay files contain raw public observation/action/reward transitions:

        ```text
        obs_before -> action -> reward -> obs_after -> done
        ```

        The latest run is mirrored in `rollout/`. Full public train rollout
        history from commands you run is kept under `rollouts/`.

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

        The default train rollout uses the configured train episode count. You
        may pass `--episodes N` for a smaller public train probe, but final
        verification should use the default command.

        Do not run validation or held-out evaluation in this learner workspace.
        """
    )


def _task_overview(scenario_config: dict) -> str:
    env_id = str(scenario_config["env_id"])
    title = str(scenario_config.get("task_title", "MiniGrid scenario"))
    intro = str(scenario_config.get("task_intro", _default_task_intro(env_id)))
    success = str(
        scenario_config.get(
            "success_condition",
            "An episode succeeds when the policy receives positive environment reward before the max-step limit.",
        )
    )
    difficulty = str(scenario_config.get("difficulty_notes", ""))

    paragraphs = [
        _wrap_paragraph(f"Task type: {title}."),
        _wrap_paragraph(intro),
        _wrap_paragraph(f"Success condition: {success}"),
    ]
    if difficulty:
        paragraphs.append(_wrap_paragraph(f"Difficulty note: {difficulty}"))
    return "\n\n".join(paragraphs)


def _wrap_paragraph(text: str) -> str:
    return textwrap.fill(text, width=88)


def _default_task_intro(env_id: str) -> str:
    if "DoorKey" in env_id:
        return (
            "The agent must navigate a MiniGrid room, find a key, open the locked "
            "door, and reach the goal using only the public egocentric observation."
        )
    if "KeyCorridor" in env_id:
        return (
            "The agent must explore a corridor-and-room MiniGrid layout, use keys "
            "and doors when needed, and complete the object-retrieval mission named "
            "by the public mission string."
        )
    return (
        "The agent must solve the configured MiniGrid task using only public "
        "observations and environment rewards."
    )


def _rollout_wrapper(scenario: str, train_episodes: int | None) -> str:
    return textwrap.dedent(
        f"""\
        import argparse
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        REPO_ROOT = Path({str(REPO_ROOT)!r})

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--split", default="train")
            parser.add_argument("--episodes", type=int, default={train_episodes!r})
            args = parser.parse_args()
            if args.split != "train":
                raise SystemExit("Only train rollout is allowed from learner workspace.")

            output_dir = Path(".run_tmp")
            if output_dir.exists():
                shutil.rmtree(output_dir)

            cmd = [
                sys.executable,
                "-m",
                "hlbench.rollout.run_policy",
                "--scenario",
                "{scenario}",
                "--split",
                "train",
                "--policy",
                str(Path("policy.py").resolve()),
                "--output-dir",
                str(output_dir.resolve()),
            ]
            if args.episodes is not None:
                cmd.extend(["--episodes", str(args.episodes)])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(cmd, cwd=Path.cwd(), env=env, check=True)

            history_dir = Path("rollouts")
            history_dir.mkdir(exist_ok=True)
            run_index = sum(
                1
                for child in history_dir.iterdir()
                if child.is_dir() and child.name.startswith("train_")
            )
            history_rollout = history_dir / f"train_{{run_index:04d}}"
            if history_rollout.exists():
                shutil.rmtree(history_rollout)
            shutil.copytree(output_dir / "rollout", history_rollout)

            local_rollout = Path("rollout")
            if local_rollout.exists():
                shutil.rmtree(local_rollout)
            shutil.copytree(history_rollout, local_rollout)
            print((local_rollout / "summary.json").read_text())
            print(f"Saved public train rollout to {{history_rollout}}")

        if __name__ == "__main__":
            main()
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--train-episodes", type=int, default=None)
    args = parser.parse_args()
    create_workspace(args.run_id, args.scenario, args.train_episodes)


if __name__ == "__main__":
    main()
