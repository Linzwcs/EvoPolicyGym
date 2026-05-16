# Minigrid DoorKey Scenario

This is the first single-environment HL-Bench pilot.

Start with:

```text
python -m hlbench.rollout.run_policy --scenario minigrid_doorkey --split train --episodes 2 --run-id minigrid_smoke
```

Then build a Codex prompt:

```text
python -m hlbench.harness.codex_prompt --scenario minigrid_doorkey --run-dir runs/minigrid_smoke --output runs/minigrid_smoke/codex_prompt.md
```

Automated one-step Codex harness run:

```text
python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id codex_minigrid_auto
```

The orchestrator creates the workspace, runs `codex exec`, evaluates train,
validation, and heldout splits outside the learner workspace, and writes:

```text
runs/codex_minigrid_auto/transition.json
runs/codex_minigrid_auto/policy.patch
runs/codex_minigrid_auto/codex/stdout.txt
runs/codex_minigrid_auto/evaluator/
```

To test only evaluator and logging without calling Codex:

```text
python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id codex_minigrid_skip --skip-codex
```

Automated multi-epoch loop with one persistent learner workspace:

```text
python -m hlbench.harness.run_codex_loop --scenario minigrid_doorkey --run-id codex_minigrid_loop --epochs 5
```

The loop keeps `runs/codex_minigrid_loop/workspace/` as the only active learner
workspace. The benchmark checkpoints every epoch to
`checkpoints/H_*/workspace/` and writes per-epoch artifacts under
`steps/epoch_*/`. Rejected or non-improving candidates are not rolled back; the
next epoch continues from the current workspace.

If Gymnasium/Minigrid dependencies are not installed yet, create a dependency-free
Codex dry-run workspace:

```text
python -m hlbench.harness.create_codex_workspace --run-id codex_minigrid_dryrun
cd runs/codex_minigrid_dryrun/workspace
codex exec < ../prompt.md
```
