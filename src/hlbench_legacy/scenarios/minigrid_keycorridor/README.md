# Minigrid KeyCorridor Scenario

This is a pilot-medium MiniGrid scenario for the HL-Bench pilot.

The current environment is `MiniGrid-KeyCorridorS3R2-v0`. The task is object
retrieval in a compact corridor-and-room layout: the mission names a colored
ball, and the policy may need to explore, find a key, open doors, reach the
target object, and pick it up. This keeps the key-door-object structure from the
harder KeyCorridor family while making reward less sparse than the previous
`MiniGrid-KeyCorridorS6R3-v0` setting.

Start with:

```text
python -m hlbench.rollout.run_policy --scenario minigrid_keycorridor --split train --episodes 1 --run-id keycorridor_smoke
```

Run the automated one-step Codex harness:

```text
python -m hlbench.harness.run_codex_step --scenario minigrid_keycorridor --run-id keycorridor_codex_step --train-episodes 2 --validation-episodes 20 --heldout-episodes 50
```

Run a bounded multi-epoch loop:

```text
python -m hlbench.harness.run_codex_loop --scenario minigrid_keycorridor --run-id keycorridor_loop --epochs 5 --train-episodes 2 --validation-episodes 20 --heldout-episodes 50
```

For harder stress tests, add a separate S6R3 scenario instead of replacing this
pilot setting.
