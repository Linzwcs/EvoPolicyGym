# Minigrid KeyCorridor HL Scenario

## Goal

Write and improve `policy.py` so it solves `MiniGrid-KeyCorridorS3R2-v0`.

This is a pilot-medium KeyCorridor task. The agent is in a compact
corridor-and-room grid. Each episode gives a mission such as picking up a
colored ball. The target object may be behind doors, so a good heuristic usually
needs to explore, recognize keys and doors from the public image observation,
pick up a key when needed, toggle doors open, navigate to the target object, and
pick it up.

MiniGrid agents can carry only one object at a time. If the policy is carrying a
key and reaches the target ball, it may need to drop the key before picking up
the ball.

An episode succeeds when the policy completes the mission and receives positive
environment reward before the max-step limit.

The policy must use only public observations from the Gymnasium environment:

- `image`
- `direction`
- `mission`
- `action_count`
- public `info` fields

Do not use hidden grid internals, object coordinates, evaluator files, held-out
seeds, or private rollout artifacts.

## Train Feedback Artifacts

Train rollout artifacts are available under:

```text
rollout/
  summary.json
  failures.jsonl
  trials.jsonl
  replays/
    trial_*.jsonl
```

Replay files contain raw public environment transitions:

```text
obs_before -> action -> reward -> obs_after -> done
```

## Policy Contract

```python
class Policy:
    def reset(self, seed: int, task_config: dict) -> None:
        ...

    def act(self, observation: dict, info: dict) -> int:
        ...
```

`act` must return an integer action in `[0, action_count)`.

## Allowed Files

```text
policy.py
policy_memory.json
notes.md
```

## Protected Files

```text
scenario.json
task_spec.md
```

## Allowed Commands

Run train feedback:

```text
python -m hlbench.rollout.run_policy --scenario minigrid_keycorridor --split train --run-id <run_id>
```

Run a quick train smoke:

```text
python -m hlbench.rollout.run_policy --scenario minigrid_keycorridor --split train --episodes 2 --run-id <run_id>
```

Compile policy:

```text
python -m compileall hlbench/scenarios/minigrid_keycorridor/policy.py
```

Do not run held-out evaluation from inside a learner session.
