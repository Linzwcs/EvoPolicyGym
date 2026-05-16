# Minigrid DoorKey HL Scenario

## Goal

Write and improve `policy.py` so it solves `MiniGrid-DoorKey-16x16-v0`.

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
python -m hlbench.rollout.run_policy --scenario minigrid_doorkey --split train --run-id <run_id>
```

Run a quick train smoke:

```text
python -m hlbench.rollout.run_policy --scenario minigrid_doorkey --split train --episodes 2 --run-id <run_id>
```

Build a Codex prompt from the latest train run:

```text
python -m hlbench.harness.codex_prompt --scenario minigrid_doorkey --run-dir runs/<run_id> --output runs/<run_id>/codex_prompt.md
```

Compile policy:

```text
python -m compileall hlbench/scenarios/minigrid_doorkey/policy.py
```

Do not run held-out evaluation from inside a learner session.
