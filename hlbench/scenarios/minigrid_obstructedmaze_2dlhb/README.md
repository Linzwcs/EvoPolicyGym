# MiniGrid ObstructedMaze 2Dlhb Scenario

Hard MiniGrid scenario for object manipulation, blocked doors, and partial-map
planning.

The policy-visible observation is the official MiniGrid public dictionary:
`image`, `direction`, `mission`, and `action_count`. The policy returns one
integer action in `[0, 6]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario minigrid_obstructedmaze_2dlhb --preset smoke --epochs 1
```
