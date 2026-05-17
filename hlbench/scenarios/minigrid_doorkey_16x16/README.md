# MiniGrid DoorKey 16x16 Scenario

Hard MiniGrid scenario for sparse-reward exploration and key-door planning.

The policy-visible observation is the official MiniGrid public dictionary:
`image`, `direction`, `mission`, and `action_count`. The policy returns one
integer action in `[0, 6]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario minigrid_doorkey_16x16 --preset smoke --epochs 1
```
