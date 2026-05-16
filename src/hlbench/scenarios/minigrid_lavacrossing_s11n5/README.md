# MiniGrid LavaCrossing S11N5 Scenario

Hard MiniGrid scenario for safe navigation through a larger lava crossing map.

The policy-visible observation is the official MiniGrid public dictionary:
`image`, `direction`, `mission`, and `action_count`. The policy returns one
integer action in `[0, 6]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario minigrid_lavacrossing_s11n5 --preset smoke --epochs 1
```
