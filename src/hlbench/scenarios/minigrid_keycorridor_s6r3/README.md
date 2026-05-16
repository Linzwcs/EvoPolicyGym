# MiniGrid KeyCorridor S6R3 Scenario

Hard MiniGrid scenario for mission-conditioned exploration across corridors and
rooms.

The policy-visible observation is the official MiniGrid public dictionary:
`image`, `direction`, `mission`, and `action_count`. The policy returns one
integer action in `[0, 6]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario minigrid_keycorridor_s6r3 --preset smoke --epochs 1
```
