# Hopper Scenario

Hard MuJoCo scenario for one-legged locomotion.

The policy receives the official Hopper state vector and returns three
continuous joint torques in `[-1, 1]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario hopper --preset smoke --epochs 1
```
