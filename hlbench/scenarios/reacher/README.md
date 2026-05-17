# Reacher Scenario

Medium MuJoCo scenario for short-horizon continuous arm control.

The policy receives the official Reacher state vector and returns two continuous
joint torques in `[-1, 1]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario reacher --preset smoke --epochs 1
```
