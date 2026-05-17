# HalfCheetah Scenario

Hard MuJoCo scenario for six-actuator locomotion.

The policy receives the official HalfCheetah state vector and returns six
continuous joint torques in `[-1, 1]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario half_cheetah --preset smoke --epochs 1
```
