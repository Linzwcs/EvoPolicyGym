# Inverted Pendulum Scenario

Medium MuJoCo scenario for continuous stabilization control.

The policy receives the official InvertedPendulum state vector and returns one
continuous cart force in `[-3, 3]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario inverted_pendulum --preset smoke --epochs 1
```
