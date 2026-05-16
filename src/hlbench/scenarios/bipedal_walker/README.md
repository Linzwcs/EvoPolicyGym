# Bipedal Walker Scenario

Medium Box2D scenario for continuous locomotion control.

The policy-visible observation is the official Gymnasium telemetry vector with
24 values: hull motion, joint positions and speeds, leg contact flags, and 10
lidar rangefinder measurements.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario bipedal_walker --preset smoke --epochs 1
```
