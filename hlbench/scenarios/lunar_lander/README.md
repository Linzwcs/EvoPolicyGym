# Lunar Lander Scenario

Medium Box2D scenario for a discrete-action landing-control task.

The policy-visible observation is the official Gymnasium telemetry vector:
`[x_position, y_position, x_velocity, y_velocity, angle, angular_velocity,
left_leg_contact, right_leg_contact]`.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario lunar_lander --preset smoke --epochs 1
```
