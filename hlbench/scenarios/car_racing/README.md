# Car Racing Scenario

Medium Box2D scenario for image-observation racing control.

The policy-visible observation is an image artifact dictionary pointing to the
current official 96x96 RGB frame. Train rollouts persist these frame artifacts;
validation and heldout evaluations do not keep frame replays.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario car_racing --preset smoke --epochs 1
```
