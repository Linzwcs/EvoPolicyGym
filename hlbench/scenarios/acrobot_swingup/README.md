# Acrobot Swing-Up Scenario

Medium Gymnasium scenario for a longer-horizon discrete-control task. It uses
`Acrobot-v1` with shared train / validation / heldout seed pools.

Smoke command:

```bash
PYTHONPATH=src python -m hlbench run --scenario acrobot_swingup --preset smoke --epochs 1
```
