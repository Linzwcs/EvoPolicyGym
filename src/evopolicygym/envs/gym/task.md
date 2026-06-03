# Gymnasium Task

This task is backed by an upstream Gymnasium environment. Implement
`system/policy.py` with a `Policy` class that receives the observation schema,
action schema, and run metadata in `__init__`.

## Observation

Observations are converted to JSON-safe Python values before they reach the
policy. Box observations are lists, Discrete observations are integers, and
Dict/Tuple spaces preserve their nested structure.

## Action

Return an action matching the declared action schema. Box actions should be
lists of numbers with the configured shape. Discrete actions should be integer
labels. Invalid numeric actions are clipped or repaired by the harness and are
marked with `action_invalid` in the transition info.

## Objective

Maximize total Gymnasium reward over each episode. Feedback reports train-case
returns for the submitted environment instances; validation and held-out cases
remain hidden.
