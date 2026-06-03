# Toy

## Objective

Solve a one-step additive control task. Maximize the return by choosing an
integer action that increases the current state.

## Policy Interface

Implement `system/policy.py` with `class Policy`. The constructor receives
`obs_space`, `action_space`, and `env_meta` as dictionaries. `reset(episode_index)`
is called before each episode, and `act(obs)` must return one action compatible
with `action_space`.

## Observation

`obs` is an integer state. By default it starts from the visible train case ID;
external case data may override the start value.

## Action

Return an integer. The toy environment converts the action with `int(action)`
and adds it to the state.

## Reward

The episode lasts one step. The new state after applying the action is returned
as the reward, and the episode terminates immediately.

## Notes

This environment is intentionally simple. Use it to verify the submit protocol,
feedback artifacts, and policy lifecycle before testing richer environments.
