# CartPole Balance

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Write a `Policy` class in `system/policy.py` with an
`act(observation, context)` method. This is a public telemetry task: the policy
and human task description both use
`[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`. Return `0`
for left force or `1` for right force.
