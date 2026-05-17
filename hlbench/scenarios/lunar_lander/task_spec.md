# Lunar Lander

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Land the spacecraft safely on the pad near the origin. The policy receives the
official LunarLander telemetry vector:
`[x_position, y_position, x_velocity, y_velocity, angle, angular_velocity,
left_leg_contact, right_leg_contact]`.

Return one of four discrete actions: `0` for no engine, `1` for the left
orientation engine, `2` for the main engine, or `3` for the right orientation
engine. Engine use costs reward, but uncontrolled descent or tilted contact can
lead to a crash.
