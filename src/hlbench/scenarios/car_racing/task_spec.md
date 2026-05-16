# Car Racing

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Drive around the generated track using the official top-down 96x96 RGB image
observation. The policy receives a dictionary containing `image_path`,
`format`, `shape`, and `dtype` for the current frame.

Return three continuous actions: `[steering, gas, brake]`. Steering is in
`[-1, 1]`, while gas and brake are in `[0, 1]`.
