# EvoPolicyGym Agent Rules

This file is staged by EvoPolicyGym at the start of every run. Your harness
process starts in the workspace directory that contains this file. Treat all
file paths below as relative to that workspace.

## Objective

Write and iteratively improve `system/policy.py`. The module must export
`class Policy` with compatible `__init__`, `reset`, and `act` methods. You may
add helper modules, tests, cached data, and persistent state under `system/`.

## Code Quality

Optimize policy behavior first, but keep the submitted project maintainable.
When the policy grows, split clear helper modules under `system/`, remove dead
experiments, keep names specific, and avoid brittle assumptions about hidden
cases, paths, timing, or implementation details. Prefer simple, testable code
over large one-off scripts.

## Paths

- `AGENTS.md`: this rules file. Read it first; do not modify or replace it.
- `system/`: writable policy project. Put all submitted code and state here.
- `feedback/`: read-only feedback artifacts from previous submits.
- `$EVOPOLICYGYM_INFO_URL`, `$EVOPOLICYGYM_TASK_URL`, `$EVOPOLICYGYM_SUBMIT_URL`:
  concrete API URLs for this run.

Do not write to feedback artifacts, logs, checkpoints, hidden data, or files
outside `system/`.

## Submit Format

Submit evaluates the current snapshot of `system/` against visible train case
IDs. The request body is JSON with one field:

```json
{"env_instances": [0, 1, 2, 3]}
```

`env_instances` may also be a comma/range string:

```json
{"env_instances": "0-3,7"}
```

Each ID must be an integer in `[0, n_env_instances)`, where
`n_env_instances` comes from `GET /info`. The server expands string specs in
order and returns the expanded list in `summary.json`.

Example with Python:

```python
import json
import os
import urllib.request

body = json.dumps({"env_instances": [0, 1, 2, 3]}).encode("utf-8")
request = urllib.request.Request(
    os.environ["EVOPOLICYGYM_SUBMIT_URL"],
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))
print(result["status"], result["summary"]["mean_return"])
```

## Policy Interface

`system/policy.py` must define:

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict): ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs): ...
```

`obs_space` is the policy input schema: every `act(obs)` call receives an
observation matching it. `action_space` is the policy output schema: every
`act(obs)` return value must match it. Both are compact schema dictionaries.
Common schema types:

| Type | Policy value |
|---|---|
| `Discrete` | integer in `[start, start + n - 1]`; `start` defaults to `0` |
| `Box` | numeric array/list with declared `shape`, `dtype`, `low`, and `high` |
| `MultiDiscrete` | integer array/list with per-dimension bounds from `nvec` |
| `MultiBinary` | binary array/list of `0/1` or booleans |
| `Tuple` | ordered tuple/list matching `spaces` |
| `Dict` | dict matching named `spaces` |
| `Text` | string |
| `Image` | image/frame tensor; feedback usually stores examples externally |

Schema fields such as `fields`, `labels`, `layout`, `channels`, and
`semantics` are hints. The environment-specific meaning of observations,
actions, and rewards is described by `GET /task`.

## Loop

1. Read `GET /task` for the environment-specific observation, action, reward,
   and `Policy` I/O contract.
2. Read `GET /info` for remaining budget, submit count, train case count,
   submit bounds, and resource limits.
3. Edit `system/policy.py` and submit a small, deliberate batch.
4. Read `feedback/submit_NNN/summary.json`. If needed, inspect
   `feedback/submit_NNN/episodes/ep_XXX/trajectory.jsonl`, `stdout.txt`,
   `stderr.txt`, and `error.txt`.
5. Improve the policy and submit again until `/info` reports
   `state.is_finalized = true`.

Do not call `/finalize`; finalization is owned by the server after the episode
budget is exhausted.

## Feedback Artifacts

Feedback paths are relative to the workspace root:

```text
feedback/submit_NNN/
├── summary.json
└── episodes/ep_XXX/
    ├── trajectory.jsonl
    ├── observations.npy   # optional for large/image observations
    ├── observations.npz   # optional for large/image observations
    ├── stdout.txt
    ├── stderr.txt
    └── error.txt          # only when an episode failed
```

`trajectory.jsonl` has one JSON object per step with `t`, `obs`, `action`,
`reward`, `terminated`, `truncated`, and `info`. Small observations are inline.
For external observation storage, `obs` may be `null`; load the observation for
step `t` from `observations.npy` or `observations.npz` at index `t`.

Some nested observations may use an explicit reference object:

```json
{
  "type": "External",
  "path": "feedback/submit_000/episodes/ep_003/observations.npz",
  "key": "image",
  "index": 12,
  "shape": [84, 84, 4],
  "dtype": "uint8"
}
```

`path` is workspace-relative. Use `key` for `.npz` arrays and `index` for the
time step or row inside the stored array.

## Budget

The budget is an episode budget. Accepted submits spend
`len(env_instances)`. Phase-1 rejects such as invalid case IDs or invalid batch
size do not spend budget. LLM thinking time and harness wall time are outside
the benchmark budget; server-side rollout execution may still have safety
limits exposed by `/info.resource_limits`.

## Policy Runtime

The server imports `system/policy.py` with `system/` on the Python path and runs
episodes in a sandboxed runtime. A new `Policy` instance is created for each
submit. Persist cross-submit state by writing files under `system/`; instance
attributes only live within one submit.

## Prohibited Behavior

- Do not read hidden validation or held-out data.
- Do not use network access, subprocesses, or other mechanisms from policy code
  to bypass the sandbox.
- Do not infer hidden seeds or cases from filesystem layout, timing, process
  state, or implementation details outside the documented API.
