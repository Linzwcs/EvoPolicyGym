# Quickstart

This walks through running the benchmark end-to-end on Pendulum-v1
with the reference PD policy, and then drafts a minimal *agent loop*
script you can adapt for a real model.

## Prerequisites

- Python 3.12+
- A POSIX shell (we use `bash`/`zsh`)

## 1. Set up

```bash
# Clone (skip if you already have the repo)
git clone <repo-url> hlbench-pro && cd hlbench-pro

# Recommended: uv. (pip works too — install pytest + hlbench with `pip install -e ".[dev]"`)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python gymnasium numpy pytest
```

Verify:

```bash
.venv/bin/python -m pytest -q
# expect: 77 passed
```

## 2. Run a benchmark with the CLI

There are two terminals: one runs the server, the other drives it.

**Terminal A — set up + serve:**

```bash
# Creates ./runs/reference-pd/pendulum/<auto-id>/ with workspace/, checkpoints/, logs/
.venv/bin/hlbench init --env pendulum --model reference-pd --exp-id demo

# Drop the reference policy into the workspace under the run dir.
RUN_DIR=./runs/reference-pd/pendulum/demo
cp agents/pd_pendulum/policy.py $RUN_DIR/workspace/system/policy.py

# Start the HTTP server pointing at that run dir.
.venv/bin/hlbench serve --run-dir $RUN_DIR --env pendulum
```

**Terminal B — submit and finalize:**

```bash
.venv/bin/hlbench info                              # see budget = 256
.venv/bin/hlbench submit --env-instances 0-7        # 8 episodes
.venv/bin/hlbench submit --env-instances 8,16,32    # 3 specific instances
.venv/bin/hlbench finalize                          # held-out + run.json
```

You'll see something like:

```
finalize: completed
  final_score:        98.30
  held_out_mean:      -168.00
  held_out_std:       107.78
  final_submit:       #1
  run.json:           ./runs/reference-pd/pendulum/demo/run.json
```

Inspect:

```bash
ls $RUN_DIR/workspace/feedback/                     # one dir per submit
cat $RUN_DIR/workspace/feedback/submit_000/summary.json | jq '.mean_return'
cat $RUN_DIR/run.json | jq '.outcome.final_score'   # 98.3
ls $RUN_DIR/checkpoints/                            # per-submit code snapshots
```

## 3. The same flow via the lib API

`hlbench.core.Server` is the only entry point you need for tests,
notebooks, or research orchestration. Agents must use HTTP per
[CLAUDE.md invariant 8](../CLAUDE.md), but the lib is fair game for
internal tooling.

```python
import shutil
from pathlib import Path
from hlbench.core.server import Server

srv = Server(
    env_id="pendulum",
    runs_root=Path("./runs"),
    model="reference-pd",
    exp_id="demo",
)
shutil.copy(
    "agents/pd_pendulum/policy.py",
    srv.workspace_dir / "system" / "policy.py",
)

for i in range(5):
    result = srv.submit(list(range(i * 4, i * 4 + 4)))
    print(f"submit {i}: {result.summary['mean_return']:.1f}")

final = srv.finalize()
print(f"final_score = {final.final_score}")
print(f"run.json    = {final.run_json_path}")
```

## 4. Sketch an agent loop

A real agent is just a script that:
1. Reads `GET /info` to learn the env constraints.
2. Drops a `Policy` into `system/policy.py`.
3. Submits, reads `feedback/submit_NNN/`, edits `policy.py`, repeats.
4. Calls `POST /finalize` (or lets the budget exhaust and finalize externally).

Stub:

```python
import json
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8765"
WORKSPACE = Path("./run")

def get(path):
    with urllib.request.urlopen(f"{URL}{path}") as r:
        return json.loads(r.read())

def post(path, body=None):
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

# 1. Inspect run config
info = get("/info")
n_inst = info["env_meta"]["n_env_instances"]
budget = info["state"]["remaining_budget"]

# 2. Drop a starter policy.
(WORKSPACE / "system" / "policy.py").write_text(my_first_policy_src)

# 3. Submit / read / iterate.
remaining = budget
ei = 0
while remaining > 0:
    n_this = min(8, remaining)
    ids = list(range(ei % n_inst, (ei + n_this) % n_inst))
    result = post("/submit", {"env_instances": ids})
    summary = result["summary"]
    print(f"submit {result['submit_id']}: mean={summary['mean_return']:.1f}")
    if summary["mean_return"] is not None and summary["mean_return"] < -300:
        # Read trajectories, decide what to change.
        with open(WORKSPACE / "feedback" / f"submit_{result['submit_id']:03d}" /
                  "episodes" / f"ep_{result['summary']['first_global_episode']:03d}" /
                  "trajectory.jsonl") as f:
            traj = [json.loads(line) for line in f]
        # ... ask the model to revise system/policy.py based on `traj` ...
    remaining = summary["remaining_budget"]
    ei += n_this

# 4. Finalize.
final = post("/finalize")
print(f"final_score = {final['final_score']}")
```

## Troubleshooting

**`hlbench: command not found`** — re-run `pip install -e .` after
editing `pyproject.toml`. The `[project.scripts]` entry is what
creates the executable.

**`connection error: ... is hlbench serve running on http://127.0.0.1:8765?`**
— start `hlbench serve` in another terminal first. The CLI is a
pure HTTP client; it doesn't carry server state.

**`HTTP 409 ... finalize`** — you've already finalized this run. Each
`Server` is single-use. To run the agent again, delete `run/` and
start over (or use a different workspace directory).

**Tests fail with `ModuleNotFoundError: gymnasium`** — install with
`uv pip install gymnasium numpy pytest`. The base install
(`hlbench` + `numpy` + `gymnasium`) is the only requirement; FastAPI
is not used.

## Next steps

- Read [`SPEC.md §2`](../SPEC.md) for the `Policy` interface and how
  observation and action shapes are conveyed.
- Read [`docs/submit-protocol.md`](./submit-protocol.md) for the
  7-phase submit lifecycle and 11 verdicts you may encounter.
- Read [`docs/findings.md`](./findings.md) for what calibration
  revealed about the metric definitions.
