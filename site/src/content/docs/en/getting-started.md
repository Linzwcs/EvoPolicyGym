---
locale: en
page: getting-started
section: start
title: "Getting started"
navTitle: "Getting started"
description: "Install EvoPolicyGym 0.3 and complete the first Evaluation or Program-evolution Run."
lead: "Install the Kernel, evaluate an example Program, and start a bounded coding-agent Run."
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## Requirements

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) `0.11.16`
- A local checkout of the repository
- Trusted Policy and Agent code

> `ProcessExecution` launches Policy and Agent subprocesses with the authority
> of your operating-system user.

## Install the Kernel

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync
uv run evopolicygym --version
```

Expected version output:

```text
evopolicygym 0.3.0
```

The Kernel supplies the Evaluation and Program-evolution lifecycle. Benchmark
distributions supply environments, Policy contracts, and scoring.

## Choose a Benchmark

Choose a distribution from the
[Environment collection](../../environments/). The commands below use the
small CartPole distribution:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
```

## Complete one Evaluation

Evaluate the example distribution's packaged Program over five deterministic
validation Episodes:

```console
uv run --project environments/gymnasium/classic_control/cartpole python - <<'PY'
from cartpole import CartPoleBenchmark, baseline_program
from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.execution import ProcessExecution

result = evaluate(
    baseline_program(),
    CartPoleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(split="validation", episodes=5, seed=42),
)
print(result.feedback.score)
print(result.feedback.content)
PY
```

The example prints the scalar score and Benchmark-defined public Feedback.
`EvaluationResult` also retains the Benchmark identity, immutable Program
digest, and sanitized Episode summaries.

`ProcessExecution.unsafe()` explicitly acknowledges execution under the
current local user.

## Run a Coding Agent (optional)

After authenticating the Codex CLI, let the Agent revise the example Program
within a small development budget:

```console
mkdir -p runs
uv run --project environments/gymnasium/classic_control/cartpole \
  python scripts/run_cartpole_codex.py \
  --model gpt-5.6-luna \
  --reasoning-effort high \
  --record-to runs/quickstart-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --episode-pool-size 60 \
  --max-episodes-per-submission 10 \
  --allow-unsafe-process
```

The Host constructs 60 fixed Run-local training Episode identities before the
Agent starts. The Agent chooses a non-empty index selector for each Submission
while spending at most 30 Episode units in total and 10 per Submission. Its
Program workspace is `runs/quickstart-001/workspace/program/`, committed
Feedback appears under `workspace/feedback/`, and Host records remain in the
Run directory.

Inside the active Agent Session, a Submission uses singleton indices and
half-open ranges:

```console
evopolicygym-session submit program --episodes "0:2,4:8"
```

The selector above evaluates indices `0, 1, 4, 5, 6, 7`. Reusing an index in a
later Submission provides a matched Episode specification and Policy seed, but
still creates a fresh Environment and Policy runtime and spends budget again.
The actual seeds remain hidden.

## What the Run does

1. The initial Policy directory became an immutable, content-addressed
   `Program`.
2. Before Agent execution, the Host built one deterministic indexed training
   pool from the Run seed.
3. The Coding Agent received a fixed workspace, Benchmark specification,
   selectable pool bounds, and finite submission and Episode authority.
4. Every Submission selected explicit pool indices; every selected index
   created a fresh Environment and fresh Policy process.
5. A completed Submission atomically published its Program, selected indices,
   Feedback, Episode summaries, and optional artifacts.
6. The Agent selected one fully published Submission as the final Program.

## Next steps

- [Read the core concepts →](../concepts/)
- [Read the Policy ABI →](../policy/)
- [Understand Evaluation and Runs →](../evaluation/)
- [Choose and configure an Environment →](../../environments/)
- [Use the `$evopolicygym` Skill with an AI coding assistant →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/skills/evopolicygym)
