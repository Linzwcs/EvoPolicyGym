---
locale: en
page: getting-started
section: start
title: "Getting started"
navTitle: "Getting started"
description: "Install EvoPolicyGym 0.3 and evaluate the CartPole baseline."
lead: "Install one Benchmark and run one deterministic Evaluation."
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## Installation

You need Python `3.12` and [`uv`](https://docs.astral.sh/uv/) `0.11.16`.

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym --version
```

Expected version output:

```text
evopolicygym 0.3.0
```

This installs the EvoPolicyGym Kernel and the independent CartPole Benchmark.

## Evaluate the baseline

Run the packaged baseline over five validation Episodes:

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

The result contains a scalar score, public Feedback, the Program digest, and
sanitized Episode summaries.

:::warning Local process execution

`ProcessExecution.unsafe()` runs Policy code with your operating-system user
permissions. It is not a sandbox. Use it only with trusted code.

:::

## Next steps

- [Create a Program](./programs.md)
- [Write a Policy](./policy.md)
- [Configure an Evaluation](./evaluation.md)
- [Run a Coding Agent](./runs.md)
- [Choose another Environment](/environments/)
