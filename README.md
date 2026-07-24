<p align="center">
  <img src="https://raw.githubusercontent.com/Linzwcs/EvoPolicyGym/main/site/public/favicon.svg" width="112" alt="EvoPolicyGym logo">
</p>

<h1 align="center">EvoPolicyGym</h1>

<p align="center">
  A benchmark kernel for evaluating how coding agents improve executable
  policies through bounded interaction and feedback.
</p>

<p align="center">
  <a href="https://github.com/Linzwcs/EvoPolicyGym/actions/workflows/ci.yml"><img src="https://github.com/Linzwcs/EvoPolicyGym/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://arxiv.org/abs/2607.02440"><img src="https://img.shields.io/badge/arXiv-2607.02440-b31b1b.svg" alt="arXiv:2607.02440"></a>
  <a href="https://github.com/Linzwcs/EvoPolicyGym/tree/v0.1.0"><img src="https://img.shields.io/badge/Paper_code-v0.1.0-6f42c1.svg" alt="Paper code: v0.1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

EvoPolicyGym studies **Autonomous Policy Evolution**: a coding agent edits a
Python Policy Program, submits immutable versions for evaluation, reads
Benchmark-defined feedback, and iterates under a fixed submission and Episode
budget. The coding agent improves the Program between evaluations; the Policy
does not learn inside an Episode.

Read the [documentation](https://linzwcs.github.io/EvoPolicyGym/), the
[architecture](ARCHITECTURE.md), or the
[paper](https://arxiv.org/abs/2607.02440). The paper's implementation,
experiment configuration, and Core16 results are preserved at
[`v0.1.0`](https://github.com/Linzwcs/EvoPolicyGym/tree/v0.1.0); they are
historical research artifacts, not outputs of the current 0.3 Kernel.

## Environments

Environment distributions are independent packages that depend only on the
public EvoPolicyGym SDK.

| Environment | Package | Description |
| --- | --- | --- |
| [CartPole](environments/cartpole/) | `evopolicygym-benchmark-cartpole` | Minimal Gymnasium reference Benchmark with public trace feedback |
| [Balatro](environments/balatro/) | `evopolicygym-benchmark-balatro` | Unofficial long-horizon Red Deck, White Stake Benchmark powered by a pinned Jackdaw engine |
| [Core16](https://linzwcs.github.io/EvoPolicyGym/results/) | [`v0.1.0` paper archive](https://github.com/Linzwcs/EvoPolicyGym/tree/v0.1.0) | The 16 control, navigation, driving, and robotics tasks used in the paper |

Balatro includes no official game assets and is not affiliated with LocalThunk
or Playstack.

## Installation

EvoPolicyGym requires Python 3.12 and uses
[uv](https://docs.astral.sh/uv/):

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync --extra dev
```

Install an Environment in its own project:

```console
cd environments/cartpole
uv sync --extra dev
```

## API

A Policy Program is a directory containing `policy.py` with a fixed
`make_policy` entry point:

```python
from evopolicygym.policy import PolicyContext, PolicyValue


class Policy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0


def make_policy(context: PolicyContext) -> Policy:
    return Policy()
```

Capture the directory as an immutable Program and evaluate it:

```python
from cartpole import CartPoleBenchmark

from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution

result = evaluate(
    Program.from_directory("policy"),
    CartPoleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(
        split="validation",
        episodes=10,
        seed=42,
    ),
)

print(result.feedback.score)
```

Every Episode receives a fresh Policy process, instance, and scratch directory.
State may persist between `act()` calls within that Episode. Invalid Actions
are never repaired, and trusted Environment failures are not converted into
Policy penalties.

## Coding-agent runs

`run()` gives a coding agent a fixed `workspace/` containing an editable
`program/` and Benchmark-authorized `feedback/`:

```python
from cartpole import CartPoleBenchmark, baseline_program

from evopolicygym import RunConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress

result = run(
    baseline_program(),
    CartPoleBenchmark(),
    agent=Codex(model="gpt-5.6-luna"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        max_submissions=16,
        episode_budget=48,
        max_episodes_per_submission=3,
    ),
    observer=ConsoleProgress(),
)
```

During the Run, the agent evaluates and selects immutable submissions with:

```console
evopolicygym submit program --episodes 3
evopolicygym finish submission-000002
```

The Host retains submitted Programs, public Feedback and Artifacts,
`events.jsonl`, the final `run.json`, and separate Agent logs. Benchmark
authors control the public Feedback content and may publish bounded traces,
replays, diagnostics, images, or reports without exposing private seeds,
paths, or execution evidence.

`ProcessExecution` is **not a sandbox**. The Agent and Policy processes run
with the authority of the current operating-system user. Use it only with
trusted code; whole-Run virtualization is planned for a later release.

## Authoring environments

External packages implement the structural `Benchmark` and `Environment`
interfaces from `evopolicygym.authoring`. An Environment owns reset, step, and
cleanup behavior. A Benchmark owns deterministic Episode planning, scoring,
sanitized Feedback, and public Artifacts.

Use `check_benchmark()` with deterministic fixtures before distribution. See
the [authoring guide](https://linzwcs.github.io/EvoPolicyGym/docs/authoring/)
and [CartPole reference package](environments/cartpole/).

## Development

```console
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build
```

EvoPolicyGym 0.3 is an alpha release. The current Kernel intentionally does not
provide process isolation, crash recovery, or Run resumption. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before
contributing runtime changes.

## Citation

The paper and its reported experiments correspond to the
[`v0.1.0`](https://github.com/Linzwcs/EvoPolicyGym/tree/v0.1.0) research
implementation. If you use EvoPolicyGym in research, please cite:

```bibtex
@article{wang2026evopolicygym,
  title   = {EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments},
  author  = {Wang, Zhilin and Song, Han and Zhan, Runzhe and Du, Jusen and
             Chen, Jiacheng and Li, Tianle and Yin, Qingyu and Wu, Yulun and
             Shen, Zhennan and Zhu, Tong and Li, Yanshu and Chen, Guanjie and
             Wong, Derek F. and Li, Yafu and Cheng, Yu and Yang, Yang},
  journal = {arXiv preprint arXiv:2607.02440},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.02440}
}
```

## License

The EvoPolicyGym Kernel is released under the [MIT License](LICENSE).
Environment distributions may include separately attributed dependencies; see
their package documentation for details.
