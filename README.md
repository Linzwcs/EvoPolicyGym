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

For AI-assisted setup, Benchmark integration, runs, and diagnostics, this
repository ships the reusable
[`use-evopolicygym` Agent Skill](skills/use-evopolicygym/). Install that folder
with a compatible Agent skill manager and invoke it as `$use-evopolicygym`.
Task-specific workflows, such as
[`optimize-balatro-policy`](skills/optimize-balatro-policy/), live beside it
and are selected explicitly per Run rather than embedded in a Benchmark.

## Environments

Environment distributions are independent packages that depend only on the
public EvoPolicyGym SDK. The catalog now spans Gymnasium, MiniGrid and BabyAI,
HighwayEnv, Gymnasium-Robotics, MetaWorld, ALE, ViZDoom, Stable-Retro, and
Balatro, together with independently implemented AtCoder and CodeChef tasks.
Registered task, size, and difficulty variants are selected through each
Benchmark's typed environment configuration. See the
[complete integration ledger](environments/STATUS.md) for exact coverage and
the environments intentionally deferred by ABI, runtime, or asset boundaries.

| Collection | Contents | Description |
| --- | --- | --- |
| [AtCoder AHC](environments/atcoder/) | AHC054 Treant's Forest, AHC057 Molecules, and AHC058 Apple Incremental Game | Long-horizon constraint placement, moving-component scheduling, hierarchical investment, and horizon-aware planning |
| [CodeChef Challenges](environments/codechef/) | WAREHOUS Warehouseman | Full-range constructive routing, storage, retrieval, and instruction-cost optimization |
| [Gymnasium Box2D](environments/gymnasium/box2d/) | LunarLander, BipedalWalker, and CarRacing | Parameterized landing, locomotion, and pixel-based driving |
| [Gymnasium Classic Control](environments/gymnasium/classic_control/) | CartPole, Acrobot, both Mountain Car variants, and Pendulum | Five independently installable control Benchmarks with semantic observations and public traces |
| [Gymnasium MuJoCo](environments/gymnasium/mujoco/) | All eleven current `v5` tasks | Parameterized continuous-control physics using official packaged models and semantic nested observations |
| [Gymnasium Toy Text](environments/gymnasium/toy_text/) | Blackjack, CliffWalking, FrozenLake, and Taxi | All four standard Toy Text tasks with typed rule and dynamics parameters |
| [MiniGrid](environments/minigrid/) | 21 standard families, all 22 WFC presets, and 40 requested BabyAI tasks | Partially observable language-conditioned navigation, procedural layouts, compositional instructions, and Episode-local memory |
| [HighwayEnv](environments/highway_env/) | All ten canonical single-agent tasks | Discrete and continuous autonomous-driving profiles |
| [Gymnasium-Robotics](environments/gymnasium_robotics/) | 21 Fetch, Maze, Adroit, Shadow Hand, and FrankaKitchen profiles | Goal-conditioned manipulation, navigation, touch sensing, and long-horizon robotics |
| [MetaWorld](environments/metaworld/) | All 50 MT1 tasks, MT10, MT50, and custom collections | Host-selected single-task and multi-task manipulation |
| [ALE](environments/ale/) | Redistributable Tetris profile | Atari RGB control without an external ROM dependency |
| [ViZDoom](environments/vizdoom/) | 12 wheel-bundled standard scenarios | First-person RGB, game-variable, audio, and hybrid-action control |
| [Stable-Retro](environments/stable_retro/) | Redistributable Airstriker Level 1 profile | Console RGB control without an external ROM dependency |
| [Jackdaw](environments/jackdaw/) | Balatro | Unofficial long-horizon Red Deck, White Stake Benchmark powered by a pinned Jackdaw engine |
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
cd environments/gymnasium/classic_control/cartpole
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
    print(context.environment_parameters)
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

A Benchmark distribution may expose a configured task through
`BenchmarkSpec.environment_parameters`. These values are fixed before an
Evaluation or Run, visible to both the Coding Agent and every Policy instance,
and included in the Evaluation and retained Run identity. Per-Episode
`scenario` values and Environment seeds remain trusted and Policy-invisible.

## Coding-agent runs

`run()` gives a coding agent a fixed `workspace/` containing an editable
`program/` and Benchmark-authorized `feedback/`. Agent Skills are independent
experiment inputs selected explicitly by the caller. Complete Skill directory
snapshots appear read-only under `workspace/skills/` and never enter the
Policy process:

```python
from cartpole import CartPoleBenchmark, baseline_program

from evopolicygym import (
    AssessmentConfig,
    RunConfig,
    ValidationConfig,
    run,
)
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress
from evopolicygym.skills import AgentSkill

result = run(
    baseline_program(),
    CartPoleBenchmark(),
    agent=Codex(
        model="gpt-5.6-luna",
        reasoning_effort="high",
    ),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    skills=(
        AgentSkill.from_directory("path/to/task-skill"),
    ),
    config=RunConfig(
        max_submissions=16,
        episode_budget=48,
        episode_pool_size=96,
        max_episodes_per_submission=3,
        validation=ValidationConfig(
            split="validation",
            episodes_per_candidate=10,
            max_candidates=3,
        ),
        assessment=AssessmentConfig(
            split="test",
            episodes=20,
        ),
    ),
    observer=ConsoleProgress(),
)
```

`Codex.reasoning_effort` is a required provider-specific experiment input.
The provider passes it to the Codex CLI as `model_reasoning_effort` and
retains it in the Run's Agent identity. Supported values are model-dependent
and are validated authoritatively by the installed Codex CLI.

An `AgentSkill` is a pathless, content-addressed snapshot containing
`SKILL.md` and any referenced files, scripts, or assets. A Run accepts up to
16 uniquely named Skills. Their names, digests, and retained workspace paths
are recorded in `run.json`, making with-Skill and without-Skill comparisons
explicit and reproducible. The combined snapshot is also bounded to 2,048
files and 64 MiB. Benchmarks never select or load Skills themselves.

During the Run, the agent evaluates immutable submissions and hands candidates
to the Host with:

```console
evopolicygym submit program --episodes "0:2,4:8"
evopolicygym finish submission-000002 submission-000007
```

`RunConfig.seed` deterministically creates one fixed Host-owned training
Episode pool before the Agent starts. `episode_pool_size` defaults to the total
Episode budget. The Agent selects public Run-local indices from
`0..episode_pool_size-1`; ranges are half-open and comma-separated items form a
union. The example selects indices `0, 1, 4, 5, 6, 7`. A selection must be
non-empty, strictly increasing, and contain no duplicate or overlapping
indices.

The same index always binds the same trusted `EpisodeSpec` and Policy seed
within one Run, so the Agent can compare Program revisions on matched
conditions. Every evaluation still creates a fresh Environment, Policy
process, Policy instance, and scratch directory. Reusing an index in a later
Submission is allowed and consumes another Episode budget unit. Actual
Environment seeds and scenarios are not published to the Agent; Validation
and Assessment use separate Host-only Episode plans that cannot be selected
through `submit`.

`finish` atomically hands an ordered candidate set to the Host and closes Agent
authority. With `ValidationConfig`, the Host waits for the Agent process to be
reaped, evaluates every candidate on the same private Validation Episodes, and
selects by the Benchmark score direction, fewer Policy failures, then argument
order. Validation has a separate Episode allocation and is never published
back into workspace Feedback. Without `ValidationConfig`, `finish` accepts
exactly one candidate and the Host selects it after Agent cleanup.

With `AssessmentConfig`, the Host then evaluates only the selected final
Program on a separately seeded held-out split. Assessment never changes the
selection and is never returned to the Agent. Its aggregate score and Policy
failure count are the Run's final benchmark evidence.

The Host retains submitted Programs, public Feedback and Artifacts,
content-addressed Agent Skill snapshots, `events.jsonl`, the final `run.json`,
separate Agent logs, and—when configured—aggregate
`validation/report.json` and `assessment/report.json` records. Benchmark
authors control the public Feedback content and may publish bounded traces,
replays, diagnostics, images, or reports without exposing private seeds,
paths, or execution evidence. Submission Feedback includes the selected
Run-local Episode index beside each public Episode summary.

`ProcessExecution` is **not a sandbox**. The Agent and Policy processes run
with the authority of the current operating-system user. Use it only with
trusted code; whole-Run virtualization is planned for a later release.

## Authoring environments

External packages implement the structural `Benchmark` and `Environment`
interfaces from `evopolicygym.authoring`. An Environment owns reset, step, and
cleanup behavior. A Benchmark owns deterministic Episode planning, scoring,
sanitized Feedback, and public Artifacts. Environment-specific constructors
validate typed parameters and bind them to the Benchmark instance; the
corresponding `BenchmarkSpec.environment_parameters` records the exact public
values that `make_environment()` applies. The generic Kernel does not accept or
interpret simulator-specific keyword arguments.

For example, the FrozenLake distribution accepts
`FrozenLakeConfig(map_name="8x8", is_slippery=True)`, publishes that
configuration in its specification, and uses the bound values whenever it
creates a fresh Environment.

Use `check_benchmark()` with deterministic fixtures before distribution. See
the [authoring guide](https://linzwcs.github.io/EvoPolicyGym/docs/authoring/)
and the [CartPole](environments/gymnasium/classic_control/cartpole/) and
[FrozenLake](environments/gymnasium/toy_text/frozen_lake/) packages.

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
