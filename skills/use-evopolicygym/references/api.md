# Public workflows

Use these examples with EvoPolicyGym 0.3. Replace imports from an Environment
distribution with the names documented by that package.

## Direct Evaluation

```python
from example_benchmark import ExampleBenchmark

from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution

program = Program.from_directory("policy")
result = evaluate(
    program,
    ExampleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(
        split="validation",
        episodes=10,
        seed=42,
        episode_timeout_seconds=30,
    ),
)

print(result.feedback.score)
for episode in result.episodes:
    print(episode.status, episode.reward, episode.failure)
for artifact in result.feedback.artifacts:
    print(artifact.name, artifact.media_type, artifact.size)
```

Use direct Evaluation before a Coding Agent Run. An `EvaluationError` denotes
a trusted Benchmark, Environment, runtime-control, cleanup, or Feedback
construction failure; it is not a Policy penalty.

## Codex Program-Evolution Run

```python
from example_benchmark import ExampleBenchmark, baseline_program

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
    ExampleBenchmark(),
    agent=Codex(
        model="MODEL_ID",
        reasoning_effort="high",
    ),
    execution=ProcessExecution.unsafe(),
    record_to="runs/example-001",
    skills=(
        AgentSkill.from_directory("skills/example-policy"),
    ),
    config=RunConfig(
        split="train",
        max_submissions=16,
        episode_budget=48,
        episode_pool_size=96,
        max_episodes_per_submission=4,
        validation=ValidationConfig(
            split="validation",
            episodes_per_candidate=10,
            max_candidates=3,
        ),
        assessment=AssessmentConfig(
            split="test",
            episodes=20,
        ),
        seed=0,
        episode_timeout_seconds=30,
        agent_timeout_seconds=3600,
    ),
    observer=ConsoleProgress(),
)

print(result.terminal_reason)
print(result.candidate_submission_ids)
print(result.final_submission_id)
print(result.validation)
print(result.assessment)
```

Use a unique `record_to`. Its parent must already exist and the Run directory
must not. Omit `max_episodes_per_submission` only when the selected Benchmark's
Feedback remains bounded for any allowed batch size and Agent-controlled
allocation is intentional.

Each selected `AgentSkill` is a complete, pathless directory snapshot with a
required `SKILL.md`. The Run projects it read-only at
`workspace/skills/<name>/` and records its digest. Do not rely on implicit
repository discovery or attach a Skill to `BenchmarkSpec`.

During search, the Agent uses:

```console
evopolicygym-session submit program --episodes "0:2,4:6"
evopolicygym-session finish submission-000003 submission-000011
```

The submit selector addresses the fixed Run-local training Episode pool.
Singletons and half-open ranges may be joined by commas, must expand to a
strictly increasing duplicate-free index list, and consume one budget unit per
selected index. Reusing an index in a later Submission preserves the hidden
Episode specification and Policy seed while still creating fresh runtime
state and consuming budget again. `episode_pool_size` defaults to
`episode_budget`.

`finish` atomically accepts an ordered candidate list. With
`ValidationConfig`, the Host closes Agent authority, reaps the Agent, evaluates
all candidates on identical private Validation Episodes, and selects by score
direction, fewer Policy failures, then argument order. Without it, exactly one
candidate is accepted. Validation is not returned through workspace Feedback.
When `AssessmentConfig` is present, the Host subsequently evaluates only the
selected Program on an independently seeded held-out split. Assessment does
not affect selection and is not returned through workspace Feedback.

## Other command-line Coding Agents

Implement the public structural `CodingAgent` protocol. Translate the
Host-owned `AgentTask` into an `AgentInvocation`; do not duplicate Run
instructions or move process supervision into the provider.

For a simple command integration, use:

```python
from evopolicygym.agents import AgentTask, command_invocation


class ExampleAgent:
    def build_invocation(self, task: AgentTask):
        return command_invocation(
            ("example-agent", "--prompt", task.instructions),
            recorded_command=("example-agent", "--prompt", "@agent/instructions.md"),
            identity={"provider": "example-agent"},
            instructions=task.instructions,
            inherited_environment=("HOME", "EXAMPLE_API_KEY"),
        )
```

Inherit only explicitly required environment-variable names. Never place
credentials in `recorded_command`, identity, instructions, Feedback, or
Artifacts.

## Policy entry point

```python
from evopolicygym.policy import PolicyContext, PolicyValue


class Policy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0


def make_policy(context: PolicyContext) -> Policy:
    return Policy()
```

`PolicyContext.environment_parameters` contains the public, Case-independent
values fixed by the configured Benchmark. It never contains the private
Episode scenario or Environment seed.

The Program snapshot includes all regular source files under its directory,
except `.git`, `__pycache__`, and `.pyc` content. Keep the submitted Program
self-contained and import-safe.
