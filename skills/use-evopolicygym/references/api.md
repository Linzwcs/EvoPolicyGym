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

from evopolicygym import RunConfig, ValidationConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress

result = run(
    baseline_program(),
    ExampleBenchmark(),
    agent=Codex(model="MODEL_ID"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/example-001",
    config=RunConfig(
        split="train",
        max_submissions=16,
        episode_budget=48,
        max_episodes_per_submission=4,
        use_benchmark_skill=True,
        validation=ValidationConfig(
            split="validation",
            episodes_per_candidate=10,
            max_candidates=3,
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
```

Use a unique `record_to`. Its parent must already exist and the Run directory
must not. Omit `max_episodes_per_submission` only when the selected Benchmark's
Feedback remains bounded for any allowed batch size and Agent-controlled
allocation is intentional.

During search, the Agent uses:

```console
evopolicygym submit program --episodes 4
evopolicygym finish submission-000003 submission-000011
```

`finish` atomically accepts an ordered candidate list. With
`ValidationConfig`, the Host closes Agent authority, reaps the Agent, evaluates
all candidates on identical private Validation Episodes, and selects by score
direction, fewer Policy failures, then argument order. Without it, exactly one
candidate is accepted. Validation is not returned through workspace Feedback.

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

The Program snapshot includes all regular source files under its directory,
except `.git`, `__pycache__`, and `.pyc` content. Keep the submitted Program
self-contained and import-safe.
