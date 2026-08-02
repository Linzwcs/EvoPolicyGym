# Direct Policy Evaluation

Use direct Evaluation to check a Program before spending Coding Agent Run
authority.

## Build a valid Program

Place a `policy.py` file in the Program directory:

```python
from evopolicygym.policy import PolicyContext, PolicyValue


class Policy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0


def make_policy(context: PolicyContext) -> Policy:
    return Policy()
```

Keep the Program self-contained and import-safe. The snapshot includes regular
source files under the directory except `.git`, `__pycache__`, and `.pyc`
content.

- Export `make_policy(context: PolicyContext)`.
- Return an object with `act(observation: PolicyValue) -> PolicyValue`.
- Read Case-independent public configuration from
  `PolicyContext.environment_parameters`.
- Never expect private Episode scenarios or Environment seeds in the context.
- Return exact Actions; the Kernel does not repair invalid Actions.
- Expect a fresh process, Policy instance, and scratch directory for every
  Episode. Persist state only between `act()` calls in the same Episode.

## Evaluate through the public SDK

Replace the example import with the names documented by the selected
Environment distribution:

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

State the lack of process isolation before evaluating code that is not already
trusted.

## Interpret the result

- Treat `policy_failed` as a scored Policy outcome with an `exception`,
  `timeout`, `invalid_action`, or `protocol_error` failure code.
- Treat `EvaluationError` as a trusted Benchmark, Environment,
  runtime-control, cleanup, or Feedback-construction failure, not a Policy
  penalty.
- Assume only that Feedback has a finite scalar score,
  Benchmark-defined public content, and bounded public Artifacts.
- Let the Benchmark define trace, replay, diagnostic, image, or report schemas.
  Do not impose a Kernel-wide trace format.
- Never infer hidden seeds, Case identity, or future Environment state from
  public Feedback.

Verify imports, deterministic setup, legal Actions, termination, cleanup, and
Artifact bounds. Report the Benchmark and environment digest, Program digest,
split, seed, Episode count, aggregate score, Episode failures, retained
Artifacts, verification performed, and the remaining lack of process
isolation.
