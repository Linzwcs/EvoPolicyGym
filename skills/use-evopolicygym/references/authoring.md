# Benchmark authoring

Build each Benchmark as an independent Python distribution that depends only
on the public EvoPolicyGym SDK and its own Environment dependencies. Use
`environments/gymnasium/classic_control/cartpole/` as the minimal live
reference.

## Package boundary

- Import authoring contracts from `evopolicygym.authoring`.
- Do not import Kernel-private modules, sibling Benchmark packages, Run
  internals, process implementations, sockets, or protocol codecs.
- Own the Benchmark, Environment, baseline Program, dependencies, tests, and
  package documentation inside the distribution.
- Keep the Environment package out of the base EvoPolicyGym wheel.

## Implement the contracts

Define a `Benchmark` with:

- `spec`: immutable public `BenchmarkSpec`, including the exact applied
  `environment_parameters`;
- `episodes(split, seed, count)`: exactly `count` deterministic
  split-and-seed-scoped `EpisodeSpec` values;
- `make_environment(episode)`: one fresh Environment;
- `feedback(records)`: one sanitized public `Feedback`.

Expose simulator-specific options through a typed distribution-owned
Benchmark constructor. Validate and bind them once, publish the exact public
values in `BenchmarkSpec.environment_parameters`, and use that same bound
configuration from `make_environment()`. Do not add generic Environment
keyword arguments to Kernel Run or Evaluation configuration. The Kernel
computes `environment_digest` and uses it with the Benchmark ID to prevent
mixing results from different configured tasks.

Define an `Environment` with:

- `reset() -> PolicyValue`;
- `step(action) -> Step`;
- `close()`.

Raise `InvalidAction` for an out-of-domain Action. Do not repair it. Return a
`Step` only for a valid Action, and terminate or truncate within
`BenchmarkSpec.max_episode_steps`.

## Preserve privacy and determinism

- Keep `EpisodeSpec`, Environment seeds, Policy seeds, Case identity, and
  scorer objects on the trusted side.
- Keep public, Case-independent Environment parameters separate from private
  per-Episode `scenario` values.
- Emit only PolicyValue observations, Actions, metadata, metrics, and
  Benchmark-authorized Feedback across public boundaries.
- Do not publish Host paths, descriptors, credentials, process evidence, or
  private seed-derived identifiers.
- Make reset and step deterministic under the Episode specification and Action
  sequence.
- Close every Environment successfully, including failure paths.

## Design Feedback

Return:

```python
Feedback(
    score=finite_number,
    content=benchmark_defined_policy_value,
    artifacts=(Artifact(...),),
)
```

The scalar score is required. Content and Artifact schemas belong to the
Benchmark. One Artifact is limited to 16 MiB; one Feedback contains at most 64
Artifacts and 64 MiB total. Bound traces independently of requested Episode
count while keeping aggregate scoring based on every record.

Keep Coding Agent Skills outside the Benchmark distribution contract.
Benchmark authors may document compatible Skills, but `BenchmarkSpec` does not
publish or select them. A caller freezes the desired directories with
`AgentSkill.from_directory()` and passes them explicitly to `run()`.

The same `episodes()`, `make_environment()`, and `feedback()` methods may be
used for server-side candidate Validation and final Assessment. Keep training,
Validation, and held-out splits genuinely disjoint when the Benchmark promises
disjoint data. The Kernel retains only aggregate score and Policy-failure
counts from these Host phases; Benchmark-defined content and Artifacts are not
published to the Agent workspace.

## Test before distribution

Use `BenchmarkFixture` and `check_benchmark()` to replay deterministic Action
sequences twice. Add unittest coverage for:

- reproducible, split-scoped Episode planning;
- fresh Environment state and reliable cleanup;
- valid termination and maximum-step behavior;
- exact invalid-Action classification;
- Policy failure without an extra Environment step;
- finite scoring, Feedback privacy, and Artifact byte bounds;
- baseline direct Evaluation through `ProcessExecution.unsafe()`;
- build isolation and installation of the independent wheel.

Run the Environment distribution's Ruff, strict mypy, unittest, and `uv build`
checks from its own project directory.
