# Case Layer

> Status: design note. Protocol-facing rules live in `docs/protocol/`.

## Goal

The case layer separates the OJ-facing API from environment internals. Agents submit integer `env_instances`; EvoPolicyGym maps those integers into pool-scoped `Case` objects before calling the environment. This keeps the agent interface simple while giving env authors a stable place to hang hidden seeds, scenario refs, dataset rows, or level ids.

## Vocabulary

| Name | Owner | Meaning |
|---|---|---|
| `Task.cases` | protocol | Number of visible train cases addressable by `env_instances` |
| `Submit.cases` | agent request | Ordered integer ids requested for one train submit |
| `Pool` | judge | A train/validation/final split with `kind`, `size`, opaque `ref`, and optional concrete cases |
| `Case` | runtime | One pool-scoped instance: `id`, opaque `ref`, and server-only `data` |
| `Secret` | env registration | Judge-only refs and score anchors; never exposed through `/info` |

## Mapping Rule

`Pool.case(i)` is the only baseline mapping from an integer id to runtime case metadata. Without an external data directory, `Env.pool(kind)` derives synthetic refs from task and secret metadata for smoke tests.

```python
pool = Pool(kind=PoolKind.train, size=8, ref="toy/train")
case = pool.case(2)  # Case(id=2, ref="toy/train/000002")
```

Formal runs should pass an external data directory. The host loads `train.json`, `valid.json`, and `heldout.json`, then builds concrete `Case` objects whose `data` may contain seeds, rows, scenario ids, or level parameters.

Use `evopolicygym data make` for seed-backed splits when an environment only needs deterministic reset seeds:

```bash
uv run evopolicygym data make \
  --env gym/taxi \
  --root data/gym/taxi \
  --seed 0 \
  --train-size 64 \
  --valid-size 64 \
  --heldout-size 256
```

```json
{
  "env": "cartpole",
  "split": "train",
  "cases": [{"seed": 1001}, {"seed": 1002}]
}
```

The integer id remains the budget unit and trace label. `Case.ref` is not a seed guarantee; it is an opaque, stable address used only inside the judge/runtime. Agents never receive `Case.data`.

`check_env(env, root)` treats `root` as a data directory with `train.json`, `valid.json`, and `heldout.json`. It validates source existence, JSON shape, pool size, duplicates, and train/validation/held-out overlap.

## Visibility

- Agents see `Task.cases`, `env_instances`, and train feedback only.
- Agents do not see validation/final pool sizes, refs, sampled ids, seeds, or returns.
- Hidden validation/final evaluation uses the same `Pool.case(i)` mechanism internally, but no hidden `Case` is serialized into feedback.
- `summary.json.env_instances` records the original requested integer ids, not internal refs.

## Environment Contract

An environment registration provides:

- `Task`: public spaces, episode length, visible train case count, and reward metadata.
- `Secret`: judge-only train/validation/final refs plus normalization anchors.
- `make() -> World`: factory for isolated episode execution.
- `Env.pool(kind) -> Pool`: split derivation from task and secret metadata.
- `World.reset(case: Case)`: reset using pool-scoped metadata, not raw agent input.
- `World.step(action) -> Turn` and `World.sample()`.

`toy` is the reference shape. It uses `case.data["start"]` or `case.id`. Real environments should read deterministic case parameters from `Case.data`, not from files under `src`.

## Fixed vs Extensible

Fixed baseline:

- Agent submit format: ordered integer ids or specs parsed into integer ids.
- Budget accounting: one requested id costs one episode, including duplicates.
- Pool split names: train, valid, final.
- Runtime boundary: `World.reset(Case)`.

Extensible per environment:

- The external JSON case schema and how `Case.data` maps to seeds, files, rows, or scenarios.
- Pool sizes for validation/final.
- Observation storage, videos, reward components, and final score value functions.
- Additional env-specific resource controls, if declared in `/info.resource_limits`.
