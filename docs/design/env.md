# Env Contract

> Status: design note. This fixes the server-side shape an environment must provide before real benchmark tasks are added.

## Goal

An `Env` is the complete registration unit for one benchmark task. Host code should not invent environment semantics; it should assemble a run from the contract declared by the environment and only override values for tests or local debugging.

## Contract

An environment registration contains:

| Field | Meaning |
|---|---|
| `task` | Agent-visible task contract: name, version, input `obs` space, output `act` space, max steps, visible train case count, reward metadata |
| `secret` | Judge-only refs, hidden pool sizes, and normalization anchors |
| `make` | Factory returning an isolated `World` for one episode |
| `value` | Optional scalar score function for hidden final scoring |
| `caps` | Optional artifact capabilities such as observations or video; non-empty caps appear in `/info.env_meta.artifacts` |
| `text` | Agent-facing task document, normally loaded from the environment-local `task.md` and served by `/task` |

`Env.pool(kind)` is the default way to derive train/validation/final `Pool` objects for smoke tests. Formal runs should supply an external data directory; host assembly then replaces those defaults with file-backed pools and updates `Task.cases` from `train.json`.

## World Boundary

Runtime calls the environment only through `World`:

```python
class World:
    def reset(self, case: Case) -> obs: ...
    def step(self, action) -> Turn: ...
    def sample(self): ...
```

`Case.id` is the pool-local integer label. `Case.ref` is an opaque stable address such as `toy/train/000002`. `Case.data` is server-only JSON loaded from the external split files; real environments should read seeds, files, rows, or scenario parameters from it without exposing that mapping to agents.

## Fixed Baseline

- `Task.name` and `Task.version` identify the environment version.
- `Task.obs` is the machine-readable policy input space; `Task.act` is the machine-readable policy output space.
- `Task.cases` is the visible train ID count returned as `env_meta.n_env_instances`.
- `Secret.train`, `Secret.valid`, and `Secret.final` are private pool refs.
- `Secret.valid_size` defaults to 64 and `Secret.final_size` defaults to 256.
- `World.reset(Case)` is the only reset boundary.
- `Env.value` may return `None` for validation and should return a scalar for final if final score differs from mean return.

## Extensible Surface

- External case source format: seed list, dataset rows, scenario files, generated levels.
- Artifact capabilities: binary observations, video, or richer diagnostics.
- Reward components and final score normalization.
- Environment-specific resource controls, if exposed through `/info.resource_limits`.

## Reference Env

`toy` is the minimal reference implementation. It uses `Case.data` when present and falls back to `case.id`, declares hidden pool sizes through `Secret`, provides a simple final `value`, and serves a short `/task` document loaded from `envs/toy/task.md`.

## Contract Check

Run `check_env(env)` before adding a new environment to the registry. The checker validates task metadata, private pool refs, `Env.pool(kind)`, case refs, hidden metadata leaks, `World.reset(Case)` / `sample()` / `step()` smoke behavior, artifact caps, and final `value` shape.

Run `check_env(env, root)` with an external data directory. The directory must contain `train.json`, `valid.json`, and `heldout.json`. Each file may be a JSON array or an object with a `cases` array plus optional `env`, `split`, and `ref` metadata. The checker verifies source existence, JSON shape, pool size, duplicate cases, and train/validation/held-out overlap.
