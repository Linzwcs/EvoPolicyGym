← [protocol index](./README.md)

# Schema Reference

This page defines the compact schemas used in `Task.obs`, `Task.act`,
`obs_space`, `action_space`, and feedback values. `Task.obs` / `obs_space` is
the policy input space, and `Task.act` / `action_space` is the policy output
space. Schemas describe structure; environment-specific semantics live in
`/task` and the environment `task.md`.

## Common Rules

- `type` is required for every schema object.
- `shape` is an array of non-negative integers, for example `[84, 84, 3]`.
- `dtype` is a NumPy-style scalar name such as `float32`, `float64`, `int64`,
  `uint8`, or `bool`.
- `low` and `high` may be scalars, full arrays, or summaries such as
  `{"shape": [84, 84, 3], "min": 0, "max": 255}` for large spaces.
- `fields`, `labels`, and `semantics` are descriptive hints. Policies must not
  rely on them when absent.
- `storage = "inline"` means feedback can embed the value directly.
  `storage = "external"` means feedback may store the value in episode files.

## Space Types

### `Discrete`

A single integer.

```json
{"type": "Discrete", "n": 6, "start": 0, "labels": ["south", "north"]}
```

Valid values are integers in `[start, start + n - 1]`. `start` defaults to `0`.
`labels` is optional and may be partial.

### `Box`

A fixed-shape numeric tensor, represented to policies as a NumPy-compatible
array or JSON-safe nested list.

```json
{
  "type": "Box",
  "shape": [3],
  "dtype": "float32",
  "low": [-1.0, -1.0, -8.0],
  "high": [1.0, 1.0, 8.0],
  "fields": [
    {"name": "cos(theta)", "range": [-1, 1]},
    {"name": "sin(theta)", "range": [-1, 1]},
    {"name": "angular_velocity", "range": [-8, 8]}
  ]
}
```

For large arrays, `low` and `high` should use summaries instead of full arrays.

### `MultiDiscrete`

A fixed-shape integer tensor with per-dimension bounds.

```json
{"type": "MultiDiscrete", "nvec": [5, 2, 2], "start": [0, 0, 0], "dtype": "int64"}
```

Dimension `i` accepts integers in `[start[i], start[i] + nvec[i] - 1]`.

### `MultiBinary`

A binary tensor.

```json
{"type": "MultiBinary", "n": 4}
```

Values are `0/1` integers or booleans with the declared shape.

### `Tuple`

An ordered product of subspaces.

```json
{"type": "Tuple", "spaces": [{"type": "Discrete", "n": 3}, {"type": "Box", "shape": [2], "dtype": "float32"}]}
```

Policy values use the same order as `spaces`.

### `Dict`

A named product of subspaces.

```json
{
  "type": "Dict",
  "spaces": {
    "image": {"type": "Image", "shape": [84, 84, 3], "dtype": "uint8"},
    "mission": {"type": "Text", "max_length": 128}
  }
}
```

Policy values are dictionaries with the same keys.

### `Text`

A string value.

```json
{"type": "Text", "min_length": 0, "max_length": 128, "charset": "utf-8"}
```

`charset` is descriptive unless an environment states stricter rules in
`task.md`.

### `Image`

An image or frame stack. This is a semantic specialization of `Box`; feedback
usually stores it externally.

```json
{
  "type": "Image",
  "shape": [84, 84, 4],
  "dtype": "uint8",
  "layout": "height_width_channels",
  "channels": "oldest_to_newest_grayscale_frames",
  "value_range": [0, 255],
  "storage": "external"
}
```

`layout`, `channels`, and `value_range` explain how to interpret pixels. The
environment task text should describe game objects, colors, coordinate axes, or
frame order when that information matters.

### Unknown Types

Adapters may expose uncommon upstream spaces as:

```json
{"type": "Graph", "repr": "Graph(...)", "storage": "external"}
```

Agents should consult `/task` and feedback examples before relying on unknown
types.

## Feedback Values

Small observations and actions are JSON-safe values matching their schema:
integers for `Discrete`, nested lists for `Box`, arrays for `Tuple`, and objects
for `Dict`.

Large values use episode artifacts. Current baseline feedback writes large
fixed-shape observations automatically when a trajectory would otherwise embed
large arrays inline:

```text
feedback/submit_NNN/episodes/ep_XXX/
├── trajectory.jsonl
├── observations.npy   # optional
└── observations.npz   # optional
```

If `trajectory.jsonl:obs` is `null`, load the observation for step `t` from the
episode observation file at index `t`. `observations.npy` stores one whole
observation array with shape `[episode_length, *obs_shape]`; `observations.npz`
stores one fixed-shape array per top-level dict field.

Feedback may also use an explicit external reference object when only one field
of a nested observation is external:

```json
{
  "type": "External",
  "path": "feedback/submit_000/episodes/ep_003/observations.npz",
  "key": "image",
  "index": 12,
  "shape": [84, 84, 4],
  "dtype": "uint8"
}
```

`path` is relative to the workspace root. `key` is omitted for `.npy` files and
required for `.npz` files with multiple arrays.

## Responsibility Split

- `Task.obs` and `Task.act`: compact machine-readable input/output structure.
- `/task` and env `task.md`: semantic explanation of the current environment.
- `feedback/`: real trajectory examples and optional large observation files.
- `AGENTS.md`: concise agent-facing operating rules, not the full schema spec.
