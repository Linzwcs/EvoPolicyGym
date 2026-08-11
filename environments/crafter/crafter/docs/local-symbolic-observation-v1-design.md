# Crafter local-symbolic observation v1

Status: implemented experimental observation contract (2026-08-10).

This document defines an information-bounded symbolic observation profile for
the pinned Crafter 1.8.3 Benchmark distribution. The profile is inspired by
Craftax's symbolic interface, but it remains a Crafter task: it does not import
Craftax, change Crafter dynamics, add Actions, or change any scoring profile.

The intended experiment separates two Policy-authoring difficulties:

1. extracting local game state and HUD values from a `64 x 64` RGB frame; and
2. using that state for survival, exploration, combat, and development.

`local-symbolic-v1` removes most of the first difficulty while preserving the
second. It deliberately does not turn Crafter into a globally observed planning
task.

## 1. Decisions

The v1 implementation must satisfy all of the following decisions.

- Keep the existing RGB profile and every existing public default unchanged.
- Add `observation_profile="local-symbolic-v1"` as an explicit, immutable
  `CrafterConfig` selection. The default remains `"rgb"`.
- Use the same pinned `crafter==1.8.3` simulator, Episode planning, Actions,
  transition order, termination rules, reward profile, and aggregate scoring as
  the corresponding RGB Benchmark.
- Publish only a player-centered local spatial crop, named inventory values,
  facing direction, sleeping state, and daylight.
- Never publish the upstream global semantic map, absolute player position,
  Environment seed, world RNG, achievement counters, or hidden simulator
  counters as a Policy observation.
- Use a distinct Benchmark ID and Environment digest for the symbolic task.
- Make the symbolic profile available to every existing Crafter scoring class;
  observation selection is orthogonal to scoring selection.
- Keep Benchmark-specific strategy instructions in a new caller-selected Agent
  Skill. Do not add a Skill name or Skill contents to `BenchmarkSpec`.
- Do not require a Kernel change.

The first release is symbolic-only, not hybrid RGB-plus-symbolic. A hybrid
profile can be proposed separately if an experiment needs it.

## 2. Compatibility and public identity

`CrafterConfig` gains one field:

```python
observation_profile: Literal["rgb", "local-symbolic-v1"] = "rgb"
```

Validation must require an exact string and one of the two published values.
Existing constructors that do not pass the field retain byte-for-byte RGB
Policy observations and their existing Benchmark IDs.

For RGB, IDs remain unchanged. For local symbolic, insert the observation
profile between the Environment name and metric name. For example:

```text
RGB canonical:
crafter/CrafterReward-v1/achievement-score-v1

Local-symbolic canonical:
crafter/CrafterReward-v1/local-symbolic-v1/achievement-score-v1

RGB LHS:
crafter/CrafterReward-v1/long-horizon-survival-score-v1

Local-symbolic LHS:
crafter/CrafterReward-v1/local-symbolic-v1/
  long-horizon-survival-score-v1
```

The line break in the last example is for display only. The actual ID is one
continuous string.

Every symbolic specification must publish at least these exact Environment
parameters in addition to the existing Crafter parameters:

```text
observation_profile = "local-symbolic-v1"
symbolic_view_rows = 7
symbolic_view_columns = 9
symbolic_player_row = 3
symbolic_player_column = 4
```

The RGB default must preserve its existing Environment parameters and
historical Environment digest, so it does not add an
`observation_profile="rgb"` parameter. Its existing RGB observation space and
Benchmark ID already identify that task. The symbolic profile publishes the
new observation parameters above and has a distinct Benchmark ID, so its
Environment digest cannot collide with RGB. Add a regression test for the
historical default RGB digest.

`include_mp4_feedback=True` is not supported with `local-symbolic-v1` in v1.
`CrafterConfig` must reject that combination. A later version may add a derived
symbolic replay, but must not label it as upstream RGB evidence.

## 3. Policy observation contract

The complete observation is one `dict[str, PolicyValue]` with exactly these
keys:

```python
{
    "terrain": TensorValue(...),
    "entities": TensorValue(...),
    "inventory": {...},
    "facing": "down",
    "sleeping": False,
    "daylight": 0.82,
}
```

No other fields are present.

### 3.1 Local geometry

Crafter is constructed with `view=(9, 9)`. Its RGB renderer reserves two cells
along the second internal view axis for the 16 inventory items, leaving an
internal spatial grid of `9 x 7` in `[x, y]` order. The public symbolic grid
transposes that crop into row-major `[row, column]` order:

```text
shape:             (7, 9)
row 0:             north/up edge of the visible crop
row 6:             south/down edge of the visible crop
column 0:          west/left edge of the visible crop
column 8:          east/right edge of the visible crop
player location:   [3, 4]
```

The crop is centered on the current player position after reset and after each
Action. It must not contain a larger radius than the RGB spatial view. At a
world boundary, cells outside the world use the public unknown ID `0` in both
spatial tensors.

The local grid remains semantic at night. This is an intentional reduction of
the RGB perception problem, not additional spatial visibility: the Policy sees
clear labels for the same local cells but never sees cells outside the local
crop.

Both tensors use C-order row-major bytes.

### 3.2 Terrain tensor

`terrain` is:

```text
carrier: TensorValue
dtype:   uint8
shape:   (7, 9)
```

The stable public IDs are:

| ID | Meaning |
| ---: | --- |
| 0 | unknown or outside the world |
| 1 | water |
| 2 | grass |
| 3 | stone |
| 4 | path |
| 5 | sand |
| 6 | tree |
| 7 | lava |
| 8 | coal |
| 9 | iron |
| 10 | diamond |
| 11 | crafting table |
| 12 | furnace |

These are Benchmark-owned IDs. The extractor must translate from the pinned
upstream representation and verify that the complete upstream material table
still matches the expected contract. It must not rely on undocumented numeric
IDs without checking them.

Terrain and entities are separate tensors. A creature standing on grass, for
example, leaves `grass` in `terrain` and places the creature ID in `entities`.
This preserves information visible around transparent sprites and is easier to
reason about than the upstream `SemanticView`, which overwrites terrain with an
object ID.

### 3.3 Entity tensor

`entities` is:

```text
carrier: TensorValue
dtype:   uint8
shape:   (7, 9)
```

The stable public IDs are:

| ID | Meaning |
| ---: | --- |
| 0 | no entity or outside the world |
| 1 | player |
| 2 | cow |
| 3 | zombie |
| 4 | skeleton |
| 5 | arrow moving left |
| 6 | arrow moving right |
| 7 | arrow moving up |
| 8 | arrow moving down |
| 9 | young plant |
| 10 | ripe plant |
| 11 | fence |

Arrow direction and plant ripeness are included because their upstream RGB
textures expose those distinctions. Exact entity health, attack cooldown,
skeleton reload counter, plant age, and any other hidden object field are not
included.

Crafter 1.8.3 permits at most one object per world cell. The extractor must
verify this through the pinned world object map. An unknown visible object type,
an invalid direction, or multiple objects in one cell is a trusted Environment
fault, not a new public `unknown entity` value.

### 3.4 Inventory mapping

`inventory` is a dictionary with exactly the following keys and exact integer
values:

```text
health
food
drink
energy
sapling
wood
stone
coal
iron
diamond
wood_pickaxe
stone_pickaxe
iron_pickaxe
wood_sword
stone_sword
iron_sword
```

Every value must be an exact Python `int` in `[0, 9]`. The field exposes no
information beyond the icons and decimal counts already rendered in the RGB
HUD; it removes the need for Policy-side icon recognition and OCR.

The observation does not include achievement counters. Although upstream
Crafter returns achievement and inventory dictionaries together in `info`,
only the inventory dictionary belongs in the Policy observation. Achievement
events may remain in trusted transition metrics and sanitized training
Feedback as they do for the RGB profile.

### 3.5 Facing, sleeping, and daylight

`facing` is one exact string from:

```text
left
right
up
down
```

It records the post-transition direction represented by the player sprite.
Attempting a blocked movement still changes facing in upstream Crafter, so the
extractor must read the resulting player state rather than infer direction from
successful displacement.

`sleeping` is an exact Python `bool` matching the post-transition player sprite
and sleep state.

`daylight` is a finite Python `float` in `[0.0, 1.0]` matching the upstream
world daylight used to render the current observation. It is public because
day/night is visibly encoded in RGB and is central to survival planning. The
extractor must not add the upstream step counter or expose a separate hidden
time phase.

## 4. Explicit privacy boundary

The following table is normative.

| Value | Policy observation | Trusted scoring/Feedback use |
| --- | --- | --- |
| Local `7 x 9` terrain | Allowed | Allowed |
| Local visible entity type/variant | Allowed | Allowed |
| Named inventory counts | Allowed | Allowed |
| Facing, sleeping, daylight | Allowed | Allowed |
| Full upstream `64 x 64 semantic` array | Forbidden | Do not publish |
| Absolute `player_pos` | Forbidden | Do not publish |
| Environment or Policy seed | Forbidden | Do not publish |
| World RNG or seed-derived identifier | Forbidden | Do not publish |
| Achievement counters | Forbidden | Aggregate/unlock diagnostics only |
| `_hunger`, `_thirst`, `_fatigue`, `_recover` | Forbidden | Do not publish |
| Entity health/cooldowns/reload | Forbidden | Do not publish |
| Exact plant age | Forbidden | Do not publish; ripe/young is allowed |
| World chunks, object indices, Host paths | Forbidden | Do not publish |

The upstream `info["semantic"]` and `info["player_pos"]` values must never be
copied wholesale into `Step.metrics`, `Feedback.content`, JSONL, NPZ, debug
text, or exception messages. Absolute position may be used transiently on the
trusted side only to select the local crop.

The symbolic observation is Case-dependent by design, but it contains no Case
identity. Public metadata may describe the fixed schema; it must not contain a
sample generated from an Episode.

## 5. Trusted extraction design

Add a small, focused module such as:

```text
src/crafter_benchmarks/symbolic.py
```

It owns:

- upstream structural Protocols used for static typing;
- runtime validation of the pinned Crafter world/player representation;
- local coordinate conversion and boundary padding;
- terrain and entity translation;
- inventory, facing, sleeping, and daylight validation; and
- construction of the bounded `PolicyValue` observation.

Do not distribute extraction logic across scoring modules or Feedback
formatters.

Upstream Crafter exposes a global semantic array only in `step()` information,
and `reset()` returns RGB alone. The implementation must therefore extract both
initial and post-Action symbolic observations from the trusted environment
state. It must not execute an implicit noop during reset. An implicit noop would
advance survival counters, RNG, creatures, reward, and horizon before the first
Policy Action and would violate the Evaluation contract.

The sequence is:

```text
reset:
  upstream reset
  validate returned RGB carrier
  stabilize pinned Crafter chunk iteration
  extract current local symbolic state
  return symbolic PolicyValue

step:
  validate exact Action
  upstream step exactly once
  validate returned RGB/info/done values
  update trusted achievement and scoring state
  extract post-Action local symbolic state
  return Step with symbolic observation
```

The extractor must not consume randomness or mutate the world. For any fixed
EpisodeSpec and Action sequence, changing only `observation_profile` must leave
upstream rewards, terminal flags, achievement events, inventory evolution, and
world state unchanged.

The existing adapter already audits pinned private Crafter structures to remove
address-dependent chunk iteration. Using additional pinned internals for
symbolic extraction is acceptable only with equally strict runtime drift checks
and real-environment tests. Missing or malformed upstream state is an
Environment fault. Do not silently fall back to RGB, emit an incomplete
observation, or substitute guessed values.

## 6. Benchmark specification

For the symbolic profile, `BenchmarkSpec.observation_space` should describe the
complete named mapping, not only say `type="dict"`. A representative structure
is:

```python
{
    "type": "mapping",
    "fields": {
        "terrain": {
            "policy_carrier": "TensorValue",
            "dtype": "uint8",
            "shape": [7, 9],
            "layout": "row-major [row, column]",
            "value_meanings": {...},
        },
        "entities": {
            "policy_carrier": "TensorValue",
            "dtype": "uint8",
            "shape": [7, 9],
            "layout": "row-major [row, column]",
            "value_meanings": {...},
        },
        "inventory": {
            "policy_carrier": "mapping of exact int",
            "keys": [...],
            "minimum": 0,
            "maximum": 9,
        },
        "facing": {
            "policy_carrier": "str",
            "values": ["left", "right", "up", "down"],
        },
        "sleeping": {"policy_carrier": "bool"},
        "daylight": {
            "policy_carrier": "float",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}
```

Metadata must name the observation contract, geometry, player center, terrain
and entity tables, inventory keys, and privacy boundary. It must state that the
profile is a Benchmark-authored local symbolic projection of Crafter 1.8.3,
not an upstream registered Environment and not Craftax.

The Action space and meaning remain the exact existing Crafter `0..16`
contract. No action mask or prerequisite-validity hint is added. Placement and
crafting Actions that lack prerequisites remain valid domain Actions with their
ordinary no-effect behavior.

## 7. Scoring invariance

Observation selection does not alter scoring.

For every existing scoring class, a fixed EpisodeSpec and fixed Action sequence
must produce the same:

- number of upstream transitions;
- natural termination or horizon truncation;
- upstream transition rewards;
- achievement event order and counts;
- shaped transition components, when applicable;
- Episode return; and
- aggregate Feedback scalar.

The only intended differences are:

- `BenchmarkSpec.id`;
- observation space and description;
- Environment digest;
- Policy observations; and
- observation Feedback artifacts.

Scores from independently chosen RGB and symbolic Policies answer a task
difficulty question and should be reported as different Benchmark profiles.
They are not interchangeable canonical Crafter results. A fixed-Action
lockstep test may compare their dynamics and scores exactly.

## 8. Training Feedback and artifacts

Trajectory JSONL retains the existing Action, reward, terminal, scoring, and
achievement-event diagnostics. It must add the public
`observation_profile="local-symbolic-v1"` to its header or manifest and must not
add global semantic state or absolute position.

Symbolic observations are small enough to retain completely for the existing
detailed-training Episode limit. Store them losslessly in chunked NPZ files
with source alignment identical to Policy calls:

```text
terrain:    uint8   [observation, 7, 9]
entities:   uint8   [observation, 7, 9]
inventory:  uint8   [observation, 16]
facing:     uint8   [observation]
sleeping:   bool    [observation]
daylight:   float64 [observation]
```

`inventory` uses the exact order listed in section 3.4. The NPZ manifest must
repeat that order. `facing` uses artifact-only IDs
`0=left, 1=right, 2=up, 3=down`; the live Policy carrier remains a string.

The initial observation is index `0`; transition `t`'s result observation is
index `t + 1`. Chunk boundaries must not omit or duplicate observations. Keep
the existing observation-per-artifact bound unless measured artifact sizes
justify a smaller public constant.

Use a symbolic-specific manifest schema such as:

```text
crafter/local-symbolic-feedback-manifest/v1
```

Do not reinterpret an existing RGB manifest version. Symbolic NPZ artifacts
remain authoritative. A derived colored-grid GIF may be added in a later
version, but it is not required for v1 and must be labeled as a viewing aid.

Validation and test splits continue to publish aggregate results only. They do
not publish symbolic trajectories or NPZ observations. Existing Artifact count,
per-Artifact byte, total byte, retention, and detailed-Episode limits remain in
force.

## 9. Baseline Program and Agent Skill

The current RGB baseline and `optimize-crafter-policy` Skill must remain
unchanged for RGB Runs.

Add a separate packaged baseline entry point, for example:

```python
local_symbolic_baseline_program() -> Program
```

The baseline must:

- validate the complete observation mapping and tensor shapes;
- return exact integer Actions in `0..16`;
- contain no fixed seed route or absolute-coordinate logic;
- use only local terrain/entities, inventory, facing, sleeping, daylight, and
  Episode-local memory; and
- remain intentionally weak but executable through direct Evaluation.

For the RGB-versus-symbolic ablation, the implemented baseline validates the
different observation ABI and then mirrors the RGB baseline's seeded fallback
Action stream. It intentionally does not exploit nearby symbolic labels before
Policy evolution begins. Otherwise any baseline survival gain would be
confounded with a hand-authored symbolic heuristic rather than measuring the
coding Agent's ability to use the clearer observation.

Add a new caller-selected Skill directory:

```text
skills/optimize-crafter-local-symbolic-policy/
```

Its `SKILL.md` should describe:

- live Policy carriers and row-major decoding;
- terrain/entity tables and coordinate orientation;
- player center and facing-to-front-cell mapping;
- the unchanged 17 Actions and progression dependencies;
- the fact that exact legal prerequisites are not supplied as an action mask;
- survival, facility adjacency, exploration, and Episode-local memory; and
- the symbolic NPZ and trajectory Feedback schemas.

The wheel may package this Skill as a resource, but the Benchmark does not
select it. `run_crafter_codex.py` freezes and passes the matching Skill only
when the caller enables Benchmark Skills. Add an
`--observation-profile {rgb,local-symbolic-v1}` launcher option and select the
matching baseline and Skill. Do not pass both Crafter strategy Skills in one
Run.

## 10. Required implementation changes

The implementation session should expect changes in these ownership areas:

```text
src/crafter_benchmarks/config.py
  add and validate observation_profile

src/crafter_benchmarks/constants.py
  own stable symbolic terrain/entity/inventory/facing tables

src/crafter_benchmarks/symbolic.py
  new trusted extraction and validation module

src/crafter_benchmarks/environment.py
  select RGB or symbolic Policy observation without changing stepping/scoring

src/crafter_benchmarks/benchmark.py
  select spec identity/schema and RGB or symbolic Feedback artifacts

src/crafter_benchmarks/baseline.py
src/crafter_benchmarks/programs/local_symbolic_baseline/
  add the symbolic baseline without changing baseline_program()

src/crafter_benchmarks/__init__.py
  export public config/baseline additions

scripts/run_crafter_codex.py
  add observation selection and matching baseline/Skill selection

skills/optimize-crafter-local-symbolic-policy/SKILL.md
  add task-specific strategy guidance

pyproject.toml
  include the new Skill and baseline resources in wheel/sdist

README.md and collection documentation
  document IDs, schema, comparison limits, commands, and privacy boundary

tests/test_crafter_benchmark.py
tests/test_run_crafter_codex.py
  add unit, real-environment, artifact, packaging, and launcher coverage
```

No code in the EvoPolicyGym Kernel should be changed for this feature.

## 11. Required tests

At minimum, add the following tests.

### Configuration and specification

- RGB remains the default and preserves existing observation bytes and IDs.
- Invalid observation profile types and names fail at configuration time.
- Symbolic plus MP4 is rejected.
- Symbolic IDs, observation schema, metadata, and Environment parameters are
  exact.
- RGB and symbolic Environment digests differ.
- Every scoring class can construct both observation profiles.

### Real-environment observation

- Reset returns exactly the six public keys and does not advance Crafter.
- Terrain/entities are exact `uint8 (7, 9)` tensors in row-major order.
- The player is at `[3, 4]` after reset and every successful movement.
- The crop orientation agrees with left/right/up/down Actions.
- Blocked movement updates facing even when the player does not move.
- World-edge padding never leaks wrapped or opposite-edge cells.
- Underlying terrain remains present beneath an entity.
- Arrow direction and plant young/ripe variants use the correct IDs.
- Inventory values match the RGB HUD source state and remain exact ints in
  `[0, 9]`.
- Sleeping and daylight match the post-transition state.
- No observation contains global semantic shape, absolute position,
  achievements, hidden counters, seed, or Host data.

Use narrow local test doubles for forced world-edge, arrow, and ripe-plant
cases, but retain at least one real upstream reset/step test.

### Determinism and dynamics invariance

- `check_benchmark()` replays a symbolic fixture twice without issues.
- Repeated symbolic reset/Action sequences produce equal PolicyValue results.
- Parallel RGB and symbolic Environments using the same EpisodeSpec and fixed
  Actions have identical rewards, metrics, termination, and Feedback score.
- Symbolic extraction does not change upstream RNG state or later world
  evolution.
- A fresh Environment owns fresh Episode state and closes idempotently.

### Failure domain

- Every non-exact integer and every integer outside `0..16` raises
  `InvalidAction` without advancing upstream state.
- Malformed or drifted upstream material/object/inventory state is an
  Environment fault, not an `InvalidAction` or Policy penalty.
- A Policy failure stops before another Crafter step, as for RGB.

### Feedback and packaging

- Symbolic trajectory and NPZ artifacts reconstruct every live observation.
- Initial/result observation alignment and chunk boundaries are exact.
- Artifact fields, dtypes, inventory/facing order, manifest version, retention,
  and byte bounds are public and tested.
- Public bytes contain no `environment_seed`, `policy_seed`, `player_pos`, Host
  path, full-map array, or hidden counter name/value.
- Validation/test Feedback emits no detailed symbolic artifacts.
- The symbolic baseline completes direct Evaluation without Policy failure.
- The built wheel contains the symbolic baseline, Skill, typed package marker,
  and documentation required at runtime.
- The launcher chooses exactly one matching baseline and Skill.

## 12. Verification commands

Run from the repository root unless noted:

```console
uv sync --project environments/crafter/crafter --extra dev
uv run --project environments/crafter/crafter \
  python -m unittest discover -s environments/crafter/crafter/tests
uv run --project environments/crafter/crafter \
  ruff check environments/crafter/crafter/src \
             environments/crafter/crafter/tests \
             environments/crafter/crafter/scripts
uv run --project environments/crafter/crafter mypy
uv build environments/crafter/crafter
```

Also run one short real direct Evaluation for RGB and symbolic using the same
Episode seed and a fixed Action Program, then compare all non-observation
outcomes. `ProcessExecution.unsafe()` is only an acknowledgement of local
process authority; it is not a sandbox.

## 13. Acceptance criteria

The feature is complete only when:

1. existing RGB public defaults, observations, scoring, Feedback, tests, and
   baseline behavior remain supported;
2. the symbolic observation exactly matches sections 3 and 4;
3. symbolic extraction consumes no Action, step, RNG draw, or hidden episode;
4. fixed Actions prove dynamics and score invariance between RGB and symbolic;
5. no global semantic map, absolute coordinate, seed, or hidden simulator state
   appears in a Policy observation or public artifact;
6. the symbolic baseline directly evaluates successfully;
7. the independent distribution passes unittest, Ruff, strict mypy, build, and
   wheel-content verification; and
8. documentation clearly labels local symbolic as a separate non-canonical
   observation profile inspired by Craftax rather than an upstream Crafter or
   Craftax registration.

## 14. Implemented wire-order note

EvoPolicyGym's canonical Policy wire encoding sorts mapping keys. Therefore,
the six live observation fields and the 16 live inventory fields are defined
by exact key sets, not by Python dictionary insertion order. The Artifact
`inventory` tensor still uses the fixed order in section 3.4, which is repeated
in `artifact-manifest.json`. Policy code must access live inventory values by
name and use the manifest order only when decoding the dense NPZ array.
