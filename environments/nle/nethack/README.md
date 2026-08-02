# EvoPolicyGym NLE NetHack Benchmark

This independently installable distribution adapts NLE 1.3.0
`NetHackScore-v0` to the public EvoPolicyGym authoring SDK. NLE 1.3.0 is the
maintained successor of the archived `facebookresearch/nle` repository and is
based on NetHack 3.6.7.

```python
from nle_benchmarks import NetHackBenchmark, NetHackConfig

benchmark = NetHackBenchmark()
development = NetHackBenchmark(NetHackConfig(max_episode_steps=256))
```

The canonical Benchmark ID is
`nle/NetHackScore-v0/mean-return-v1`.

## Profile

- fixed neutral human male Monk (`mon-hum-neu-mal`);
- the upstream 23-action `TASK_ACTIONS` profile;
- public map, status, message, inventory, and input-mode observations;
- up to 5,000 Policy steps per Episode;
- score-delta reward with the upstream constant frozen-step penalty;
- mean Episode return as the primary metric.

`NetHackConfig(max_episode_steps=...)` selects a shorter bounded development
profile. The horizon is published in `environment_parameters`, so EvoPolicyGym
assigns a distinct Environment digest. Shortened results are not directly
comparable with the canonical profile.

## Determinism and privacy

Each Episode receives one split-scoped hidden root seed. The adapter derives
independent NLE core, display, and level-generation seeds, disables anti-TAS
reseeding, and derives time effects from the seed. Bones files, NLE rendering,
ttyrec recording, and saved game directories are disabled. Every Episode owns
one fresh NLE instance and closes it on every evaluator exit path.

The Policy never receives `EpisodeSpec`, any RNG seed, NLE's privileged
`internal` observation, Host paths, ttyrec data, or evaluator objects. It sees
only bounded `PolicyValue` data. Actions must be exact integers from 0 through
22; invalid Actions are rejected without advancing NLE.

## Scoring

The scalar score is mean shaped NLE return. A Policy failure receives
`-max_episode_steps`, preventing an invalid or crashing Policy from escaping
the frozen-step penalty. Feedback also reports game score, depth, steps,
deaths, ascensions, truncations, and Policy failures. It additionally reports
the exact score-component diagnostics `frozen_steps`, `mean_frozen_steps`,
`frozen_step_fraction`, and `mean_frozen_penalty`; these are aggregate numeric
feedback, not Environment-selected trajectory content.

NLE 1.3.0 exposes its own step-limit exit as `done=True`,
`end_status=ABORTED`, while its returned Gymnasium truncation flag is computed
before the final private step-counter increment. The adapter translates only
that horizon case to `terminated=False, truncated=True`. Natural deaths remain
terminated, including a death on the final allowed step. This normalization
does not change the upstream reward or Episode length.

Validation and Assessment use disjoint private Episode seed domains. They run
only after Agent cleanup and retain aggregate Host results; they neither build
nor publish detailed Artifacts. The private Episode scenario contains only a
Feedback-scope marker, which does not change NLE dynamics and is never visible
to the Policy or Agent.

## Complete raw training evidence

Every successful training submission publishes every Policy-visible initial
observation, every post-Action observation, and every transition from every
evaluated Episode. The Benchmark does not select Episodes or frames, sample
time, render an image, create a video, add an overlay, or maintain a separate
human-observer channel.

```text
artifacts/
├── artifact-manifest.json
└── bulk/
    ├── observations-000000.npz
    ├── observations-000001.npz
    └── episodes/
        ├── episode-000000/trajectory-000000.jsonl.gz
        └── episode-000001/trajectory-000000.jsonl.gz
```

Each deterministic gzip JSONL trajectory contains an Episode header followed
by every transition in order: public Episode/step/observation ordinals, Action
and Action name, reward, public metrics, and termination flags. It contains no
Environment seed, Policy seed, Host path, process evidence, or privileged NLE
state. Indices preserve
`observation[t] -> action[t] -> observation[t + 1]`.

Each NPZ contains at most 1,024 observations and must be opened with
`allow_pickle=False`. It stores the exact public tensors and a reversible fixed
width encoding of all other public fields:

- `glyphs`: `int16 [N,21,79]`;
- `chars`, `colors`: `uint8 [N,21,79]`;
- `stats`: `int64 [N,27]` in the manifest's named order;
- `message_bytes`, `message_lengths`;
- inventory counts, letters, descriptions and lengths, glyphs, and classes;
- input-mode codes;
- public Episode and observation ordinals.

Strings use Latin-1, matching the Policy projection. The permanent manifest
declares every array mapping, condition bit and input-mode code, lists every
chunk and byte size, and reports complete Episode/transition/observation
counts. The NPZ is source data, not an environment-produced visualization.

The public per-submission training batch limit is 64 Episodes; the hidden
training pool may be larger. At the 5,000-step horizon the
worst possible batch needs 313 observation chunks, 64 trajectories and one
manifest (378 Artifacts), below EvoPolicyGym's 1,024-Artifact limit. A 1,024
observation chunk has approximately 11.6 MiB of fixed uncompressed payload,
below the 16 MiB per-Artifact limit even before normal NLE compression.

## Agent-controlled analysis and retention

The raw NPZ and trajectory files use `retention="bulk"`; compact Feedback and
`artifact-manifest.json` are permanent. EvoPolicyGym applies the configured
capacity to both the Host record and Agent-visible mirror, evicts only older
bulk files oldest-submission-first, and always protects the newest complete
submission. The default capacity is 1 GiB and should be calibrated from real
optimized trajectories.

On 2026-08-02, the trusted packaged baseline was measured at a short 64-step
development horizon. Batches of 1, 4, 16, and 64 Episodes produced respectively
9,145, 26,591, 110,581, and 426,724 compressed bulk bytes; the 64-Episode run
contained 4,066 transitions and 4,130 observations and completed evaluation
plus encoding in about 4.3 seconds. This is only a compression and plumbing
smoke test. It is not representative of an optimized Policy surviving near
the canonical 5,000-step horizon.

The Agent owns `workspace/analysis/`, which is never submitted and is not
pruned by bulk retention. It decides which observations to decode, whether to
turn terminal arrays into images or other representations, which frames or
segments to inspect, and what derived scripts, images, summaries, or samples
to preserve there. Only `workspace/program/` becomes a submission. This keeps
visual interpretation and evidence selection inside Agent authority.

The runner enables Codex's `view_image` tool and verifies that the same uv
environment provides NumPy, Pillow, ImageIO, and imageio-ffmpeg. These are
Agent analysis tools; no Benchmark source imports them to create visual
Feedback. Install the complete environment with:

```console
uv sync --locked --extra dev --extra agent-tools
```

## Core16-style run

The runner defaults to `gpt-5.6-luna`, 32 submissions, a 1,024-Episode search
budget, at most 64 Episodes per submission, 64 Validation Episodes for up to
three candidates, and 256 held-out Assessment Episodes. The Benchmark skill is
disabled unless explicitly enabled.

```console
.venv/bin/python scripts/run_nle_codex.py \
  --record-to .local/runs/nle-core16-<run-id> \
  --allow-unsafe-process
```

The package also supplies an observation-aware exploration
`baseline_program()` and an optional `optimize-nethack-policy` Agent skill.
The runner prints Program digests and groups candidates whose complete
aggregate Validation feedback is exactly identical. This is a post-run audit
signal only: the Agent still chooses candidates, and the Host neither inspects
nor rewrites their behavior.

## Development

```console
uv lock --check
uv sync --locked --extra dev --extra agent-tools
uv run --locked --extra agent-tools ruff check src tests scripts
uv run --locked --extra agent-tools mypy
uv run --locked --extra agent-tools python -m unittest discover -s tests
uv build .
```

The direct Evaluation test runs only the trusted packaged baseline through
`ProcessExecution.unsafe()`. It is not a sandbox and executes with the current
operating-system user's authority. The Codex runner has the same limitation.

## Upstream and license

The adapter code is MIT licensed. NLE and NetHack are separate dependencies
distributed under the NetHack General Public License. This distribution does
not vendor NLE, NetHack source, ttyrec data, or game assets. The pinned NLE
release and its license define the supported simulator behavior.
