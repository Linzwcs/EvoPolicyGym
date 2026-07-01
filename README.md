# EvoPolicyGym

EvoPolicyGym is benchmark infrastructure for evaluating whether coding agents can
improve executable policies from budget-limited environment feedback. It follows
an online-judge style protocol: an agent edits code in a workspace, submits
policy rollouts through a local API, receives feedback artifacts, and continues
until the episode budget is exhausted.

The current implementation focuses on a local Python harness, reproducible run
artifacts, agent adapters, and Gymnasium-style environment integration.

## Status

EvoPolicyGym is alpha software. The active package is `src/evopolicygym`; frozen
v1 material lives under `archive/v1/` for reference only.

## Install

```bash
uv sync
```

For the main Core-16 experiment stack, install the Gymnasium and compatible
environment families:

```bash
uv sync --extra dev --extra env-gym --extra env-compatible
```

The same setup is wrapped by:

```bash
scripts/setup-env.sh --core
```

`--core` installs the dependencies needed by `config/main-128-*.toml`:
Gymnasium classic control, Box2D, MuJoCo, MiniGrid, HighwayEnv, and
Gymnasium-Robotics. The small smoke configs only need the base package:

```bash
scripts/setup-env.sh --smoke
```

Optional environment families are split into extras and should be installed
only when needed:

```bash
uv sync --extra env-visual
uv sync --extra env-multi
uv sync --extra env-web
uv sync --extra env-heavy
uv sync --extra env-jax
uv sync --extra env-mario
```

`env-jax` and `env-mario` are separate runtime targets. Gymnasium's JAX envs
need `numpy>=2.1`, while MO-Gymnasium's Mario extra currently pins
`numpy<2.0`, so they should be tested in separate virtual environments.

### Runtime Assets

Some optional environment families need non-Python assets:

- BrowserGym MiniWoB++: run `scripts/setup-env.sh --core --web`. This installs
  `env-web`, checks out `Farama-Foundation/miniwob-plusplus` under ignored
  `third_party/miniwob-plusplus` at commit
  `7fd85d71a4b60325c6585396ec4f48377d049838`, and installs Playwright
  Chromium. EvoPolicyGym auto-detects this path from the repository root; if
  running elsewhere, set `MINIWOB_URL` to the printed `file://.../miniwob/`
  URL.
- Atari/ALE: install Gymnasium assets with `scripts/setup-env.sh --core
  --atari-roms`, which runs AutoROM into the active `.venv`.
- MiniGrid WFC assets are vendored in
  `src/evopolicygym/envs/gym/assets/minigrid_wfc_patterns/`; no extra manual
  step is needed.

See [`docs/envs/overview.md`](docs/envs/overview.md) for the broader optional
environment roadmap.

## Quickstart

Run the unit suite:

```bash
uv run python -m unittest discover -s tests
```

Run the CLI help:

```bash
uv run evopolicygym --help
```

The old `feedbackgym` package and CLI names are not supported.

Create reproducible external case splits:

```bash
uv run evopolicygym data make \
  --env gym/taxi \
  --root data/gym/taxi \
  --seed 0 \
  --train-size 64 \
  --valid-size 64 \
  --heldout-size 256
```

Run a local command-agent benchmark session:

```bash
uv run evopolicygym run \
  --env toy \
  --runs runs \
  --model script \
  --exp-id smoke-001 \
  --budget 8 \
  --agent command -- python agent.py
```

Run from a TOML/JSON config:

```bash
uv run evopolicygym run --config docs/examples/cartpole-codex.toml
```

Run the small live-agent smoke suite:

```bash
uv run evopolicygym suite --config config/smoke-8-suite.toml
```

Run the 128-budget main suites:

```bash
uv run evopolicygym suite --config config/main-128-codex-suite.toml
uv run evopolicygym suite --config config/main-128-claude-suite.toml
uv run evopolicygym suite --config config/main-128-kimi-suite.toml
```

These configs require the corresponding CLI (`codex`, `claude`, or `kimi`) to
be authenticated locally. The Codex config uses `bypass = true` so the harness
can reach the local EvoPolicyGym HTTP API; server-side policy rollout remains
controlled by EvoPolicyGym.

## Repository Layout

- `src/evopolicygym/`: package source.
- `config/`: checked-in smoke and main experiment suite configs.
- `scripts/`: local setup helpers.
- `docs/protocol/`: normative protocol draft.
- `docs/envs/`: environment coverage notes, roadmap, and discovery output.
- `docs/examples/`: small checked-in example configs and fixtures.
- `tests/`: standard-library `unittest` suite.
- `third_party/`: ignored local runtime assets such as MiniWoB++ HTML.
- `archive/v1/`: frozen legacy code, docs, analysis, and v0 run data.

Generated run outputs belong under `runs/` or `experiment/` and are ignored by
default. Local generated case splits belong under ignored `data/`; checked-in
fixtures live under `docs/examples/data/`.

## Agent Adapters

EvoPolicyGym currently includes adapters for:

- generic persistent JSONL command agents;
- OpenAI Codex CLI;
- Claude Code;
- Kimi Code.

Each adapter preserves one logical agent session across benchmark turns while
the server controls rollout budget, feedback artifacts, hidden validation, and
final scoring.

## Environment Discovery

Check the structured EvoPolicyGym environment manifest and run smoke checks for
registered environments:

```bash
uv run evopolicygym check-envs
uv run evopolicygym check-envs --env gym/taxi
uv run evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1
uv run evopolicygym check-envs --discover --min-level L0
```

Regenerate the installed environment registry report with:

```bash
uv run evopolicygym discover-envs \
  --output docs/envs/discovered.json \
  --markdown docs/envs/env_list.md
```

The discovery report reflects installed optional packages; it is not a promise
that every discovered task already has a calibrated EvoPolicyGym scoring setup.

For Core-16 readiness checks, use:

```bash
uv run evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1 --timeout 60
```

## Safety

EvoPolicyGym executes agent-authored Python policies. Treat benchmark runs as
untrusted code execution. Use sandboxing, isolated workspaces, and disposable
credentials for live agent experiments. See [`SECURITY.md`](SECURITY.md).

## Citation

```bibtex
@software{evopolicygym2026,
  title  = {EvoPolicyGym},
  author = {Zhilin Wang and Han Song and Runzhe Zhan and Jusen Du and Jiacheng Chen and Tianle Li and Qingyu Yin and Yulun Wu and Zhennan Shen and Tong Zhu and Yanshu Li and Guanjie Chen and Derek F. Wong and Yafu Li and Yu Cheng and Yang Yang},
  year   = {2026},
  url    = {https://github.com/Linzwcs/EvoPolicyGym}
}
```

## License

EvoPolicyGym is released under the MIT License. See [`LICENSE`](LICENSE).
