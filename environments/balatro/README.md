# EvoPolicyGym Balatro Benchmark

This directory contains an unofficial, independently installable Balatro
Benchmark for EvoPolicyGym. It uses Jackdaw as the trusted headless rules
engine and exposes a semantic `PolicyValue` interface intended for Programs
written by coding agents.

The first profile is deliberately narrow:

- Red Deck (`b_red`);
- White Stake (`stake=1`);
- one complete run per Episode;
- deterministic, split-scoped hidden seeds;
- 1000 points for a win plus one point per Blind cleared;
- zero score for a Policy failure.

The `jackdaw-active-content-v1` content profile excludes objects whose effects
are inactive in the pinned engine. Exclusion happens while constructing the
RNG-backed pool, before selection:

- Tags: `tag_rare`, `tag_uncommon`, and `tag_voucher`;
- Vouchers: `v_omen_globe`, `v_telescope`, `v_observatory`,
  `v_directors_cut`, and `v_retcon`.

The partially implemented `v_illusion` remains available with a tooltip that
describes only its active behavior. Working prerequisites such as Crystal Ball
and Magic Trick also remain available. The exact exclusions are machine-readable
in `BenchmarkSpec.metadata.excluded_content`. Because these restrictions change
seed-to-content trajectories, this profile uses the `run-score-v2` Benchmark ID
and its results are not directly comparable with `run-score-v1`.

Jackdaw is maintained as vendored source under `vendor/jackdaw/`. Its base is
commit `c84dca9227b40eb5f7ff9fd7cd78945aa07854ce`, the immutable head of
upstream [PR #1](https://github.com/TylerFlar/jackdaw-balatro/pull/1). On top
of that base, this copy absorbs the reviewed gameplay and RNG fixes from
upstream [PR #2](https://github.com/TylerFlar/jackdaw-balatro/pull/2),
[PR #5](https://github.com/TylerFlar/jackdaw-balatro/pull/5), and
[PR #6](https://github.com/TylerFlar/jackdaw-balatro/pull/6). The exact commit
stack and known remaining limitations are recorded in
`vendor/jackdaw/EVOPOLICYGYM_VENDOR.md`.

## Policy interface

Observations are semantic dictionaries rather than Jackdaw's 235-dimensional
RL tensor. They contain the current phase, public run progress and resources,
the Blind, hand, Jokers, consumables, shop, pack, draw-pile composition, poker
hand levels, and a `legal_actions` description.

The Benchmark specification embeds the stable core manual in
`metadata.policy_guide`: the run loop, Benchmark reward versus game dollars,
phase transitions, exact Action shapes, observation semantics, scoring,
economy, card modifiers, and win/loss conditions. Coding Agents receive this
guide in their Run instruction without receiving an unseen card catalog.

The package also provides the `optimize-balatro-policy` authoring skill.
Program-Evolution Runs may opt in with
`RunConfig(use_benchmark_skill=True)` or the Balatro script's
`--benchmark-skill` flag. Enabled Runs expose it read-only as
`workspace/skill/SKILL.md`. It guides evidence allocation, strategy
development, and Policy hardening without adding hidden game state or entering
the Policy process.

Every currently visible Joker, Enhancement, Tarot, Planet, Spectral card,
Voucher, Booster, Blind, and skip Tag carries a version-pinned `rule` object
derived from the implemented Jackdaw behavior. Visible Edition and Seal names
refer to the exact definitions in the core guide:

```json
{
  "key": "j_jolly",
  "name": "Jolly Joker",
  "rule": {
    "summary": "Jolly Joker: +conditional Mult if hand contains Pair.",
    "parameters": {"effect": "Type Mult", "t_mult": 8, "type": "Pair"},
    "rarity": {"level": 1, "name": "Common"}
  }
}
```

Jokers whose tooltip changes with the round, such as Ancient Joker, Castle,
The Idol, and Mail-In Rebate, additionally expose the currently visible target
under `rule.visible_state`.

When a Blind is cleared, `round_earnings` exposes the human-visible cash-out
breakdown: `blind_dollar_reward`, unused-hand and discard bonuses, Joker
dollars, interest, rental cost, and `total_dollars`. The transition's
top-level `reward` remains the separate Benchmark reward.

A minimal decision looks like:

```python
if observation["phase"] == "blind_select":
    return {"kind": "select_blind"}

if observation["phase"] == "selecting_hand":
    return {
        "kind": "play_hand",
        "card_indices": [0, 2, 4],
    }
```

Actions are strict tagged objects. Unknown or missing fields, invalid entity
indices, duplicate card indices, illegal phases, and invalid consumable targets
raise `InvalidAction`; Actions are never repaired. Card selection order is
preserved because it can affect scoring.

## Feedback and replay

Feedback reports the objective score, win rate, mean Ante reached, mean Blinds
cleared, Policy failures, and the pinned engine revision. It deliberately does
not publish derived strategy diagnostics, action summaries, or automatic
best/worst analysis.

`replay.jsonl` retains complete semantic replays in Episode order within a
15 MiB public-artifact budget. Aggregate Feedback and Episode summaries still
cover every requested Episode; `replay_episodes` and
`replay_episodes_omitted` report detailed replay coverage. Every retained
initial state and transition state is the complete observation that the Policy
actually received, including `deck`, `poker_hands`, owned `vouchers`, awarded
`tags`, and `legal_actions`. This lets a Policy-authoring Agent reproduce its
decision context without receiving Environment or Policy seeds. A visual
player may render only the fields it needs; that presentation choice does not
reduce the artifact.

Each transition's top-level `reward` is the Benchmark reward. A Blind's
spendable in-game payout is a separate `blind.dollar_reward` value.

## Development

From this directory:

```console
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` in the Evaluation test is explicitly unsafe and provides no
isolation. Its test Program is a trusted package fixture.

## Scope and attribution

This project is not affiliated with LocalThunk, Playstack, or the official
Balatro project. It includes no official card art, sprites, music, fonts, or
other game assets.

The rules engine is
[TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro),
licensed under MIT. Jackdaw describes itself as an alpha-quality 1:1 Python
reimplementation and supplies live-game validation tooling, but this Benchmark
does not claim exhaustive equivalence with every official Balatro version.
The vendored revision, our deterministic replay checks, and future live
conformance results together define the supported profile.
