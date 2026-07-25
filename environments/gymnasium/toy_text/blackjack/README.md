# EvoPolicyGym Blackjack Benchmark

This independently installable distribution adapts Gymnasium `Blackjack-v1`
through EvoPolicyGym's public authoring SPI. Each Episode is one
infinite-deck Blackjack hand.

## Public interface

```python
from blackjack import (
    BlackjackBenchmark,
    BlackjackConfig,
    baseline_program,
)

benchmark = BlackjackBenchmark(
    BlackjackConfig(natural=False, sab=True)
)
```

The default matches the Gymnasium 1.3.0 `Blackjack-v1` registration.
`BlackjackConfig` publishes the `natural` and `sab` rule switches through
`BenchmarkSpec.environment_parameters`; they contribute to
`environment_digest` and are delivered to every Policy. When `sab=True`,
Gymnasium ignores the `natural` bonus and applies the Sutton-and-Barto natural
rule.

## Contract

The positional Gymnasium tuple is converted into a semantic observation:

```json
{
  "player_sum": 16,
  "dealer_showing": 10,
  "usable_ace": false
}
```

Action `0` sticks and Action `1` hits. Actions must be exact integers;
booleans, floats, structured values, and out-of-range integers raise
`InvalidAction`.

Cards are drawn with replacement from:

```text
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
```

Face cards each have value 10. The dealer draws until reaching at least 17.
A win returns `+1`, a draw `0`, and a loss `-1`. When `sab=False` and
`natural=True`, a winning natural pays `+1.5`.

The scalar Benchmark score is mean terminal reward. Blackjack has high
per-hand variance, so useful comparisons should evaluate many Episodes.
Policy failure receives `-1`, equal to the minimum complete hand return.

Gymnasium defines no TimeLimit for this environment. The Benchmark declares a
32-action safety horizon; valid hit/stick play terminates naturally well
before it.

## Feedback and trace

Feedback reports mean reward, mean actions per hand, wins, draws, losses,
Policy failures, and bounded trace coverage. `trace.jsonl` retains at most 32
complete hands with every semantic observation seen by the Policy, unmodified
Actions, rewards, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, dealer hole cards,
and private runtime evidence are never published.

`baseline_program()` always sticks immediately. It is intentionally weak and
leaves hand strategy to the coding agent.

## Development

From the repository root:

```console
cd environments/gymnasium/toy_text/blackjack
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
