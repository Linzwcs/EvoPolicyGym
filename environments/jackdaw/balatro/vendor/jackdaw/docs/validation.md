# Validation Scenarios

Jackdaw validates its engine by running identical actions on both the Python simulator and live Balatro, then comparing the resulting state. When the sim and live diverge, the scenario fails and reports exactly which fields differ.

This is the primary tool for finding and fixing bugs in the engine.

## Prerequisites

- Balatro installed and running
- [BalatroBot](https://github.com/coder/balatrobot) mod installed and listening (default: `127.0.0.1:12346`)

## Running Scenarios

```bash
jackdaw validate                          # all ~250 scenarios
jackdaw validate --category jokers        # all 150 joker scenarios
jackdaw validate --category tarots        # all 22 tarot scenarios
jackdaw validate --category planets       # all 13 planet scenarios
jackdaw validate --category spectrals     # all 18 spectral scenarios
jackdaw validate --category boss_blinds   # all 28 boss blind scenarios
jackdaw validate --category modifiers     # all 20 modifier scenarios
jackdaw validate --scenario joker_jolly   # single scenario by name
jackdaw validate --host 127.0.0.1 --port 12346  # custom connection
```

## How It Works

Each scenario:

1. Creates a fresh `SimBackend` (in-process engine) and connects to a `LiveBackend` (BalatroBot)
2. Starts identical runs on both with the same seed, deck, and stake
3. Uses `add`/`set` debug commands to inject specific game state
4. Executes the same actions on both
5. Calls `compare_state()` to diff the results

## Scenario Files

All scenarios live in `jackdaw/cli/scenarios/`:

| File | Category | What it tests |
|------|----------|---------------|
| `jokers.py` | jokers | 150+ joker effects |
| `boss_blinds.py` | boss_blinds | Boss blind effects |
| `tarots.py` | tarots | Tarot consumables |
| `modifiers.py` | modifiers | Enhancements, editions, seals |
| `spectrals.py` | spectrals | Spectral cards |
| `planets.py` | planets | Planet cards |
| `tags.py` | tags | Tag effects |
| `helpers.py` | — | Shared utilities and runner patterns |
| `__init__.py` | — | Registry, `Scenario` and `ScenarioResult` types |

## Writing a Scenario

### Minimal Example

```python
from jackdaw.cli.scenarios import register, ScenarioResult
from jackdaw.cli.scenarios.helpers import (
    Handle,
    start_both,
    select_blind,
    add_both,
    play_hand,
    compare_state,
)

@register(
    name="my_scenario",
    category="jokers",
    description="Test that my joker does the right thing",
)
def _my_scenario(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    # 1. Start identical runs on sim + live
    start_both(sim, live, seed="MY_SEED", delay=delay)

    # 2. Get to the right game phase
    select_blind(sim, live, delay=delay)

    # 3. Inject the state you want to test
    add_both(sim, live, key="j_jolly")

    # 4. Execute an action
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    # 5. Compare sim vs live
    diffs = compare_state(sim, live, label="after play with j_jolly")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"My scenario: {'PASS' if not diffs else 'FAIL'}",
    )
```

If you're adding a scenario to a new file, register the import in `__init__.py`'s `get_all_scenarios()`.

### Run it

```bash
jackdaw validate --scenario my_scenario
```

## Using Runner Patterns

Most scenarios follow a standard pattern. Instead of writing the full flow yourself, use a runner:

### Joker (passive, triggers on any hand)

```python
from jackdaw.cli.scenarios.helpers import run_joker_scenario

@register(name="joker_joker", category="jokers", description="+4 Mult")
def _joker_joker(sim, live, *, delay=0.3):
    return run_joker_scenario(sim, live, joker_key="j_joker", seed="J_JOKER", delay=delay)
```

This starts a game, selects blind, adds the joker, plays cards `[0,1,2,3,4]`, and compares.

### Joker (needs a specific hand)

```python
from jackdaw.cli.scenarios.helpers import run_joker_with_setup

@register(name="joker_jolly", category="jokers", description="+8 Mult if Pair")
def _joker_jolly(sim, live, *, delay=0.3):
    return run_joker_with_setup(
        sim, live,
        joker_key="j_jolly",
        hand_preset="PAIR",       # injects cards that form a pair
        delay=delay,
    )
```

Available hand presets (defined in `helpers.py`):

| Preset | Hand triggered |
|--------|---------------|
| `PAIR` | Pair |
| `TWO_PAIR` | Two Pair |
| `THREE_KIND` | Three of a Kind |
| `FOUR_KIND` | Four of a Kind |
| `FULL_HOUSE` | Full House |
| `STRAIGHT` | Straight |
| `FLUSH_HEARTS` | Flush (Hearts) |
| `FLUSH_DIAMONDS` | Flush (Diamonds) |
| `FLUSH_CLUBS` | Flush (Clubs) |
| `FLUSH_SPADES` | Flush (Spades) |
| `STRAIGHT_FLUSH` | Straight Flush |
| `FACE_CARDS` | Face cards (J, Q, K) |
| `EVEN_RANKS` | Even-ranked cards |
| `ODD_RANKS` | Odd-ranked cards |
| `FIBONACCI` | Fibonacci ranks (A, 2, 3, 5, 8) |
| `HIGH_CARD` | High Card (no pairs/straights/flushes) |

You can also pass a list of card keys directly instead of a preset name:

```python
run_joker_with_setup(sim, live, joker_key="j_hack", hand_preset=["H_2", "D_3", "S_4", "C_5", "H_2"])
```

### Consumable

```python
from jackdaw.cli.scenarios.helpers import run_consumable_scenario

@register(name="tarot_fool", category="tarots", description="The Fool")
def _tarot_fool(sim, live, *, delay=0.3):
    return run_consumable_scenario(
        sim, live,
        consumable_key="c_fool",
        targets=[0, 1],              # card indices to target (or None)
        delay=delay,
    )
```

### Card Modifier

```python
from jackdaw.cli.scenarios.helpers import run_modifier_scenario

@register(name="mod_glass_foil", category="modifiers", description="Glass + Foil Ace")
def _mod_glass_foil(sim, live, *, delay=0.3):
    return run_modifier_scenario(
        sim, live,
        card_key="H_A",
        enhancement="m_glass",
        edition="foil",
        seal="Red",
        delay=delay,
    )
```

## Debug API Reference

### `add` — inject cards

```python
add_both(sim, live,
    key="j_jolly",              # required: card key
    edition="foil",             # optional: foil, holo, polychrome, negative
    enhancement="m_glass",      # optional: m_bonus, m_mult, m_wild, m_glass, m_steel, m_stone, m_gold, m_lucky
    seal="Red",                 # optional: Red, Blue, Gold, Purple
    eternal=True,               # optional: cannot be sold/destroyed
    perishable=True,            # optional: destroyed at end of round
    rental=True,                # optional: costs $1 per round
)
```

Card key naming:
- **Jokers**: `j_joker`, `j_jolly`, `j_zany`, ...
- **Tarots**: `c_fool`, `c_magician`, `c_high_priestess`, ...
- **Planets**: `c_mercury`, `c_venus`, `c_earth`, ...
- **Spectrals**: `c_familiar`, `c_grim`, `c_incantation`, ...
- **Playing cards**: `{suit}_{rank}` — `H_A`, `S_T`, `D_K`, `C_2`, ...
  - Suits: `H` (Hearts), `D` (Diamonds), `C` (Clubs), `S` (Spades)
  - Ranks: `2`-`9`, `T` (10), `J`, `Q`, `K`, `A`

### `set` — modify game state

```python
set_both(sim, live,
    money=100,        # dollars
    hands=10,         # hands remaining in round
    discards=7,       # discards remaining
    ante=5,           # current ante
    round=3,          # current round
    chips=500,        # chips scored this round
    shop=True,        # repopulate shop
)
```

## Helper Functions

All helpers operate on both backends simultaneously:

### Setup

| Function | What it does |
|----------|-------------|
| `start_both(sim, live, seed, deck, stake, delay)` | Start identical runs on both |
| `select_blind(sim, live, delay)` | Select the current blind |
| `skip_blind(sim, live, delay)` | Skip the current blind |
| `add_both(sim, live, **params)` | Inject a card on both |
| `set_both(sim, live, **params)` | Modify game state on both |

### Actions

| Function | What it does |
|----------|-------------|
| `play_hand(sim, live, cards, delay)` | Play cards by index |
| `discard(sim, live, cards, delay)` | Discard cards by index |
| `use_consumable(sim, live, consumable, cards, delay)` | Use a consumable |
| `buy_card(sim, live, index, delay)` | Buy a shop card |
| `sell_joker(sim, live, index, delay)` | Sell a joker |
| `sell_consumable(sim, live, index, delay)` | Sell a consumable |
| `cash_out(sim, live, delay)` | Cash out after beating a blind |
| `next_round(sim, live, delay)` | Leave shop, advance to next round |
| `reroll_shop(sim, live, delay)` | Reroll shop offerings |

### Flow Control

| Function | What it does |
|----------|-------------|
| `play_through_blind(sim, live, max_hands)` | Play hands until blind is beaten or game ends |
| `advance_past_blind(sim, live, delay)` | Select blind, play through, cash out, next round |
| `get_state(handle)` | Get current game phase string from one backend |
| `get_hand_count(handle)` | Get number of cards in hand from one backend |

### Comparison

```python
diffs = compare_state(sim, live, label="after play", check_round=True, check_shop=False)
```

Returns a list of diff strings. Compares: game phase, money, ante, chips, hands left, discards left, hand cards, deck size, jokers (ordered), consumables.

## Multi-Step Scenarios

For scenarios that need more than a single action, just chain helpers:

```python
@register(name="complex_example", category="jokers", description="Multi-step test")
def _complex_example(sim, live, *, delay=0.3):
    start_both(sim, live, seed="COMPLEX", delay=delay)

    # Play through the small blind first
    select_blind(sim, live, delay=delay)
    play_through_blind(sim, live, delay=delay)
    cash_out(sim, live, delay=delay)
    next_round(sim, live, delay=delay)

    # Now test our joker on the big blind
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_blueprint")
    add_both(sim, live, key="j_jolly")
    set_both(sim, live, money=50)

    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)

    diffs = compare_state(sim, live, label="after big blind play")
    return ScenarioResult(passed=not diffs, diffs=diffs, details="Complex example")
```

## Grouped Sub-Results

Use `sub_results` to report multiple checks within one scenario:

```python
@register(name="multi_check", category="jokers", description="Multiple plays")
def _multi_check(sim, live, *, delay=0.3):
    start_both(sim, live, seed="MULTI", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")

    all_subs = []

    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs1 = compare_state(sim, live, label="hand 1")
    all_subs.append(("hand 1", ScenarioResult(passed=not diffs1, diffs=diffs1)))

    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs2 = compare_state(sim, live, label="hand 2")
    all_subs.append(("hand 2", ScenarioResult(passed=not diffs2, diffs=diffs2)))

    all_passed = all(r.passed for _, r in all_subs)
    return ScenarioResult(passed=all_passed, sub_results=all_subs, details="Multi check")
```
