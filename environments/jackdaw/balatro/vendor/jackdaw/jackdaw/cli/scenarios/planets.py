"""Planet card validation scenarios — all 13 planets.

Each scenario adds a planet, uses it, and verifies the hand level-up
matches between sim and live.

Authoritative source: balatro_source/game.lua (P_CENTERS, Planet set).
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    add_both,
    compare_state,
    play_hand,
    select_blind,
    start_both,
    use_consumable,
)

# ---------------------------------------------------------------------------
# All planets — each levels up a specific hand type
# ---------------------------------------------------------------------------

_PLANETS: list[tuple[str, str, str]] = [
    ("c_mercury", "Mercury", "Pair"),
    ("c_venus", "Venus", "Three of a Kind"),
    ("c_earth", "Earth", "Full House"),
    ("c_mars", "Mars", "Four of a Kind"),
    ("c_jupiter", "Jupiter", "Flush"),
    ("c_saturn", "Saturn", "Straight"),
    ("c_uranus", "Uranus", "Two Pair"),
    ("c_neptune", "Neptune", "Straight Flush"),
    ("c_pluto", "Pluto", "High Card"),
    ("c_planet_x", "Planet X", "Five of a Kind"),
    ("c_ceres", "Ceres", "Flush House"),
    ("c_eris", "Eris", "Flush Five"),
]

for _key, _name, _hand in _PLANETS:

    def _make_fn(key: str = _key, name: str = _name):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            seed = f"P_{key.upper()}"
            start_both(sim, live, seed=seed, delay=delay)
            select_blind(sim, live, delay=delay)
            add_both(sim, live, key=key)
            use_consumable(sim, live, 0, delay=delay)
            # Play a hand to verify leveled scoring
            play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
            diffs = compare_state(sim, live, label=f"after {name} use + play")
            return ScenarioResult(
                passed=not diffs,
                diffs=diffs,
                details=f"{name}: {'PASS' if not diffs else 'FAIL'}",
            )

        return fn

    register(
        name=f"planet_{_key[2:]}",
        category="planets",
        description=f"{_name}: level up {_hand}",
    )(_make_fn(_key, _name))


# ---------------------------------------------------------------------------
# Black Hole — levels up ALL hand types by 1
# ---------------------------------------------------------------------------


@register(
    name="planet_black_hole",
    category="planets",
    description="Black Hole: level up all hand types by 1",
)
def _planet_black_hole(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="P_BLACK_HOLE", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="c_black_hole")
    use_consumable(sim, live, 0, delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="after Black Hole use + play")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Black Hole: {'PASS' if not diffs else 'FAIL'}",
    )
