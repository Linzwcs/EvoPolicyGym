"""Tarot card validation scenarios — all 22 tarots.

Each scenario adds a tarot to consumables, uses it, and compares results.

Authoritative source: balatro_source/game.lua (P_CENTERS, Tarot set).
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    add_both,
    compare_state,
    run_consumable_scenario,
    select_blind,
    set_both,
    start_both,
    use_consumable,
)

# ---------------------------------------------------------------------------
# Enhancement tarots — apply enhancement to highlighted cards
# (max_highlighted from source)
# ---------------------------------------------------------------------------

_ENHANCEMENT_TAROTS: list[tuple[str, str, str, int]] = [
    # (key, name, enhancement, max_highlighted)
    ("c_magician", "The Magician", "m_lucky", 2),
    ("c_empress", "The Empress", "m_mult", 2),
    ("c_heirophant", "The Hierophant", "m_bonus", 2),
    ("c_lovers", "The Lovers", "m_wild", 1),
    ("c_chariot", "The Chariot", "m_steel", 1),
    ("c_justice", "Justice", "m_glass", 1),
    ("c_devil", "The Devil", "m_gold", 1),
    ("c_tower", "The Tower", "m_stone", 1),
]

for _key, _name, _enh, _max_h in _ENHANCEMENT_TAROTS:

    def _make_fn(key: str = _key, max_h: int = _max_h):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_consumable_scenario(
                sim,
                live,
                consumable_key=key,
                targets=list(range(min(max_h, 5))),  # target first N hand cards
                delay=delay,
            )

        return fn

    register(
        name=f"tarot_{_key[2:]}",
        category="tarots",
        description=f"{_name}: apply {_enh} to up to {_max_h} cards",
    )(_make_fn(_key, _max_h))

# ---------------------------------------------------------------------------
# Suit change tarots — change suit of highlighted cards
# ---------------------------------------------------------------------------

_SUIT_TAROTS: list[tuple[str, str, str]] = [
    ("c_star", "The Star", "Diamonds"),
    ("c_moon", "The Moon", "Clubs"),
    ("c_sun", "The Sun", "Hearts"),
    ("c_world", "The World", "Spades"),
]

for _key, _name, _suit in _SUIT_TAROTS:

    def _make_fn(key: str = _key):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_consumable_scenario(
                sim,
                live,
                consumable_key=key,
                targets=[0, 1, 2],  # max_highlighted = 3 for suit tarots
                delay=delay,
            )

        return fn

    register(
        name=f"tarot_{_key[2:]}",
        category="tarots",
        description=f"{_name}: change up to 3 cards to {_suit}",
    )(_make_fn(_key))

# ---------------------------------------------------------------------------
# Transformation tarots
# ---------------------------------------------------------------------------


@register(
    name="tarot_strength",
    category="tarots",
    description="Strength: increase rank of up to 2 cards by 1",
)
def _tarot_strength(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(
        sim, live, consumable_key="c_strength", targets=[0, 1], delay=delay
    )


@register(
    name="tarot_death",
    category="tarots",
    description="Death: convert left card to copy of right card (2 cards)",
)
def _tarot_death(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_death", targets=[0, 1], delay=delay)


@register(
    name="tarot_hanged_man",
    category="tarots",
    description="The Hanged Man: destroy up to 2 selected cards",
)
def _tarot_hanged_man(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(
        sim, live, consumable_key="c_hanged_man", targets=[0, 1], delay=delay
    )


# ---------------------------------------------------------------------------
# Generation tarots
# ---------------------------------------------------------------------------


@register(
    name="tarot_fool",
    category="tarots",
    description="The Fool: creates copy of last Tarot/Planet used",
)
def _tarot_fool(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    # Use a planet first, then use The Fool to copy it
    start_both(sim, live, seed="T_FOOL", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="c_mercury")  # planet
    use_consumable(sim, live, 0, delay=delay)
    add_both(sim, live, key="c_fool")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="fool after copy")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Fool: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="tarot_high_priestess",
    category="tarots",
    description="The High Priestess: creates up to 2 random Planets",
)
def _tarot_high_priestess(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_high_priestess", delay=delay)


@register(
    name="tarot_emperor",
    category="tarots",
    description="The Emperor: creates up to 2 random Tarots",
)
def _tarot_emperor(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_emperor", delay=delay)


@register(
    name="tarot_judgement", category="tarots", description="Judgement: creates a random Joker"
)
def _tarot_judgement(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_judgement", delay=delay)


# ---------------------------------------------------------------------------
# Economy / edition tarots
# ---------------------------------------------------------------------------


@register(name="tarot_hermit", category="tarots", description="The Hermit: doubles money (max $20)")
def _tarot_hermit(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="T_HERMIT", delay=delay)
    select_blind(sim, live, delay=delay)
    set_both(sim, live, money=15)
    add_both(sim, live, key="c_hermit")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="hermit after use")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Hermit: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="tarot_temperance",
    category="tarots",
    description="Temperance: gain total sell value of jokers (max $50)",
)
def _tarot_temperance(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="T_TEMPERANCE", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")
    add_both(sim, live, key="c_temperance")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="temperance after use")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Temperance: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="tarot_wheel_of_fortune",
    category="tarots",
    description="Wheel of Fortune: 1 in 4 chance to add edition to random joker",
)
def _tarot_wheel(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="T_WHEEL", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")
    add_both(sim, live, key="c_wheel_of_fortune")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="wheel after use")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Wheel: {'PASS' if not diffs else 'FAIL'}"
    )
