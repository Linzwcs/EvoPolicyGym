"""Spectral card validation scenarios — all 18 spectrals.

Each scenario adds a spectral, uses it, and compares effects.

Authoritative source: balatro_source/game.lua (P_CENTERS, Spectral set).
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
# Seal spectrals — apply seal to 1 highlighted card
# ---------------------------------------------------------------------------

_SEAL_SPECTRALS: list[tuple[str, str, str]] = [
    ("c_talisman", "Talisman", "Gold"),
    ("c_deja_vu", "Deja Vu", "Red"),
    ("c_trance", "Trance", "Blue"),
    ("c_medium", "Medium", "Purple"),
]

for _key, _name, _seal in _SEAL_SPECTRALS:

    def _make_fn(key: str = _key, name: str = _name):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_consumable_scenario(sim, live, consumable_key=key, targets=[0], delay=delay)

        return fn

    register(
        name=f"spectral_{_key[2:]}",
        category="spectrals",
        description=f"{_name}: apply {_seal} seal to 1 card",
    )(_make_fn(_key, _name))

# ---------------------------------------------------------------------------
# Creation / destruction spectrals
# ---------------------------------------------------------------------------


@register(
    name="spectral_familiar",
    category="spectrals",
    description="Familiar: destroy 1 random card, create 3 random Enhanced face cards",
)
def _spectral_familiar(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_familiar", targets=[0], delay=delay)


@register(
    name="spectral_grim",
    category="spectrals",
    description="Grim: destroy 1 random card, create 2 random Enhanced Aces",
)
def _spectral_grim(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_grim", targets=[0], delay=delay)


@register(
    name="spectral_incantation",
    category="spectrals",
    description="Incantation: destroy 1 random card, create 4 random Enhanced number cards",
)
def _spectral_incantation(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(
        sim, live, consumable_key="c_incantation", targets=[0], delay=delay
    )


@register(
    name="spectral_cryptid",
    category="spectrals",
    description="Cryptid: create 2 copies of 1 selected card",
)
def _spectral_cryptid(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_cryptid", targets=[0], delay=delay)


@register(
    name="spectral_immolate",
    category="spectrals",
    description="Immolate: destroy 5 random cards, gain $20",
)
def _spectral_immolate(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_immolate", delay=delay)


@register(
    name="spectral_sigil",
    category="spectrals",
    description="Sigil: convert all hand cards to a single random suit",
)
def _spectral_sigil(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_sigil", delay=delay)


@register(
    name="spectral_ouija",
    category="spectrals",
    description="Ouija: convert all hand cards to a single random rank, -1 hand size",
)
def _spectral_ouija(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_ouija", delay=delay)


# ---------------------------------------------------------------------------
# Edition spectrals
# ---------------------------------------------------------------------------


@register(
    name="spectral_ectoplasm",
    category="spectrals",
    description="Ectoplasm: add Negative to random joker, -1 hand size",
)
def _spectral_ectoplasm(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="S_ECTOPLASM", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")
    add_both(sim, live, key="c_ectoplasm")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="ectoplasm")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Ectoplasm: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="spectral_hex",
    category="spectrals",
    description="Hex: add Polychrome to random joker, destroy rest",
)
def _spectral_hex(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="S_HEX", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")
    add_both(sim, live, key="j_greedy_joker")
    add_both(sim, live, key="c_hex")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="hex")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Hex: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Joker creation spectrals
# ---------------------------------------------------------------------------


@register(
    name="spectral_wraith",
    category="spectrals",
    description="Wraith: create random Rare joker, set money to $0",
)
def _spectral_wraith(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="S_WRAITH", delay=delay)
    select_blind(sim, live, delay=delay)
    set_both(sim, live, money=10)
    add_both(sim, live, key="c_wraith")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="wraith")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Wraith: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="spectral_ankh",
    category="spectrals",
    description="Ankh: copy random joker, destroy all others",
)
def _spectral_ankh(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="S_ANKH", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker")
    add_both(sim, live, key="j_greedy_joker")
    add_both(sim, live, key="c_ankh")
    use_consumable(sim, live, 0, delay=delay)
    diffs = compare_state(sim, live, label="ankh")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Ankh: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="spectral_soul", category="spectrals", description="The Soul: create a Legendary joker"
)
def _spectral_soul(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_consumable_scenario(sim, live, consumable_key="c_soul", delay=delay)
