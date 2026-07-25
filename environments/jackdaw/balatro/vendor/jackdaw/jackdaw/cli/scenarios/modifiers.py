"""Card modifier validation scenarios — enhancements, editions, seals, combinations.

Tests that card modifiers (enhancements, editions, seals) produce identical
scoring results between sim and live.

Authoritative source: balatro_source/game.lua (P_CENTERS, Enhanced/Edition sets)
and balatro_source/card.lua (Card:eval_card).

Enhancement values (from game.lua):
  m_bonus:  +30 chips when scored
  m_mult:   +4 mult when scored
  m_wild:   counts as any suit
  m_glass:  x2 mult, 1 in 4 chance to destroy
  m_steel:  x1.5 mult while in hand (not played)
  m_stone:  +50 chips, no rank/suit, always scores
  m_gold:   +$3 at end of round
  m_lucky:  1 in 5 for +20 mult, 1 in 15 for +$20

Edition values (from game.lua):
  e_foil:        +50 chips
  e_holo:        +10 mult
  e_polychrome:  x1.5 mult
  e_negative:    +1 joker slot

Seal effects:
  Red:    retrigger card once
  Blue:   creates planet card if held at end of round
  Gold:   +$3 when held at end of round
  Purple: creates tarot when discarded
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    add_both,
    compare_state,
    discard,
    get_hand_count,
    play_hand,
    run_modifier_scenario,
    select_blind,
    start_both,
)

# ---------------------------------------------------------------------------
# Enhancement scenarios
# ---------------------------------------------------------------------------

_ENHANCEMENTS: list[tuple[str, str]] = [
    ("m_bonus", "+30 chips when scored"),
    ("m_mult", "+4 mult when scored"),
    ("m_wild", "counts as any suit"),
    ("m_glass", "x2 mult, 1/4 chance to destroy"),
    ("m_steel", "x1.5 mult while held (not played)"),
    ("m_stone", "+50 chips, always scores, no rank/suit"),
    ("m_gold", "+$3 at end of round when held"),
    ("m_lucky", "1/5 for +20 mult, 1/15 for +$20"),
]

for _enh, _desc in _ENHANCEMENTS:

    def _make_fn(enh: str = _enh):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_modifier_scenario(sim, live, card_key="H_A", enhancement=enh, delay=delay)

        return fn

    register(
        name=f"modifier_{_enh[2:]}",
        category="modifiers",
        description=f"Enhancement {_enh}: {_desc}",
    )(_make_fn(_enh))

# ---------------------------------------------------------------------------
# Edition scenarios
# ---------------------------------------------------------------------------

_EDITIONS: list[tuple[str, str]] = [
    ("foil", "+50 chips"),
    ("holo", "+10 mult"),
    ("polychrome", "x1.5 mult"),
]

for _ed, _desc in _EDITIONS:

    def _make_fn(ed: str = _ed):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_modifier_scenario(sim, live, card_key="H_A", edition=ed, delay=delay)

        return fn

    register(
        name=f"modifier_{_ed}",
        category="modifiers",
        description=f"Edition {_ed}: {_desc}",
    )(_make_fn(_ed))


# Negative edition on joker (adds joker slot)
@register(
    name="modifier_negative_joker",
    category="modifiers",
    description="Negative edition on joker: +1 joker slot",
)
def _modifier_negative(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="M_NEGATIVE", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_joker", edition="negative")
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="negative joker")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Negative joker: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Seal scenarios
# ---------------------------------------------------------------------------


@register(
    name="modifier_red_seal",
    category="modifiers",
    description="Red seal: retrigger card once when scored",
)
def _modifier_red_seal(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(sim, live, card_key="H_A", seal="Red", delay=delay)


@register(
    name="modifier_gold_seal",
    category="modifiers",
    description="Gold seal: +$3 when held at end of round",
)
def _modifier_gold_seal(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="M_GOLD_SEAL", delay=delay)
    select_blind(sim, live, delay=delay)
    # Add a gold sealed card — don't play it so it's held at end of round
    add_both(sim, live, key="H_A", seal="Gold")
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)  # play without the gold card
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="gold seal end of round")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Gold seal: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="modifier_blue_seal",
    category="modifiers",
    description="Blue seal: creates planet if held at end of round",
)
def _modifier_blue_seal(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="M_BLUE_SEAL", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="H_A", seal="Blue")
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="blue seal end of round")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Blue seal: {'PASS' if not diffs else 'FAIL'}"
    )


@register(
    name="modifier_purple_seal",
    category="modifiers",
    description="Purple seal: creates tarot when discarded",
)
def _modifier_purple_seal(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="M_PURPLE_SEAL", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="H_A", seal="Purple")
    # Discard the purple-sealed card to trigger tarot creation.
    # It was appended to the end of hand, so discard the last card.
    last_idx = get_hand_count(sim) - 1
    discard(sim, live, [last_idx], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="purple seal discard")
    return ScenarioResult(
        passed=not diffs, diffs=diffs, details=f"Purple seal: {'PASS' if not diffs else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Combination scenarios
# ---------------------------------------------------------------------------


@register(
    name="modifier_glass_polychrome",
    category="modifiers",
    description="Glass + Polychrome: x2 * x1.5 = x3 mult",
)
def _modifier_glass_poly(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_glass", edition="polychrome", delay=delay
    )


@register(
    name="modifier_steel_holo",
    category="modifiers",
    description="Steel + Holo: x1.5 mult + 10 mult while held",
)
def _modifier_steel_holo(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_steel", edition="holo", delay=delay
    )


@register(
    name="modifier_stone_foil",
    category="modifiers",
    description="Stone + Foil: +50 chips + 50 chips",
)
def _modifier_stone_foil(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_stone", edition="foil", delay=delay
    )


@register(
    name="modifier_lucky_red_seal",
    category="modifiers",
    description="Lucky + Red seal: retrigger lucky roll",
)
def _modifier_lucky_red(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_lucky", seal="Red", delay=delay
    )


@register(
    name="modifier_bonus_gold_seal",
    category="modifiers",
    description="Bonus + Gold seal: +30 chips scored + $3 held",
)
def _modifier_bonus_gold(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_bonus", seal="Gold", delay=delay
    )


@register(
    name="modifier_mult_foil", category="modifiers", description="Mult + Foil: +4 mult + 50 chips"
)
def _modifier_mult_foil(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_modifier_scenario(
        sim, live, card_key="H_A", enhancement="m_mult", edition="foil", delay=delay
    )
