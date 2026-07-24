"""Blind chip requirement scaling.

Matches ``get_blind_amount`` in ``misc_functions.lua:919``.

The actual chip target for a blind is::

    target = get_blind_amount(ante, scaling) × blind_mult × ante_scaling

Where ``blind_mult`` is 1 for Small, 1.5 for Big, 2 for Boss, and
``ante_scaling`` is 1.0 normally or 2.0 for Plasma Deck.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Hardcoded base amounts for antes 1-8 (from misc_functions.lua:922-944)
# ---------------------------------------------------------------------------

_AMOUNTS: dict[int, list[int]] = {
    1: [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000],
    2: [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000],
    3: [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000],
}

# Blind type multipliers (from P_BLINDS: bl_small.mult, bl_big.mult, boss.mult)
BLIND_MULT: dict[str, float] = {
    "Small": 1.0,
    "Big": 1.5,
    "Boss": 2.0,
}


def get_blind_amount(ante: int, scaling: int = 1) -> int:
    """Return the base chip requirement for a given ante and scaling level.

    Args:
        ante: Current ante number (1-8 uses hardcoded tables, 9+ exponential).
        scaling: Difficulty scaling level from stake:
            1 = White/Red stake (default), 2 = Green-Blue, 3 = Purple-Gold.

    Returns:
        Base chip amount (before blind_mult and ante_scaling multipliers).
    """
    if ante < 1:
        return 100

    amounts = _AMOUNTS.get(scaling, _AMOUNTS[1])

    if ante <= 8:
        return amounts[ante - 1]  # 0-indexed

    # Exponential formula for antes 9+ (misc_functions.lua:927-929)
    k = 0.75
    a = amounts[7]  # ante 8 base amount
    b = 1.6
    c = ante - 8
    d = 1 + 0.2 * c

    amount = math.floor(a * (b + (k * c) ** d) ** c)

    # Significant-figure rounding (misc_functions.lua:929)
    # amount = amount - amount % (10 ^ floor(log10(amount) - 1))
    if amount > 0:
        digits = math.floor(math.log10(amount)) - 1
        rounding = 10**digits
        amount = amount - amount % rounding

    return amount


def get_blind_target(
    ante: int,
    blind_type: str,
    scaling: int = 1,
    ante_scaling: float = 1.0,
) -> int:
    """Return the actual chip target for a specific blind.

    Args:
        ante: Current ante number.
        blind_type: ``"Small"``, ``"Big"``, or ``"Boss"``.
        scaling: Stake scaling level (1/2/3).
        ante_scaling: Deck ante scaling (1.0 normally, 2.0 for Plasma).

    Returns:
        Chip target the player must reach to beat this blind.
    """
    base = get_blind_amount(ante, scaling)
    mult = BLIND_MULT.get(blind_type, 2.0)
    return math.floor(base * mult * ante_scaling)
