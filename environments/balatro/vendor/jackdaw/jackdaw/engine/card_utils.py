"""Card utility functions for shop and pack generation.

Ports ``poll_edition`` from ``common_events.lua:2055`` as a pure function.

Source references
-----------------
- common_events.lua:2055 — ``poll_edition``
- card.lua:369            — ``Card:set_cost`` (implemented on Card itself)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jackdaw.engine.rng import PseudoRandom


# ---------------------------------------------------------------------------
# poll_edition — common_events.lua:2055
# ---------------------------------------------------------------------------


def poll_edition(
    key: str,
    rng: PseudoRandom,
    rate: float = 1.0,
    mod: float = 1.0,
    *,
    no_neg: bool = False,
    guaranteed: bool = False,
) -> dict | None:
    """Roll for a card edition, matching ``poll_edition`` (common_events.lua:2055).

    A single random value is drawn and compared top-down against thresholds;
    the first match wins.  Normal-mode thresholds (rate=1, mod=1):

    +-------------+---------------------------+------------------+
    | Edition     | Threshold (roll >)        | Base chance      |
    +=============+===========================+==================+
    | Negative    | 1 − 0.003 × mod           | 0.3 %            |
    +-------------+---------------------------+------------------+
    | Polychrome  | 1 − 0.006 × rate × mod    | 0.3 %            |
    +-------------+---------------------------+------------------+
    | Holo        | 1 − 0.02  × rate × mod    | 1.4 %            |
    +-------------+---------------------------+------------------+
    | Foil        | 1 − 0.04  × rate × mod    | 2.0 %            |
    +-------------+---------------------------+------------------+
    | None        | everything else           | 96.0 %           |
    +-------------+---------------------------+------------------+

    Negative ignores *rate* — only *mod* affects its threshold.
    *rate* mirrors ``G.GAME.edition_rate``: 1 base, ×2 with Hone, ×2 again
    with Glow Up.

    Guaranteed mode (Wheel of Fortune, Aura, etc.) overrides both *rate* and
    *mod* with ``rate=1, mod=25``, giving: Foil 50 %, Holo 35 %,
    Polychrome 7.5 %, Negative 7.5 %.

    Args:
        key: RNG stream key (advances the named stream once via ``rng.random``).
        rng: :class:`~jackdaw.engine.rng.PseudoRandom` instance.
        rate: ``G.GAME.edition_rate`` — scales Foil/Holo/Poly thresholds.
        mod: Additional modifier (normally 1.0).
        no_neg: If True, the Negative edition is excluded from the draw
            (used by Aura and Wheel of Fortune).
        guaranteed: If True, use ×25 multiplier (sets rate=1, mod=25).
            Equivalent to the Lua ``_guaranteed`` flag.

    Returns:
        Edition dict (e.g. ``{"foil": True}``) or ``None`` for no edition.
    """
    if guaranteed:
        rate = 1.0
        mod = 25.0

    roll = rng.random(key)

    if not no_neg and roll > 1 - 0.003 * mod:
        return {"negative": True}
    if roll > 1 - 0.006 * rate * mod:
        return {"polychrome": True}
    if roll > 1 - 0.02 * rate * mod:
        return {"holo": True}
    if roll > 1 - 0.04 * rate * mod:
        return {"foil": True}
    return None
