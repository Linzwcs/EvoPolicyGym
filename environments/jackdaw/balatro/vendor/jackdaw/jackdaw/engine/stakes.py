"""Cumulative stake modifier system.

Ports the stake-effect loop from ``game.lua:2049-2059``.  Stakes are
additive — stake 8 includes all effects from stakes 1–7.

Source references
-----------------
- game.lua:2049  — stake modifier application loop
- game.lua:2063  — ``enable_eternals_in_shop`` flag
- game.lua:2067  — ``enable_perishables_in_shop`` flag
- game.lua:2071  — ``enable_rentals_in_shop`` flag

Stake levels
------------

+-------+-------+-------+----------------------------------------------+
| Level | Key   | Name  | Effect                                       |
+=======+=======+=======+==============================================+
|   1   | white | White | No modifier (base difficulty)                |
+-------+-------+-------+----------------------------------------------+
|   2   | red   | Red   | Small Blind gives no money reward            |
+-------+-------+-------+----------------------------------------------+
|   3   | green | Green | Scaling level 2 (ante chip targets increase) |
+-------+-------+-------+----------------------------------------------+
|   4   | black | Black | Eternal jokers can appear in shop (30%)      |
+-------+-------+-------+----------------------------------------------+
|   5   | blue  | Blue  | −1 starting discard                          |
+-------+-------+-------+----------------------------------------------+
|   6   | purple| Purple| Scaling level 3 (overrides Green's level 2)  |
+-------+-------+-------+----------------------------------------------+
|   7   | orange| Orange| Perishable jokers can appear in shop (30%)   |
+-------+-------+-------+----------------------------------------------+
|   8   | gold  | Gold  | Rental jokers can appear in shop (30%)       |
+-------+-------+-------+----------------------------------------------+
"""

from __future__ import annotations

from typing import Any

# Default starting parameters (White Stake / base run).
# These match the values in ``G.GAME.starting_params`` at game start.
DEFAULT_STARTING_PARAMS: dict[str, int] = {
    "hands": 4,
    "discards": 3,
    "hand_size": 8,
    "joker_slots": 5,
    "consumable_slots": 2,
}


def apply_stake_modifiers(stake: int, game_state: dict[str, Any]) -> None:
    """Apply all stake effects cumulatively to *game_state* in-place.

    Stakes are additive: stake 8 includes all effects from levels 1–7.
    Mirrors the stake-modifier application loop in ``game.lua:2049-2059``.

    Parameters
    ----------
    stake:
        Integer stake level (1–8).  Level 1 (White) applies no modifiers.
    game_state:
        Mutable game-state dict.  This function reads and mutates two
        sub-dicts:

        ``modifiers`` (created if absent):
            Boolean / integer flags that change shop or scoring behaviour.

            * ``no_blind_reward`` (dict) — blind names with no payout
            * ``scaling`` (int) — ante chip-target scaling level
            * ``enable_eternals_in_shop`` (bool)
            * ``enable_perishables_in_shop`` (bool)
            * ``enable_rentals_in_shop`` (bool)

        ``starting_params`` (must exist; not created by this function):
            Per-run integer counters that may be decremented.

            * ``discards`` — decremented by 1 at stake ≥ 5 (Blue Stake)
    """
    if stake < 2:
        return

    modifiers: dict[str, Any] = game_state.setdefault("modifiers", {})

    # -- Stake 2: Red Chip — Small Blind yields no money reward --
    blind_rewards: dict[str, bool] = modifiers.setdefault("no_blind_reward", {})
    blind_rewards["Small"] = True

    # -- Stake 3: Green Chip — scaling level 2 --
    if stake >= 3:
        modifiers["scaling"] = 2

    # -- Stake 4: Black Chip — Eternal jokers in shop --
    if stake >= 4:
        modifiers["enable_eternals_in_shop"] = True

    # -- Stake 5: Blue Chip — lose one starting discard --
    if stake >= 5:
        game_state["starting_params"]["discards"] -= 1

    # -- Stake 6: Purple Chip — scaling level 3 (overrides Green's 2) --
    if stake >= 6:
        modifiers["scaling"] = 3

    # -- Stake 7: Orange Chip — Perishable jokers in shop --
    if stake >= 7:
        modifiers["enable_perishables_in_shop"] = True

    # -- Stake 8: Gold Chip — Rental jokers in shop --
    if stake >= 8:
        modifiers["enable_rentals_in_shop"] = True
