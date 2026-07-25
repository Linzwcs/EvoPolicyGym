"""End-of-round money calculation.

Ports ``evaluate_round`` (state_events.lua:1135) and rental deduction
(state_events.lua:108) as pure functions.

**Order of operations** (matching Lua source):

1. Rental deduction (state_events.lua:108) — each rental joker costs
   ``rental_rate`` ($3).  Fires *before* the interest calculation so that
   interest is computed on the *post-rental* balance.
2. Blind reward — ``blind.dollars`` (0 when ``no_blind_reward`` stake).
3. Unused hands bonus — ``hands_left × (money_per_hand or 1)``.
   Skipped when ``modifiers['no_extra_hand_money']`` is set.
4. Unused discards bonus — ``discards_left × money_per_discard``.
   Only present when ``modifiers['money_per_discard']`` is set (Green Deck).
5. Joker dollar bonuses — via :func:`on_end_of_round` (``calc_dollar_bonus``).
6. Interest — ``interest_amount × min(effective_money // 5, interest_cap // 5)``
   where ``effective_money = money − rental_cost``.
   Skipped when ``modifiers['no_interest']`` is set or ``effective_money < 5``.

Source references
-----------------
- state_events.lua:96-110  — rental deduction, end-of-round joker evals
- state_events.lua:1135-1208 — evaluate_round (blind + hands + discards
  + joker bonuses + interest)
- state_events.lua:433 — discard cost modifier (Golden Needle challenge)
- game.lua:1909-1915 — ``interest_cap=25``, ``interest_amount=1``,
  ``rental_rate=3`` defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom

from jackdaw.engine.jokers import GameSnapshot, on_end_of_round

# ---------------------------------------------------------------------------
# RoundEarnings — per-round cash-out descriptor
# ---------------------------------------------------------------------------


@dataclass
class RoundEarnings:
    """Breakdown of money earned (and lost) at end of a round.

    All monetary values are in whole dollars.  ``rental_cost`` is a
    *positive* value representing money lost; it is subtracted in ``total``.

    ``total`` is the net dollar change applied by ``ease_dollars``:
    ``blind_reward + unused_hands_bonus + unused_discards_bonus
    + joker_dollars + interest − rental_cost``
    """

    blind_reward: int = 0
    """Dollars for beating the blind (``blind.dollars``).

    Set to 0 by stake modifiers (``no_blind_reward``) or skipped blinds.
    """

    unused_hands_bonus: int = 0
    """``hands_left × money_per_hand`` (default $1/hand).

    0 when ``modifiers['no_extra_hand_money']`` is truthy.
    """

    unused_discards_bonus: int = 0
    """``discards_left × money_per_discard``.

    0 unless ``modifiers['money_per_discard']`` is set (Green Deck).
    """

    interest: int = 0
    """``interest_amount × min(effective_money // 5, interest_cap // 5)``.

    Computed on post-rental balance.  0 when ``modifiers['no_interest']``
    or ``effective_money < 5``.
    """

    joker_dollars: int = 0
    """Sum of per-round joker dollar bonuses (Golden Joker $4, Cloud 9 …)."""

    rental_cost: int = 0
    """``rental_rate × count(rental jokers)``.  Positive; subtracted from total."""

    total: int = 0
    """Net dollars earned this round."""


# ---------------------------------------------------------------------------
# Discard cost helper (Golden Needle challenge)
# ---------------------------------------------------------------------------


def calculate_discard_cost(game_state: dict[str, Any]) -> int:
    """Return the dollar cost charged per discard action.

    Mirrors the ``G.GAME.modifiers.discard_cost`` check in
    ``state_events.lua:433`` — the Golden Needle challenge sets this to 1.

    Returns 0 if no discard cost modifier is active.
    """
    return game_state.get("modifiers", {}).get("discard_cost", 0)


# ---------------------------------------------------------------------------
# calculate_round_earnings
# ---------------------------------------------------------------------------


def calculate_round_earnings(
    blind: Blind,
    hands_left: int,
    discards_left: int,
    money: int,
    jokers: list[Card],
    game_state: dict[str, Any],
    rng: PseudoRandom | None = None,
    *,
    joker_dollars: int | None = None,
) -> RoundEarnings:
    """Compute end-of-round earnings for a beaten blind.

    Mirrors ``evaluate_round`` (state_events.lua:1135) combined with
    the rental deduction from state_events.lua:108.

    ``game_state`` keys used:

    +-----------------------+----------+--------------------------------------+
    | Key                   | Default  | Description                          |
    +=======================+==========+======================================+
    | ``interest_cap``      | 25       | ``G.GAME.interest_cap`` — maximum    |
    |                       |          | money that earns interest             |
    |                       |          | (Lua default 25 → 5 brackets → $5   |
    |                       |          | max at 1× rate; Seed Money→50,       |
    |                       |          | Money Tree→100).                     |
    +-----------------------+----------+--------------------------------------+
    | ``interest_amount``   | 1        | Dollars per $5 bracket.              |
    |                       |          | To the Moon adds +1 per copy.        |
    +-----------------------+----------+--------------------------------------+
    | ``rental_rate``       | 3        | Cost per rental joker per round.     |
    +-----------------------+----------+--------------------------------------+
    | ``modifiers``         | {}       | Dict of run modifiers:               |
    |                       |          | ``no_extra_hand_money``: bool        |
    |                       |          | ``money_per_hand``: int (default 1)  |
    |                       |          | ``money_per_discard``: int|None      |
    |                       |          | ``no_interest``: bool                |
    +-----------------------+----------+--------------------------------------+

    Args:
        blind: The :class:`~jackdaw.engine.blind.Blind` beaten this round.
        hands_left: Unused hands at round end.
        discards_left: Unused discards at round end.
        money: Bank balance *before* this round's earnings are applied.
        jokers: Active joker cards.
        game_state: Run-level state dict.
        rng: PseudoRandom instance (for end-of-round joker RNG effects).
        joker_dollars: Pre-computed joker dollar bonus from the
            ``on_end_of_round`` call in ``_round_won``.  When provided,
            ``on_end_of_round`` is **not** called again, avoiding a
            duplicate RNG consumption that would desync the PRNG state.

    Returns:
        :class:`RoundEarnings` with all components and their net total.
    """
    modifiers: dict[str, Any] = game_state.get("modifiers", {})
    interest_amount: int = game_state.get("interest_amount", 1)
    interest_cap: int = game_state.get("interest_cap", 25)
    rental_rate: int = game_state.get("rental_rate", 3)

    # ------------------------------------------------------------------
    # Step 1 — Rental deduction (state_events.lua:108)
    # calculate_rental() → ease_dollars(-G.GAME.rental_rate) per rental joker
    # Fires BEFORE evaluate_round; effective_money is used for interest.
    # ------------------------------------------------------------------
    rental_cost = sum(rental_rate for j in jokers if not j.debuff and j.ability.get("rental"))
    effective_money = money - rental_cost

    # ------------------------------------------------------------------
    # Step 2 — Blind reward (state_events.lua:1139)
    # ------------------------------------------------------------------
    blind_reward = blind.dollars

    # ------------------------------------------------------------------
    # Step 3 — Unused hands bonus (state_events.lua:1165)
    # hands_left * (modifiers.money_per_hand or 1)
    # ------------------------------------------------------------------
    if hands_left > 0 and not modifiers.get("no_extra_hand_money"):
        money_per_hand: int = modifiers.get("money_per_hand", 1)
        unused_hands_bonus = hands_left * money_per_hand
    else:
        unused_hands_bonus = 0

    # ------------------------------------------------------------------
    # Step 4 — Unused discards bonus (state_events.lua:1170)
    # Only when modifiers.money_per_discard is set (Green Deck)
    # ------------------------------------------------------------------
    money_per_discard: int | None = modifiers.get("money_per_discard")
    if discards_left > 0 and money_per_discard:
        unused_discards_bonus = discards_left * money_per_discard
    else:
        unused_discards_bonus = 0

    # ------------------------------------------------------------------
    # Step 5 — Joker dollar bonuses (state_events.lua:1175)
    # calc_dollar_bonus per joker: Golden Joker, Cloud 9, Satellite, etc.
    #
    # When joker_dollars is pre-computed (passed from _round_won), skip
    # on_end_of_round to avoid duplicate RNG consumption.
    # ------------------------------------------------------------------
    if joker_dollars is None:
        game_snap = GameSnapshot(
            money=money,
            hands_left=hands_left,
            discards_left=discards_left,
            joker_count=len(jokers),
        )
        end_result = on_end_of_round(jokers, game_snap, rng)
        joker_dollars = end_result["dollars_earned"]

    # ------------------------------------------------------------------
    # Step 6 — Interest (state_events.lua:1191)
    # interest_amount * min(effective_money // 5, interest_cap // 5)
    # Requires effective_money >= 5 and modifiers.no_interest not set.
    # ------------------------------------------------------------------
    if effective_money >= 5 and not modifiers.get("no_interest"):
        interest = interest_amount * min(
            effective_money // 5,
            interest_cap // 5,
        )
    else:
        interest = 0

    total = (
        blind_reward
        + unused_hands_bonus
        + unused_discards_bonus
        + joker_dollars
        + interest
        - rental_cost
    )

    return RoundEarnings(
        blind_reward=blind_reward,
        unused_hands_bonus=unused_hands_bonus,
        unused_discards_bonus=unused_discards_bonus,
        interest=interest,
        joker_dollars=joker_dollars,
        rental_cost=rental_cost,
        total=total,
    )
