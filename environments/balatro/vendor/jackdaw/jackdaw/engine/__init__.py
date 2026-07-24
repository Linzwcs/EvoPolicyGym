"""Jackdaw engine — headless Balatro simulator.

Public API:

    initialize_run(back_key, stake, seed, ...) → game_state
    step(game_state, action) → game_state
    get_legal_actions(game_state) → list[Action]
    simulate_run(back_key, stake, seed, agent, ...) → game_state
"""

from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run

__all__ = [
    # Core functions
    "initialize_run",
    "step",
    "get_legal_actions",
    "simulate_run",
    "random_agent",
    "greedy_play_agent",
    # Types
    "Action",
    "GamePhase",
    "IllegalActionError",
    # Action types
    "BuyCard",
    "CashOut",
    "Discard",
    "NextRound",
    "OpenBooster",
    "PickPackCard",
    "PlayHand",
    "RedeemVoucher",
    "Reroll",
    "SelectBlind",
    "SellCard",
    "SkipBlind",
    "SkipPack",
    "SortHand",
    "SwapHandLeft",
    "SwapHandRight",
    "SwapJokersLeft",
    "SwapJokersRight",
    "UseConsumable",
]
