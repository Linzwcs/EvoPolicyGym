"""Adapter between jackdaw Action types and balatrobot JSON-RPC calls.

Converts our frozen dataclass actions into balatrobot RPC payloads,
and converts balatrobot game state responses into our game_state dict.

See ``docs/balatrobot-action-mapping.md`` for the full mapping table.
"""

from __future__ import annotations

from typing import Any

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
)

# ---------------------------------------------------------------------------
# Action → balatrobot RPC
# ---------------------------------------------------------------------------

_STATE_MAP = {
    "BLIND_SELECT": GamePhase.BLIND_SELECT,
    "SELECTING_HAND": GamePhase.SELECTING_HAND,
    "ROUND_EVAL": GamePhase.ROUND_EVAL,
    "SHOP": GamePhase.SHOP,
    "SMODS_BOOSTER_OPENED": GamePhase.PACK_OPENING,
    "GAME_OVER": GamePhase.GAME_OVER,
}


def action_to_rpc(action: Action, game_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a jackdaw Action to a balatrobot JSON-RPC method + params.

    Returns ``{"method": str, "params": dict}`` suitable for sending
    as a JSON-RPC 2.0 request body (minus jsonrpc/id fields).

    *game_state* is required for swap actions so that the full
    permutation array can be constructed for the balatrobot RPC.
    """
    match action:
        case PlayHand(card_indices=indices):
            return {"method": "play", "params": {"cards": list(indices)}}

        case Discard(card_indices=indices):
            return {"method": "discard", "params": {"cards": list(indices)}}

        case SelectBlind():
            return {"method": "select", "params": {}}

        case SkipBlind():
            return {"method": "skip", "params": {}}

        case BuyCard(shop_index=idx):
            return {"method": "buy", "params": {"card": idx}}

        case SellCard(area=area, card_index=idx):
            if area == "jokers":
                return {"method": "sell", "params": {"joker": idx}}
            else:
                return {"method": "sell", "params": {"consumable": idx}}

        case UseConsumable(card_index=idx, target_indices=targets):
            params: dict[str, Any] = {"consumable": idx}
            if targets:
                params["cards"] = list(targets)
            return {"method": "use", "params": params}

        case RedeemVoucher(card_index=idx):
            return {"method": "buy", "params": {"voucher": idx}}

        case OpenBooster(card_index=idx):
            return {"method": "buy", "params": {"pack": idx}}

        case PickPackCard(card_index=idx, target_indices=targets):
            params = {"card": idx}
            if targets:
                params["targets"] = list(targets)
            return {"method": "pack", "params": params}

        case SkipPack():
            return {"method": "pack", "params": {"skip": True}}

        case Reroll():
            return {"method": "reroll", "params": {}}

        case NextRound():
            return {"method": "next_round", "params": {}}

        case CashOut():
            return {"method": "cash_out", "params": {}}

        case SortHand(mode=mode):
            return {"method": "rearrange", "params": {"sort": mode}}

        case SwapHandLeft(idx=idx) | SwapHandRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("hand", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapHandLeft) else idx + 1
            order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"hand": order}}

        case SwapJokersLeft(idx=idx) | SwapJokersRight(idx=idx):
            assert game_state is not None, "game_state required for swap actions"
            n = len(game_state.get("jokers", []))
            order = list(range(n))
            other = idx - 1 if isinstance(action, SwapJokersLeft) else idx + 1
            order[idx], order[other] = order[other], order[idx]
            return {"method": "rearrange", "params": {"jokers": order}}

        case _:
            raise ValueError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Balatrobot state → game_state
# ---------------------------------------------------------------------------


def bot_state_to_game_state(bot: dict[str, Any]) -> dict[str, Any]:
    """Convert a balatrobot gamestate response to our game_state dict.

    Maps the key fields for validation and comparison. Does NOT create
    a fully functional game_state (no RNG, no Card objects) — this is
    for read-only comparison purposes.
    """
    gs: dict[str, Any] = {}

    # Phase
    state_str = bot.get("state", "")
    gs["phase"] = _STATE_MAP.get(state_str, state_str)

    # Economy
    gs["dollars"] = bot.get("money", 0)

    # Ante / round
    gs["round_resets"] = {"ante": bot.get("ante_num", 1)}
    gs["round"] = bot.get("round_num", 0)

    # Round state
    br = bot.get("round", {})
    gs["current_round"] = {
        "hands_left": br.get("hands_left", 0),
        "discards_left": br.get("discards_left", 0),
        "hands_played": br.get("hands_played", 0),
        "discards_used": br.get("discards_used", 0),
        "reroll_cost": br.get("reroll_cost", 5),
    }
    gs["chips"] = br.get("chips", 0)

    # Cards — extract keys only (no full Card objects)
    gs["hand_keys"] = [c["key"] for c in bot.get("hand", {}).get("cards", [])]
    gs["hand_count"] = len(gs["hand_keys"])

    gs["deck_size"] = bot.get("cards", {}).get("count", 0)
    gs["deck_keys"] = [c["key"] for c in bot.get("cards", {}).get("cards", [])]

    gs["joker_keys"] = [c["key"] for c in bot.get("jokers", {}).get("cards", [])]
    gs["joker_count"] = len(gs["joker_keys"])

    gs["consumable_keys"] = [c["key"] for c in bot.get("consumables", {}).get("cards", [])]

    # Blinds
    blinds = bot.get("blinds", {})
    gs["blind_info"] = {}
    for btype in ("small", "big", "boss"):
        bi = blinds.get(btype, {})
        gs["blind_info"][btype] = {
            "name": bi.get("name", ""),
            "status": bi.get("status", ""),
            "score": bi.get("score", 0),
            "tag_name": bi.get("tag_name", ""),
        }

    # Seed / deck / stake
    gs["seed"] = bot.get("seed", "")
    gs["deck_type"] = bot.get("deck", "")
    gs["stake_type"] = bot.get("stake", "")
    gs["won"] = bot.get("won", False)

    return gs


# ---------------------------------------------------------------------------
# Game state → balatrobot response (reverse direction)
# ---------------------------------------------------------------------------


def game_state_to_bot_response(gs: dict[str, Any]) -> dict[str, Any]:
    """Convert our game_state to balatrobot's JSON response format.

    Delegates to :func:`jackdaw.bridge.serializer.game_state_to_bot_response`.
    """
    from jackdaw.bridge.serializer import (
        game_state_to_bot_response as _serialize,
    )

    return _serialize(gs)


# ---------------------------------------------------------------------------
# Game state → comparison keys
# ---------------------------------------------------------------------------


def extract_comparison_keys(gs: dict[str, Any]) -> dict[str, Any]:
    """Extract the key fields from our game_state for comparison with bot state.

    Returns a flat dict suitable for field-by-field comparison with the
    output of :func:`bot_state_to_game_state`.
    """
    cr = gs.get("current_round", {})
    return {
        "dollars": gs.get("dollars", 0),
        "chips": gs.get("chips", 0),
        "ante": gs.get("round_resets", {}).get("ante", 1),
        "hands_left": cr.get("hands_left", 0),
        "discards_left": cr.get("discards_left", 0),
        "hand_count": len(gs.get("hand", [])),
        "deck_size": len(gs.get("deck", [])),
        "joker_count": len(gs.get("jokers", [])),
    }
