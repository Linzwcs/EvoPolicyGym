"""Deserialize balatrobot JSON-RPC calls into engine Actions.

The reverse of ``action_to_rpc()`` in ``balatrobot_adapter.py`` — given a
JSON-RPC method name and params dict, produce the correct engine Action.
"""

from __future__ import annotations

from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
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
    SwapHandRight,
    SwapJokersRight,
    UseConsumable,
)

# Methods that are queries or lifecycle commands, not game actions.
_QUERY_METHODS = frozenset(
    {
        "gamestate",
        "health",
        "start",
        "menu",
        "save",
        "load",
        "screenshot",
        "set",
        "add",
        "rpc.discover",
    }
)


def rpc_to_action(method: str, params: dict | None = None) -> Action | None:
    """Convert a balatrobot JSON-RPC method + params to an engine Action.

    Returns ``None`` for query-only methods (gamestate, health, start, etc.).
    Raises ``ValueError`` for unknown methods or malformed params.
    """
    if params is None:
        params = {}

    if method in _QUERY_METHODS:
        return None

    if method == "play":
        return PlayHand(card_indices=tuple(params.get("cards", ())))

    if method == "discard":
        return Discard(card_indices=tuple(params.get("cards", ())))

    if method == "select":
        return SelectBlind()

    if method == "skip":
        return SkipBlind()

    if method == "buy":
        if "card" in params:
            return BuyCard(shop_index=params["card"])
        if "voucher" in params:
            return RedeemVoucher(card_index=params["voucher"])
        if "pack" in params:
            return OpenBooster(card_index=params["pack"])
        raise ValueError(f"buy: unrecognized params {params!r}")

    if method == "sell":
        if "joker" in params:
            return SellCard(area="jokers", card_index=params["joker"])
        if "consumable" in params:
            return SellCard(area="consumables", card_index=params["consumable"])
        raise ValueError(f"sell: unrecognized params {params!r}")

    if method == "use":
        if "consumable" not in params:
            raise ValueError(f"use: missing 'consumable' in params {params!r}")
        targets = params.get("cards")
        return UseConsumable(
            card_index=params["consumable"],
            target_indices=tuple(targets) if targets else None,
        )

    if method == "reroll":
        return Reroll()

    if method == "next_round":
        return NextRound()

    if method == "cash_out":
        return CashOut()

    if method == "pack":
        if params.get("skip"):
            return SkipPack()
        if "card" in params:
            targets = params.get("targets")
            return PickPackCard(
                card_index=params["card"],
                target_indices=tuple(targets) if targets else None,
            )
        raise ValueError(f"pack: unrecognized params {params!r}")

    if method == "rearrange":
        if "sort" in params:
            return SortHand(mode=params["sort"])
        if "hand" in params:
            return _permutation_to_swap(params["hand"], "hand")
        if "jokers" in params:
            return _permutation_to_swap(params["jokers"], "jokers")
        raise ValueError(f"rearrange: unrecognized params {params!r}")

    raise ValueError(f"Unknown RPC method: {method!r}")


def _permutation_to_swap(perm: list[int], area: str) -> Action:
    """Convert a permutation array to a swap action.

    For adjacent swaps, returns the canonical ``SwapRight(lower_idx)`` form.
    For non-adjacent permutations (legacy), decomposes to the first adjacent
    swap needed via a bubble-sort scan.
    """
    n = len(perm)

    # Find positions that differ from identity
    diffs = [i for i in range(n) if perm[i] != i]

    if len(diffs) == 2:
        lo, hi = diffs
        if hi - lo == 1:
            # Adjacent swap — use canonical SwapRight(lower_idx)
            if area == "hand":
                return SwapHandRight(idx=lo)
            return SwapJokersRight(idx=lo)

    # Non-adjacent or complex permutation: find first bubble-sort inversion
    for i in range(n - 1):
        if perm[i] > perm[i + 1]:
            if area == "hand":
                return SwapHandRight(idx=i)
            return SwapJokersRight(idx=i)

    # No inversion found — find first out-of-place element and swap right
    for i in range(n):
        if perm[i] != i:
            if area == "hand":
                return SwapHandRight(idx=i)
            return SwapJokersRight(idx=i)

    raise ValueError(f"Identity permutation passed to _permutation_to_swap: {perm!r}")
