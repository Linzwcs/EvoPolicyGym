"""Strict PolicyValue Actions translated to Jackdaw engine Actions."""

from __future__ import annotations

from typing import Any, NoReturn, cast

from evopolicygym.authoring import InvalidAction
from evopolicygym.policy import PolicyValue
from jackdaw.engine import (
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
from jackdaw.engine.consumables import can_use_consumable
from jackdaw.env import get_action_mask, get_consumable_target_spec
from jackdaw.env.consumable_targets import (
    get_valid_target_cards,
    validate_card_targets,
)

_SIMPLE_ACTIONS: dict[str, tuple[int, type[object], dict[str, object]]] = {
    "select_blind": (2, SelectBlind, {}),
    "skip_blind": (3, SkipBlind, {}),
    "cash_out": (4, CashOut, {}),
    "reroll_shop": (5, Reroll, {}),
    "next_round": (6, NextRound, {}),
    "skip_pack": (7, SkipPack, {}),
    "sort_hand_by_rank": (19, SortHand, {"mode": "rank"}),
    "sort_hand_by_suit": (20, SortHand, {"mode": "suit"}),
}

_ENTITY_ACTIONS: dict[str, tuple[int, type[object], str, dict[str, object]]] = {
    "buy_card": (8, BuyCard, "shop_index", {}),
    "sell_joker": (9, SellCard, "card_index", {"area": "jokers"}),
    "sell_consumable": (
        10,
        SellCard,
        "card_index",
        {"area": "consumables"},
    ),
    "redeem_voucher": (12, RedeemVoucher, "card_index", {}),
    "open_booster": (13, OpenBooster, "card_index", {}),
    "swap_joker_left": (15, SwapJokersLeft, "idx", {}),
    "swap_joker_right": (16, SwapJokersRight, "idx", {}),
    "swap_hand_left": (17, SwapHandLeft, "idx", {}),
    "swap_hand_right": (18, SwapHandRight, "idx", {}),
}

_PHASES_WITH_CONSUMABLES = {
    GamePhase.BLIND_SELECT,
    GamePhase.SELECTING_HAND,
    GamePhase.ROUND_EVAL,
    GamePhase.SHOP,
}


def decode_action(value: PolicyValue, game_state: dict[str, Any]) -> Action:
    """Validate one complete Action without repairing or replacing it."""

    if type(value) is not dict:
        _invalid()
    kind = value.get("kind")
    if type(kind) is not str:
        _invalid()

    if kind in {"play_hand", "discard"}:
        _require_exact_keys(value, {"kind", "card_indices"})
        indices = _card_indices(value["card_indices"])
        action_type = 0 if kind == "play_hand" else 1
        if not _type_is_legal(game_state, action_type):
            _invalid()
        hand = _trusted_list(game_state, "hand")
        if not 1 <= len(indices) <= min(5, len(hand)):
            _invalid()
        if any(index >= len(hand) for index in indices):
            _invalid()
        if kind == "play_hand":
            return PlayHand(card_indices=indices)
        return Discard(card_indices=indices)

    simple = _SIMPLE_ACTIONS.get(kind)
    if simple is not None:
        _require_exact_keys(value, {"kind"})
        action_type, constructor, arguments = simple
        if not _type_is_legal(game_state, action_type):
            _invalid()
        return constructor(**arguments)

    entity = _ENTITY_ACTIONS.get(kind)
    if entity is not None:
        _require_exact_keys(value, {"kind", "target_index"})
        target_index = _index(value["target_index"])
        action_type, constructor, argument_name, arguments = entity
        if not _entity_is_legal(game_state, action_type, target_index):
            _invalid()
        keyword_arguments = dict(arguments)
        keyword_arguments[argument_name] = target_index
        return constructor(**keyword_arguments)

    if kind == "use_consumable":
        _require_exact_keys(
            value,
            {"kind", "target_index", "card_indices"},
        )
        return _use_consumable(
            game_state,
            target_index=_index(value["target_index"]),
            card_indices=_card_indices(value["card_indices"], allow_empty=True),
        )

    if kind == "pick_pack_card":
        _require_exact_keys(
            value,
            {"kind", "target_index", "card_indices"},
        )
        return _pick_pack_card(
            game_state,
            target_index=_index(value["target_index"]),
            card_indices=_card_indices(value["card_indices"], allow_empty=True),
        )

    _invalid()


def legal_action_descriptors(game_state: dict[str, Any]) -> list[PolicyValue]:
    """Describe the complete currently legal factored Action domain."""

    mask = get_action_mask(game_state)
    descriptors: list[PolicyValue] = []
    hand = _trusted_list(game_state, "hand")

    for kind, action_type in (("play_hand", 0), ("discard", 1)):
        if bool(mask.type_mask[action_type]):
            descriptors.append(
                {
                    "kind": kind,
                    "card_indices": _policy_int_list(list(range(len(hand)))),
                    "min_cards": 1,
                    "max_cards": min(5, len(hand)),
                    "selection_order_matters": kind == "play_hand",
                }
            )

    for kind, (action_type, _, _) in _SIMPLE_ACTIONS.items():
        if bool(mask.type_mask[action_type]):
            descriptors.append({"kind": kind})

    target_areas = {
        "buy_card": "shop.cards",
        "sell_joker": "jokers",
        "sell_consumable": "consumables",
        "redeem_voucher": "shop.vouchers",
        "open_booster": "shop.boosters",
        "swap_joker_left": "jokers",
        "swap_joker_right": "jokers",
        "swap_hand_left": "hand",
        "swap_hand_right": "hand",
    }
    for kind, (action_type, _, _, _) in _ENTITY_ACTIONS.items():
        targets = _legal_entity_indices(mask, action_type)
        if targets:
            descriptors.append(
                {
                    "kind": kind,
                    "target_area": target_areas[kind],
                    "target_indices": _policy_int_list(targets),
                }
            )

    consumables = _trusted_list(game_state, "consumables")
    usable: list[PolicyValue] = []
    for index, card in enumerate(consumables):
        target = _consumable_descriptor(game_state, card, index)
        if target is not None:
            usable.append(target)
    if usable:
        descriptors.append(
            {
                "kind": "use_consumable",
                "target_area": "consumables",
                "targets": usable,
            }
        )

    if _phase(game_state) == GamePhase.PACK_OPENING:
        pack_targets: list[PolicyValue] = []
        for index, card in enumerate(_trusted_list(game_state, "pack_cards")):
            target = _pack_card_descriptor(game_state, card, index)
            if target is not None:
                pack_targets.append(target)
        if pack_targets:
            descriptors.append(
                {
                    "kind": "pick_pack_card",
                    "target_area": "pack.cards",
                    "targets": pack_targets,
                }
            )

    return descriptors


def _use_consumable(
    game_state: dict[str, Any],
    *,
    target_index: int,
    card_indices: tuple[int, ...],
) -> Action:
    if _phase(game_state) not in _PHASES_WITH_CONSUMABLES:
        _invalid()
    consumables = _trusted_list(game_state, "consumables")
    if target_index >= len(consumables):
        _invalid()
    card = consumables[target_index]
    hand = _trusted_list(game_state, "hand")
    if not validate_card_targets(card, card_indices, hand, game_state):
        _invalid()
    highlighted = [hand[index] for index in card_indices]
    if not can_use_consumable(
        card,
        highlighted=highlighted,
        hand_cards=hand,
        jokers=_trusted_list(game_state, "jokers"),
        consumables=consumables,
        consumable_limit=_trusted_int(game_state, "consumable_slots"),
        joker_limit=_trusted_int(game_state, "joker_slots"),
        game_state=game_state,
    ):
        _invalid()
    return UseConsumable(
        card_index=target_index,
        target_indices=card_indices or None,
    )


def _pick_pack_card(
    game_state: dict[str, Any],
    *,
    target_index: int,
    card_indices: tuple[int, ...],
) -> Action:
    if _phase(game_state) != GamePhase.PACK_OPENING:
        _invalid()
    cards = _trusted_list(game_state, "pack_cards")
    if target_index >= len(cards):
        _invalid()
    if _trusted_int(game_state, "pack_choices_remaining") <= 0:
        _invalid()
    card = cards[target_index]
    card_set = _card_set(card)
    if card_set in {"Tarot", "Planet", "Spectral"}:
        if not validate_card_targets(
            card,
            card_indices,
            _trusted_list(game_state, "hand"),
            game_state,
        ):
            _invalid()
    elif card_indices:
        _invalid()
    if card_set == "Joker":
        jokers = _trusted_list(game_state, "jokers")
        negative = bool((getattr(card, "edition", None) or {}).get("negative"))
        if len(jokers) >= _trusted_int(game_state, "joker_slots") and not negative:
            _invalid()
    return PickPackCard(
        card_index=target_index,
        target_indices=card_indices or None,
    )


def _consumable_descriptor(
    game_state: dict[str, Any],
    card: Any,
    index: int,
) -> PolicyValue | None:
    hand = _trusted_list(game_state, "hand")
    spec = get_consumable_target_spec(card, game_state)
    valid = get_valid_target_cards(card, hand, game_state)
    if spec.needs_card_targets:
        if len(valid) < spec.min_targets:
            return None
        candidates = valid[: spec.max_targets]
    else:
        candidates = []
    sample = tuple(candidates[: spec.min_targets])
    if not can_use_consumable(
        card,
        highlighted=[hand[item] for item in sample],
        hand_cards=hand,
        jokers=_trusted_list(game_state, "jokers"),
        consumables=_trusted_list(game_state, "consumables"),
        consumable_limit=_trusted_int(game_state, "consumable_slots"),
        joker_limit=_trusted_int(game_state, "joker_slots"),
        game_state=game_state,
    ):
        return None
    return {
        "target_index": index,
        "card_indices": _policy_int_list(valid),
        "min_cards": spec.min_targets,
        "max_cards": spec.max_targets,
    }


def _pack_card_descriptor(
    game_state: dict[str, Any],
    card: Any,
    index: int,
) -> PolicyValue | None:
    if _card_set(card) == "Joker":
        negative = bool((getattr(card, "edition", None) or {}).get("negative"))
        if (
            len(_trusted_list(game_state, "jokers")) >= _trusted_int(game_state, "joker_slots")
            and not negative
        ):
            return None
    if _card_set(card) not in {"Tarot", "Planet", "Spectral"}:
        return {
            "target_index": index,
            "card_indices": [],
            "min_cards": 0,
            "max_cards": 0,
        }
    hand = _trusted_list(game_state, "hand")
    spec = get_consumable_target_spec(card, game_state)
    valid = get_valid_target_cards(card, hand, game_state)
    if spec.needs_card_targets and len(valid) < spec.min_targets:
        return None
    return {
        "target_index": index,
        "card_indices": _policy_int_list(valid),
        "min_cards": spec.min_targets,
        "max_cards": spec.max_targets,
    }


def _phase(game_state: dict[str, Any]) -> GamePhase:
    value = game_state.get("phase")
    if isinstance(value, GamePhase):
        return value
    if type(value) is str:
        return GamePhase(value)
    raise RuntimeError("Jackdaw returned an invalid phase")


def _type_is_legal(game_state: dict[str, Any], action_type: int) -> bool:
    mask = get_action_mask(game_state)
    return bool(mask.type_mask[action_type])


def _entity_is_legal(
    game_state: dict[str, Any],
    action_type: int,
    target_index: int,
) -> bool:
    mask = get_action_mask(game_state)
    if not bool(mask.type_mask[action_type]):
        return False
    target_mask = mask.entity_masks.get(action_type)
    return (
        target_mask is not None
        and target_index < len(target_mask)
        and bool(target_mask[target_index])
    )


def _legal_entity_indices(mask: Any, action_type: int) -> list[int]:
    if not bool(mask.type_mask[action_type]):
        return []
    target_mask = mask.entity_masks.get(action_type)
    if target_mask is None:
        return []
    return [index for index, allowed in enumerate(target_mask) if bool(allowed)]


def _policy_int_list(values: list[int]) -> list[PolicyValue]:
    return cast(list[PolicyValue], values)


def _card_indices(
    value: PolicyValue,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if type(value) is not list:
        _invalid()
    indices = tuple(_index(item) for item in value)
    if not allow_empty and not indices:
        _invalid()
    if len(indices) != len(set(indices)):
        _invalid()
    return indices


def _index(value: PolicyValue) -> int:
    if type(value) is not int or value < 0:
        _invalid()
    return value


def _require_exact_keys(
    value: dict[str, PolicyValue],
    expected: set[str],
) -> None:
    if set(value) != expected:
        _invalid()


def _trusted_list(game_state: dict[str, Any], key: str) -> list[Any]:
    value = game_state.get(key)
    if type(value) is not list:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _trusted_int(game_state: dict[str, Any], key: str) -> int:
    value = game_state.get(key)
    if type(value) is not int:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _card_set(card: Any) -> str:
    ability = getattr(card, "ability", None)
    if type(ability) is not dict:
        return ""
    value = ability.get("set", "")
    return value if type(value) is str else ""


def _invalid() -> NoReturn:
    raise InvalidAction()


__all__ = ["decode_action", "legal_action_descriptors"]
