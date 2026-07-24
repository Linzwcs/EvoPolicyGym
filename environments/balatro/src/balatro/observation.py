"""Semantic, bounded observations projected from trusted Jackdaw state."""

from __future__ import annotations

import math
from typing import Any, cast

from evopolicygym.policy import PolicyValue
from jackdaw.engine import GamePhase
from jackdaw.engine.blind import Blind
from jackdaw.engine.data.prototypes import VOUCHERS
from jackdaw.engine.economy import RoundEarnings
from jackdaw.engine.scoring import ScoreResult

from .actions import legal_action_descriptors
from .rules import blind_rule, visible_card_rule, visible_tag_rule

_MAX_SCORE_BREAKDOWN_ENTRIES = 32
_MAX_SCORE_BREAKDOWN_TEXT = 240
_HAND_TYPES = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)

_ABILITY_FIELDS = (
    "effect",
    "mult",
    "x_mult",
    "h_mult",
    "h_x_mult",
    "h_dollars",
    "p_dollars",
    "t_mult",
    "t_chips",
    "bonus",
    "extra",
    "extra_value",
    "perma_bonus",
    "h_size",
    "d_size",
    "type",
)


def encode_observation(
    game_state: dict[str, Any],
    *,
    step_count: int,
) -> dict[str, PolicyValue]:
    """Return the public state a coding-agent-authored Policy can reason over."""

    current_round = _trusted_dict(game_state, "current_round")
    round_resets = _trusted_dict(game_state, "round_resets")
    hand = _trusted_list(game_state, "hand")
    jokers = _trusted_list(game_state, "jokers")
    consumables = _trusted_list(game_state, "consumables")
    shop_cards = _optional_list(game_state, "shop_cards")
    shop_vouchers = _optional_list(game_state, "shop_vouchers")
    shop_boosters = _optional_list(game_state, "shop_boosters")
    pack_cards = _optional_list(game_state, "pack_cards")

    return {
        "schema": "evopolicygym-balatro/observation-v1",
        "phase": _phase(game_state).value,
        "progress": {
            "ante": _exact_int(round_resets.get("ante", 1), "ante"),
            "rounds_cleared": _exact_int(
                game_state.get("round", 0),
                "round",
            ),
            "win_ante": _exact_int(
                game_state.get("win_ante", 8),
                "win_ante",
            ),
            "blind_on_deck": _optional_text(
                game_state.get("blind_on_deck"),
            ),
            "won": bool(game_state.get("won", False)),
            "steps": step_count,
        },
        "resources": {
            "money": _exact_int(game_state.get("dollars", 0), "dollars"),
            "chips": _exact_int(game_state.get("chips", 0), "chips"),
            "hands_left": _exact_int(
                current_round.get("hands_left", 0),
                "hands_left",
            ),
            "discards_left": _exact_int(
                current_round.get("discards_left", 0),
                "discards_left",
            ),
            "reroll_cost": _exact_int(
                current_round.get("reroll_cost", 0),
                "reroll_cost",
            ),
            "free_rerolls": _exact_int(
                current_round.get("free_rerolls", 0),
                "free_rerolls",
            ),
            "hand_size": _exact_int(
                game_state.get("hand_size", 0),
                "hand_size",
            ),
            "joker_slots": _exact_int(
                game_state.get("joker_slots", 0),
                "joker_slots",
            ),
            "consumable_slots": _exact_int(
                game_state.get("consumable_slots", 0),
                "consumable_slots",
            ),
        },
        "rules": {
            "deck": _text(
                game_state.get("selected_back_key", ""),
                "selected_back_key",
            ),
            "stake": _exact_int(game_state.get("stake", 1), "stake"),
        },
        "blind": _blind(game_state),
        "last_hand": _last_hand(game_state),
        "round_earnings": _round_earnings(game_state),
        "hand": [
            _card(
                card,
                index=index,
                hide_face_down=True,
                game_state=game_state,
            )
            for index, card in enumerate(hand)
        ],
        "jokers": [
            _card(
                card,
                index=index,
                hide_face_down=False,
                game_state=game_state,
            )
            for index, card in enumerate(jokers)
        ],
        "consumables": [
            _card(
                card,
                index=index,
                hide_face_down=False,
                game_state=game_state,
            )
            for index, card in enumerate(consumables)
        ],
        "shop": {
            "cards": [
                _card(
                    card,
                    index=index,
                    hide_face_down=False,
                    game_state=game_state,
                )
                for index, card in enumerate(shop_cards)
            ],
            "vouchers": [
                _card(
                    card,
                    index=index,
                    hide_face_down=False,
                    game_state=game_state,
                )
                for index, card in enumerate(shop_vouchers)
            ],
            "boosters": [
                _card(
                    card,
                    index=index,
                    hide_face_down=False,
                    game_state=game_state,
                )
                for index, card in enumerate(shop_boosters)
            ],
        },
        "pack": {
            "type": _text(game_state.get("pack_type", ""), "pack_type"),
            "choices_remaining": _exact_int(
                game_state.get("pack_choices_remaining", 0),
                "pack_choices_remaining",
            ),
            "cards": [
                _card(
                    card,
                    index=index,
                    hide_face_down=False,
                    game_state=game_state,
                )
                for index, card in enumerate(pack_cards)
            ],
        },
        "deck": _deck(game_state),
        "poker_hands": _poker_hands(game_state),
        "vouchers": _vouchers(game_state),
        "tags": _tags(game_state),
        "legal_actions": legal_action_descriptors(game_state),
    }


def replay_state(observation: PolicyValue) -> dict[str, PolicyValue]:
    """Select the compact visual state retained in public replay artifacts."""

    if type(observation) is not dict:
        raise ValueError("Balatro replay observation is invalid")
    selected: dict[str, PolicyValue] = {}
    for key in (
        "phase",
        "progress",
        "resources",
        "rules",
        "blind",
        "last_hand",
        "round_earnings",
        "hand",
        "jokers",
        "consumables",
        "shop",
        "pack",
    ):
        if key not in observation:
            raise ValueError("Balatro replay observation is incomplete")
        selected[key] = observation[key]
    return selected


def _last_hand(game_state: dict[str, Any]) -> dict[str, PolicyValue]:
    result = game_state.get("last_score_result")
    if result is None:
        return {
            "handname": "",
            "hand_level": "",
            "hand_type": "",
            "level": 0,
            "chips": 0,
            "mult": 0,
            "total": 0,
            "dollars_earned": 0,
            "debuffed": False,
            "scoring_cards": [],
            "breakdown": [],
            "breakdown_truncated": False,
        }
    if not isinstance(result, ScoreResult):
        raise RuntimeError("Jackdaw returned an invalid last_score_result")

    hand_type = _text(result.hand_type, "last_score_result.hand_type")
    level = _last_hand_level(game_state, hand_type)
    scoring_cards = result.scoring_cards
    breakdown = result.breakdown
    if type(scoring_cards) is not list:
        raise RuntimeError("Jackdaw returned invalid scoring_cards")
    if type(breakdown) is not list or any(type(item) is not str for item in breakdown):
        raise RuntimeError("Jackdaw returned invalid score breakdown")

    if len(breakdown) <= _MAX_SCORE_BREAKDOWN_ENTRIES:
        public_breakdown = [
            item[:_MAX_SCORE_BREAKDOWN_TEXT] for item in breakdown
        ]
    else:
        public_breakdown = [
            item[:_MAX_SCORE_BREAKDOWN_TEXT]
            for item in breakdown[: _MAX_SCORE_BREAKDOWN_ENTRIES - 1]
        ]
        public_breakdown.append(
            breakdown[-1][:_MAX_SCORE_BREAKDOWN_TEXT],
        )

    return {
        # Keep the original Jackdaw-shaped names for the replay UI and
        # existing Policies while exposing explicit semantic aliases.
        "handname": hand_type,
        "hand_level": f"Level {level}" if level else "",
        "hand_type": hand_type,
        "level": level,
        "chips": _finite_number(
            result.chips,
            "last_score_result.chips",
        ),
        "mult": _finite_number(
            result.mult,
            "last_score_result.mult",
        ),
        "total": _exact_int(
            result.total,
            "last_score_result.total",
        ),
        "dollars_earned": _exact_int(
            result.dollars_earned,
            "last_score_result.dollars_earned",
        ),
        "debuffed": bool(result.debuffed),
        "scoring_cards": [
            _scoring_card(card)
            for card in scoring_cards[:5]
        ],
        "breakdown": public_breakdown,
        "breakdown_truncated": len(breakdown) > _MAX_SCORE_BREAKDOWN_ENTRIES,
    }


def _round_earnings(game_state: dict[str, Any]) -> PolicyValue:
    earnings = game_state.get("round_earnings")
    if earnings is None:
        return None
    if not isinstance(earnings, RoundEarnings):
        raise RuntimeError("Jackdaw returned invalid round_earnings")
    return {
        "blind_dollar_reward": _exact_int(
            earnings.blind_reward,
            "round_earnings.blind_reward",
        ),
        "unused_hands_bonus": _exact_int(
            earnings.unused_hands_bonus,
            "round_earnings.unused_hands_bonus",
        ),
        "unused_discards_bonus": _exact_int(
            earnings.unused_discards_bonus,
            "round_earnings.unused_discards_bonus",
        ),
        "joker_dollars": _exact_int(
            earnings.joker_dollars,
            "round_earnings.joker_dollars",
        ),
        "interest": _exact_int(
            earnings.interest,
            "round_earnings.interest",
        ),
        "rental_cost": _exact_int(
            earnings.rental_cost,
            "round_earnings.rental_cost",
        ),
        "total_dollars": _exact_int(
            earnings.total,
            "round_earnings.total",
        ),
    }


def _last_hand_level(game_state: dict[str, Any], hand_type: str) -> int:
    if hand_type == "NULL":
        return 0
    levels = game_state.get("hand_levels")
    if levels is None or not callable(getattr(levels, "get_state", None)):
        raise RuntimeError("Jackdaw returned invalid hand levels")
    state = levels.get_state(hand_type)
    return _exact_int(getattr(state, "level", None), "hand.level")


def _scoring_card(card: Any) -> dict[str, PolicyValue]:
    public = _card(card, index=0, hide_face_down=True)
    return {
        key: public[key]
        for key in (
            "key",
            "name",
            "facing",
            "rank",
            "suit",
            "chips",
            "enhancement",
            "edition",
            "seal",
            "debuffed",
        )
    }


def _blind(game_state: dict[str, Any]) -> PolicyValue:
    blind = game_state.get("blind")
    if blind is None or _phase(game_state) == GamePhase.BLIND_SELECT:
        round_resets = _trusted_dict(game_state, "round_resets")
        blind_on_deck = _optional_text(game_state.get("blind_on_deck"))
        choices = round_resets.get("blind_choices", {})
        if type(choices) is not dict or blind_on_deck is None:
            return None
        key = choices.get(blind_on_deck)
        if type(key) is not str:
            return None
        modifiers = _optional_dict(game_state, "modifiers")
        no_rewards = modifiers.get("no_blind_reward", {})
        no_reward = type(no_rewards) is dict and bool(no_rewards.get(blind_on_deck, False))
        blind = Blind.create(
            key,
            _exact_int(round_resets.get("ante", 1), "ante"),
            scaling=_exact_int(modifiers.get("scaling", 1), "scaling"),
            ante_scaling=float(
                _trusted_dict(game_state, "starting_params").get(
                    "ante_scaling",
                    1,
                )
            ),
            no_blind_reward=no_reward,
        )
    blind_on_deck = _optional_text(game_state.get("blind_on_deck"))
    blind_tags = _trusted_dict(game_state, "round_resets").get(
        "blind_tags",
        {},
    )
    skip_tag: PolicyValue = None
    if type(blind_tags) is dict and blind_on_deck is not None:
        skip_tag_key = _optional_text(blind_tags.get(blind_on_deck))
        if skip_tag_key is not None:
            skip_tag = visible_tag_rule(skip_tag_key)
    return {
        "key": _text(getattr(blind, "key", ""), "blind.key"),
        "name": _text(getattr(blind, "name", ""), "blind.name"),
        "rule": blind_rule(
            _text(getattr(blind, "key", ""), "blind.key"),
        ),
        "target_chips": _exact_int(
            getattr(blind, "chips", 0),
            "blind.chips",
        ),
        "dollar_reward": _exact_int(
            getattr(blind, "dollars", 0),
            "blind.dollars",
        ),
        "boss": bool(getattr(blind, "boss", False)),
        "disabled": bool(getattr(blind, "disabled", False)),
        "triggered": bool(getattr(blind, "triggered", False)),
        "debuff": _safe_public_value(
            getattr(blind, "debuff_config", {}),
        ),
        "skip_tag": skip_tag,
    }


def _tags(game_state: dict[str, Any]) -> list[PolicyValue]:
    result: list[PolicyValue] = []
    for entry in _optional_list(game_state, "awarded_tags"):
        if type(entry) is not dict:
            raise RuntimeError("Jackdaw returned invalid awarded tag")
        key = _text(entry.get("key"), "awarded_tag.key")
        public = visible_tag_rule(key)
        source_blind = _optional_text(entry.get("blind"))
        if source_blind is not None:
            public["source_blind"] = source_blind
        result.append(public)
    return result


def _card(
    card: Any,
    *,
    index: int,
    hide_face_down: bool,
    game_state: dict[str, Any] | None = None,
) -> dict[str, PolicyValue]:
    facing = getattr(card, "facing", "front")
    face_down = hide_face_down and facing == "back"
    base = getattr(card, "base", None)
    ability = getattr(card, "ability", {})
    if type(ability) is not dict:
        raise RuntimeError("Jackdaw returned an invalid card ability")
    name = ability.get("name", "")
    card_set = ability.get("set", "")
    rank: str | None = None
    suit: str | None = None
    chips: int | None = None
    if base is not None and not face_down:
        rank = _text(getattr(getattr(base, "rank", None), "value", ""), "rank")
        suit = _text(getattr(getattr(base, "suit", None), "value", ""), "suit")
        chips = _exact_int(getattr(base, "nominal", 0), "nominal")
        if not name:
            name = f"{rank} of {suit}"
    public_ability: dict[str, PolicyValue] = {}
    for key in _ABILITY_FIELDS:
        if key in ability:
            public_ability[key] = _safe_public_value(ability[key])
    edition = getattr(card, "edition", None)
    public_edition: str | None = None
    if type(edition) is dict:
        public_edition = next(
            (str(key) for key, enabled in edition.items() if enabled),
            None,
        )
    key = _text(getattr(card, "center_key", ""), "center_key")
    public_rule = (
        None
        if face_down
        else visible_card_rule(
            key=key,
            card_set=_text(card_set, "card.set"),
            parameters=public_ability,
            visible_state=_visible_joker_state(key, game_state),
        )
    )
    return {
        "index": index,
        "key": None if face_down else key,
        "name": "Face-down card" if face_down else _text(name, "card.name"),
        "set": None if face_down else _text(card_set, "card.set"),
        "rule": public_rule,
        "facing": _text(facing, "card.facing"),
        "rank": rank,
        "suit": suit,
        "chips": chips,
        "enhancement": (
            None if face_down else _text(getattr(card, "center_key", ""), "center_key")
        ),
        "edition": public_edition,
        "seal": _optional_text(getattr(card, "seal", None)),
        "debuffed": bool(getattr(card, "debuff", False)),
        "cost": _exact_int(getattr(card, "cost", 0), "card.cost"),
        "sell_value": _exact_int(
            getattr(card, "sell_cost", 0),
            "card.sell_cost",
        ),
        "eternal": bool(getattr(card, "eternal", False)),
        "perishable": bool(getattr(card, "perishable", False)),
        "rental": bool(getattr(card, "rental", False)),
        "ability": public_ability,
    }


def _visible_joker_state(
    key: str,
    game_state: dict[str, Any] | None,
) -> dict[str, PolicyValue]:
    if game_state is None:
        return {}
    current_round = _trusted_dict(game_state, "current_round")
    source_key: str
    fields: tuple[str, ...]
    if key == "j_idol":
        source_key, fields = "idol_card", ("rank", "suit")
    elif key == "j_mail":
        source_key, fields = "mail_card", ("rank",)
    elif key == "j_ancient":
        source_key, fields = "ancient_card", ("suit",)
    elif key == "j_castle":
        source_key, fields = "castle_card", ("suit",)
    else:
        return {}
    value = current_round.get(source_key)
    if type(value) is not dict:
        raise RuntimeError(f"Jackdaw returned invalid {source_key}")
    return {
        f"target_{field}": _text(value.get(field), f"{source_key}.{field}")
        for field in fields
    }


def _deck(game_state: dict[str, Any]) -> dict[str, PolicyValue]:
    deck = _trusted_list(game_state, "deck")
    suit_counts: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    for card in deck:
        base = getattr(card, "base", None)
        if base is None:
            continue
        suit = _text(getattr(getattr(base, "suit", None), "value", ""), "suit")
        rank = _text(getattr(getattr(base, "rank", None), "value", ""), "rank")
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    return {
        "draw_pile": len(deck),
        "discard_pile": len(_trusted_list(game_state, "discard_pile")),
        "suit_counts": dict(sorted(suit_counts.items())),
        "rank_counts": dict(sorted(rank_counts.items())),
    }


def _poker_hands(game_state: dict[str, Any]) -> list[PolicyValue]:
    levels = game_state.get("hand_levels")
    if levels is None or not callable(getattr(levels, "get_state", None)):
        raise RuntimeError("Jackdaw returned invalid hand levels")
    result: list[PolicyValue] = []
    for name in _HAND_TYPES:
        state = levels.get_state(name)
        result.append(
            {
                "name": name,
                "level": _exact_int(getattr(state, "level", 0), "hand.level"),
                "chips": _exact_int(getattr(state, "chips", 0), "hand.chips"),
                "mult": _exact_int(getattr(state, "mult", 0), "hand.mult"),
                "played": _exact_int(
                    getattr(state, "played", 0),
                    "hand.played",
                ),
                "visible": bool(getattr(state, "visible", False)),
            }
        )
    return result


def _safe_public_value(value: object, *, depth: int = 0) -> PolicyValue:
    if value is None:
        return None
    if type(value) is bool:
        return bool(value)
    if type(value) is str:
        return str(value)
    if type(value) is bytes:
        return bytes(value)
    if type(value) is int:
        value = int(value)
        if -(2**63) <= value <= 2**64 - 1:
            return value
        return str(value)
    if type(value) is float:
        value = float(value)
        return value if math.isfinite(value) else None
    if depth >= 4:
        return None
    if type(value) is list:
        items = cast(list[object], value)
        return [_safe_public_value(item, depth=depth + 1) for item in items[:32]]
    if type(value) is tuple:
        items_tuple = cast(tuple[object, ...], value)
        return [_safe_public_value(item, depth=depth + 1) for item in items_tuple[:32]]
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        result: dict[str, PolicyValue] = {}
        for key in sorted(mapping, key=str)[:32]:
            if type(key) is str:
                result[key] = _safe_public_value(
                    mapping[key],
                    depth=depth + 1,
                )
        return result
    return None


def _vouchers(game_state: dict[str, Any]) -> list[PolicyValue]:
    keys = sorted(
        str(key)
        for key, owned in _optional_dict(
            game_state,
            "used_vouchers",
        ).items()
        if owned
    )
    result: list[PolicyValue] = []
    for key in keys:
        prototype = VOUCHERS.get(key)
        if prototype is None:
            raise RuntimeError(f"Jackdaw returned unknown owned Voucher {key}")
        result.append(
            {
                "key": key,
                "name": prototype.name,
                "rule": visible_card_rule(
                    key=key,
                    card_set="Voucher",
                    parameters={},
                ),
            }
        )
    return result


def _phase(game_state: dict[str, Any]) -> GamePhase:
    value = game_state.get("phase")
    if isinstance(value, GamePhase):
        return value
    if type(value) is str:
        return GamePhase(value)
    raise RuntimeError("Jackdaw returned an invalid phase")


def _trusted_dict(game_state: dict[str, Any], key: str) -> dict[str, Any]:
    value = game_state.get(key)
    if type(value) is not dict:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _optional_dict(game_state: dict[str, Any], key: str) -> dict[str, Any]:
    value = game_state.get(key, {})
    if type(value) is not dict:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _trusted_list(game_state: dict[str, Any], key: str) -> list[Any]:
    value = game_state.get(key)
    if type(value) is not list:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _optional_list(game_state: dict[str, Any], key: str) -> list[Any]:
    value = game_state.get(key, [])
    if type(value) is not list:
        raise RuntimeError(f"Jackdaw returned invalid {key}")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise RuntimeError(f"Jackdaw returned invalid {name}")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise RuntimeError("Jackdaw returned invalid optional text")
    return value


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"Jackdaw returned invalid {name}")
    return value


def _finite_number(value: object, name: str) -> int | float:
    if type(value) is int:
        return int(value)
    if type(value) is float and math.isfinite(value):
        return float(value)
    raise RuntimeError(f"Jackdaw returned invalid {name}")


__all__ = ["encode_observation", "replay_state"]
