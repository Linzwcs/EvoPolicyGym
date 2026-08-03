"""One fresh deterministic Jackdaw game state per Episode."""

from __future__ import annotations

from typing import Any

from evopolicygym.authoring import EpisodeSpec, Step
from evopolicygym.policy import PolicyValue
from jackdaw.engine import GamePhase, initialize_run
from jackdaw.engine import step as jackdaw_step
from jackdaw.engine.card import reset_sort_id_counter

from .actions import decode_action
from .config import BalatroConfig
from .observation import encode_observation

MAX_EPISODE_STEPS = 2_048
WIN_BONUS = 1_000
CONTENT_PROFILE = "jackdaw-active-content-v1"
EXCLUDED_TAG_KEYS = (
    "tag_rare",
    "tag_uncommon",
    "tag_voucher",
)
EXCLUDED_VOUCHER_KEYS = (
    "v_omen_globe",
    "v_telescope",
    "v_observatory",
    "v_directors_cut",
    "v_retcon",
)
_CONTENT_PROFILE_CHALLENGE: dict[str, Any] = {
    "id": CONTENT_PROFILE,
    "restrictions": {
        "banned_cards": [{"id": key} for key in EXCLUDED_VOUCHER_KEYS],
        "banned_tags": [{"id": key} for key in EXCLUDED_TAG_KEYS],
        "banned_other": [],
    },
}


class BalatroEnvironment:
    """A white-stake Red Deck run powered by Jackdaw's trusted engine."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: BalatroConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BalatroConfig:
            raise TypeError("config must be BalatroConfig")
        if episode.scenario is not None:
            raise ValueError("the Balatro profile does not use an Episode scenario")
        self._environment_seed = episode.environment_seed
        self._config = config
        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._started = False
        self._done = False
        self._closed = False
        self._last_observation: dict[str, PolicyValue] | None = None
        self._action_counts: dict[str, int] = {}
        self._hand_type_counts: dict[str, int] = {}
        self._purchased_card_types: dict[str, int] = {}
        self._pack_pick_types: dict[str, int] = {}
        self._cards_played = 0
        self._cards_discarded = 0
        self._total_hand_score = 0
        self._best_hand_score = 0
        self._total_money_gained = 0
        self._total_money_spent = 0
        self._peak_money = 0
        self._total_purchase_cost = 0
        self._total_sale_value = 0
        self._best_blind_progress = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        reset_sort_id_counter()
        self._state = initialize_run(
            self._config.deck,
            self._config.stake,
            f"EPG{self._environment_seed:016X}",
            challenge=_CONTENT_PROFILE_CHALLENGE,
        )
        observation = encode_observation(self._state, step_count=0)
        self._peak_money = _observation_int(observation, "resources", "money")
        self._last_observation = observation
        self._started = True
        return observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._state is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        previous_observation = self._last_observation
        if previous_observation is None:
            raise RuntimeError("Environment has no previous observation")
        engine_action = decode_action(action, self._state)
        action_kind = _action_kind(action)
        phase_before = _observation_text(previous_observation, "phase")
        previous_score = _run_score(self._state)
        jackdaw_step(self._state, engine_action)
        self._steps += 1
        score = _run_score(self._state)
        terminated = _is_terminal(self._state)
        truncated = self._steps >= MAX_EPISODE_STEPS and not terminated
        self._done = terminated or truncated
        observation = encode_observation(
            self._state,
            step_count=self._steps,
        )
        reward = score - previous_score
        self._update_diagnostics(
            action,
            action_kind=action_kind,
            previous=previous_observation,
            current=observation,
        )
        self._last_observation = observation
        return Step(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            metrics=self._step_metrics(
                action,
                action_kind=action_kind,
                phase_before=phase_before,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            ),
        )

    def _update_diagnostics(
        self,
        action: PolicyValue,
        *,
        action_kind: str,
        previous: dict[str, PolicyValue],
        current: dict[str, PolicyValue],
    ) -> None:
        self._action_counts[action_kind] = self._action_counts.get(action_kind, 0) + 1
        selected_cards = _selected_card_count(action)
        if action_kind == "play_hand":
            self._cards_played += selected_cards
            hand_type = _observation_text(current, "last_hand", "hand_type")
            if hand_type:
                self._hand_type_counts[hand_type] = self._hand_type_counts.get(hand_type, 0) + 1
            hand_score = _observation_int(current, "last_hand", "total")
            self._total_hand_score += hand_score
            self._best_hand_score = max(self._best_hand_score, hand_score)
        elif action_kind == "discard":
            self._cards_discarded += selected_cards

        money_before = _observation_int(previous, "resources", "money")
        money_after = _observation_int(current, "resources", "money")
        money_delta = money_after - money_before
        if money_delta > 0:
            self._total_money_gained += money_delta
        elif money_delta < 0:
            self._total_money_spent -= money_delta
        self._peak_money = max(self._peak_money, money_after)

        transaction = _transaction_card(previous, action, action_kind)
        if transaction is not None:
            card_type = _policy_text(transaction, "set")
            if action_kind in {"buy_card", "redeem_voucher", "open_booster"}:
                self._purchased_card_types[card_type] = (
                    self._purchased_card_types.get(card_type, 0) + 1
                )
                self._total_purchase_cost += _policy_int(transaction, "cost")
            elif action_kind in {"sell_joker", "sell_consumable"}:
                self._total_sale_value += _policy_int(transaction, "sell_value")
            elif action_kind == "pick_pack_card":
                self._pack_pick_types[card_type] = self._pack_pick_types.get(card_type, 0) + 1

        _, _, _, blind_progress = _blind_progress(current)
        self._best_blind_progress = max(
            self._best_blind_progress,
            blind_progress,
        )

    def _step_metrics(
        self,
        action: PolicyValue,
        *,
        action_kind: str,
        phase_before: str,
        observation: dict[str, PolicyValue],
        reward: int,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, PolicyValue]:
        metrics = _metrics(self._state_or_raise())
        blind_chips, blind_target, chips_to_target, blind_progress = _blind_progress(observation)
        last_hand_score = _observation_int(observation, "last_hand", "total")
        last_hand_type = _observation_text(observation, "last_hand", "hand_type")
        hand_plays = self._action_counts.get("play_hand", 0)
        metrics.update(
            {
                "step_count": self._steps,
                "action_kind": action_kind,
                "phase_before": phase_before,
                "phase_after": _observation_text(observation, "phase"),
                "selected_card_count": _selected_card_count(action),
                "benchmark_reward": reward,
                "reward_event": reward != 0,
                "blind_cleared_this_step": reward > 0,
                "money": _observation_int(observation, "resources", "money"),
                "hands_left": _observation_int(
                    observation,
                    "resources",
                    "hands_left",
                ),
                "discards_left": _observation_int(
                    observation,
                    "resources",
                    "discards_left",
                ),
                "blind_chips": blind_chips,
                "blind_target_chips": blind_target,
                "chips_to_target": chips_to_target,
                "blind_progress_fraction": blind_progress,
                "final_blind_progress_fraction": blind_progress,
                "best_blind_progress_fraction": self._best_blind_progress,
                "last_hand_type": last_hand_type,
                "last_hand_level": _observation_int(
                    observation,
                    "last_hand",
                    "level",
                ),
                "last_hand_chips": _observation_number(
                    observation,
                    "last_hand",
                    "chips",
                ),
                "last_hand_mult": _observation_number(
                    observation,
                    "last_hand",
                    "mult",
                ),
                "last_hand_score": last_hand_score,
                "best_hand_score": self._best_hand_score,
                "mean_hand_score": (self._total_hand_score / hand_plays if hand_plays else 0.0),
                "cards_played": self._cards_played,
                "cards_discarded": self._cards_discarded,
                "total_money_gained": self._total_money_gained,
                "total_money_spent": self._total_money_spent,
                "peak_money": self._peak_money,
                "total_purchase_cost": self._total_purchase_cost,
                "total_sale_value": self._total_sale_value,
                "owned_jokers": len(_observation_list(observation, "jokers")),
                "owned_consumables": len(_observation_list(observation, "consumables")),
                "owned_vouchers": len(_observation_list(observation, "vouchers")),
                "awarded_tags": len(_observation_list(observation, "tags")),
                "action_counts": _policy_counts(self._action_counts),
                "hand_type_counts": _policy_counts(self._hand_type_counts),
                "purchased_card_types": _policy_counts(self._purchased_card_types),
                "pack_pick_types": _policy_counts(self._pack_pick_types),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return metrics

    def _state_or_raise(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Environment has no state")
        return self._state

    def close(self) -> None:
        if self._closed:
            return
        self._state = None
        self._last_observation = None
        self._closed = True


def _is_terminal(game_state: dict[str, Any]) -> bool:
    phase = game_state.get("phase")
    if phase == GamePhase.GAME_OVER:
        return True
    return bool(game_state.get("won", False)) and phase == GamePhase.SHOP


def _run_score(game_state: dict[str, Any]) -> int:
    rounds = game_state.get("round", 0)
    if type(rounds) is not int:
        raise RuntimeError("Jackdaw returned invalid round")
    return rounds + (WIN_BONUS if game_state.get("won", False) else 0)


def _metrics(game_state: dict[str, Any]) -> dict[str, PolicyValue]:
    round_resets = game_state.get("round_resets")
    if type(round_resets) is not dict:
        raise RuntimeError("Jackdaw returned invalid round_resets")
    ante = round_resets.get("ante", 1)
    rounds = game_state.get("round", 0)
    if type(ante) is not int or type(rounds) is not int:
        raise RuntimeError("Jackdaw returned invalid progress")
    return {
        "run_score": _run_score(game_state),
        "won": bool(game_state.get("won", False)),
        "ante": ante,
        "rounds_cleared": rounds,
    }


def _action_kind(action: PolicyValue) -> str:
    if type(action) is not dict:
        raise RuntimeError("validated Balatro Action is not an object")
    value = action.get("kind")
    if type(value) is not str:
        raise RuntimeError("validated Balatro Action has no kind")
    return value


def _selected_card_count(action: PolicyValue) -> int:
    if type(action) is not dict:
        return 0
    values = action.get("card_indices")
    return len(values) if type(values) is list else 0


def _blind_progress(
    observation: dict[str, PolicyValue],
) -> tuple[int, int, int, float]:
    chips = _observation_int(observation, "resources", "chips")
    blind = observation.get("blind")
    if type(blind) is not dict:
        return chips, 0, 0, 0.0
    target = _policy_int(blind, "target_chips")
    remaining = max(target - chips, 0)
    progress = min(chips / target, 1.0) if target > 0 else 0.0
    return chips, target, remaining, progress


def _transaction_card(
    observation: dict[str, PolicyValue],
    action: PolicyValue,
    action_kind: str,
) -> dict[str, PolicyValue] | None:
    if type(action) is not dict:
        return None
    target = action.get("target_index")
    if type(target) is not int:
        return None
    area: PolicyValue
    if action_kind == "buy_card":
        area = _observation_dict(observation, "shop").get("cards")
    elif action_kind == "redeem_voucher":
        area = _observation_dict(observation, "shop").get("vouchers")
    elif action_kind == "open_booster":
        area = _observation_dict(observation, "shop").get("boosters")
    elif action_kind == "sell_joker":
        area = observation.get("jokers")
    elif action_kind == "sell_consumable":
        area = observation.get("consumables")
    elif action_kind == "pick_pack_card":
        area = _observation_dict(observation, "pack").get("cards")
    else:
        return None
    if type(area) is not list or not 0 <= target < len(area):
        raise RuntimeError("validated Balatro Action target is not public")
    card = area[target]
    if type(card) is not dict:
        raise RuntimeError("Balatro transaction target is invalid")
    return card


def _observation_dict(
    observation: dict[str, PolicyValue],
    key: str,
) -> dict[str, PolicyValue]:
    value = observation.get(key)
    if type(value) is not dict:
        raise RuntimeError(f"Balatro observation {key} is invalid")
    return value


def _observation_list(
    observation: dict[str, PolicyValue],
    key: str,
) -> list[PolicyValue]:
    value = observation.get(key)
    if type(value) is not list:
        raise RuntimeError(f"Balatro observation {key} is invalid")
    return value


def _observation_int(
    observation: dict[str, PolicyValue],
    group: str,
    key: str,
) -> int:
    return _policy_int(_observation_dict(observation, group), key)


def _observation_number(
    observation: dict[str, PolicyValue],
    group: str,
    key: str,
) -> int | float:
    value = _observation_dict(observation, group).get(key)
    if type(value) is int:
        return int(value)
    if type(value) is float:
        return float(value)
    raise RuntimeError(f"Balatro observation {group}.{key} is invalid")


def _observation_text(
    observation: dict[str, PolicyValue],
    group: str,
    key: str | None = None,
) -> str:
    value = observation.get(group)
    if key is not None:
        if type(value) is not dict:
            raise RuntimeError(f"Balatro observation {group} is invalid")
        value = value.get(key)
    if type(value) is not str:
        raise RuntimeError(f"Balatro observation {group} is invalid")
    return value


def _policy_int(value: dict[str, PolicyValue], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise RuntimeError(f"Balatro public value {key} is invalid")
    return item


def _policy_text(value: dict[str, PolicyValue], key: str) -> str:
    item = value.get(key)
    if type(item) is not str:
        raise RuntimeError(f"Balatro public value {key} is invalid")
    return item


def _policy_counts(values: dict[str, int]) -> dict[str, PolicyValue]:
    return {key: value for key, value in sorted(values.items())}


__all__ = [
    "CONTENT_PROFILE",
    "EXCLUDED_TAG_KEYS",
    "EXCLUDED_VOUCHER_KEYS",
    "BalatroEnvironment",
    "MAX_EPISODE_STEPS",
    "WIN_BONUS",
]
