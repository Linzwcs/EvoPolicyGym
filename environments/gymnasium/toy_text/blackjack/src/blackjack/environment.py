"""One fresh Gymnasium Blackjack-v1 Environment per Episode."""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import BlackjackConfig

_ACTIONS = frozenset({0, 1})
_MAX_EPISODE_STEPS = 32


class BlackjackEnvironment:
    """The seeded strict adapter around configured Blackjack-v1."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: BlackjackConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BlackjackConfig:
            raise TypeError("config must be BlackjackConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Blackjack configuration belongs in BlackjackConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(
                "Blackjack-v1",
                natural=config.natural,
                sab=config.sab,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._observation: dict[str, PolicyValue] | None = None
        self._initial_player_natural = False
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        self._observation = public_observation
        self._initial_player_natural = (
            _integer_field(public_observation, "player_sum") == 21
            and _boolean_field(public_observation, "usable_ace")
        )
        self._started = True
        return public_observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()

        previous_observation = self._observation
        if previous_observation is None:
            raise RuntimeError("Blackjack observation is unavailable")
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Blackjack returned invalid termination flags")
        self._steps += 1
        if self._steps >= _MAX_EPISODE_STEPS and not terminated:
            truncated = True
        public_observation = _observation(observation)
        public_reward = _reward(reward, config=self._config)
        metrics = _transition_metrics(
            previous_observation,
            public_observation,
            action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            initial_player_natural=self._initial_player_natural,
            step_count=self._steps,
        )
        self._observation = public_observation
        self._done = terminated or truncated
        return Step(
            observation=public_observation,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(value: object) -> dict[str, PolicyValue]:
    if type(value) is not tuple or len(value) != 3:
        raise RuntimeError("Blackjack returned an invalid observation")
    player_sum = _integer(value[0], name="player sum")
    dealer_showing = _integer(value[1], name="dealer card")
    usable_ace = _boolean(value[2], name="usable-ace flag")
    if not 0 <= player_sum <= 31:
        raise RuntimeError("Blackjack returned an invalid player sum")
    if not 1 <= dealer_showing <= 10:
        raise RuntimeError("Blackjack returned an invalid dealer card")
    return {
        "player_sum": player_sum,
        "dealer_showing": dealer_showing,
        "usable_ace": usable_ace,
    }


def _integer(value: object, *, name: str) -> int:
    if type(value) is bool:
        raise RuntimeError(f"Blackjack returned an invalid {name}")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise RuntimeError(
            f"Blackjack returned an invalid {name}"
        ) from error


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    integer = _integer(value, name=name)
    if integer not in {0, 1}:
        raise RuntimeError(f"Blackjack returned an invalid {name}")
    return bool(integer)


def _reward(value: object, *, config: BlackjackConfig) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError("Blackjack returned an invalid reward")
    reward = float(cast(int | float, value))
    if not math.isfinite(reward):
        raise RuntimeError("Blackjack returned a non-finite reward")
    allowed = {-1.0, 0.0, 1.0}
    if config.natural and not config.sab:
        allowed.add(1.5)
    if reward not in allowed:
        raise RuntimeError("Blackjack returned an unknown reward")
    return reward


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    action: int,
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    initial_player_natural: bool,
    step_count: int,
) -> dict[str, PolicyValue]:
    previous_sum = _integer_field(previous, "player_sum")
    current_sum = _integer_field(current, "player_sum")
    previous_usable_ace = _boolean_field(previous, "usable_ace")
    current_usable_ace = _boolean_field(current, "usable_ace")
    if _integer_field(previous, "dealer_showing") != _integer_field(
        current,
        "dealer_showing",
    ):
        raise RuntimeError("Blackjack changed the dealer's showing card")

    possible_drawn_cards: list[PolicyValue] = []
    if action == 1:
        for card in range(1, 11):
            if _after_draw(previous_sum, previous_usable_ace, card) == (
                current_sum,
                current_usable_ace,
            ):
                possible_drawn_cards.append(card)
        if not possible_drawn_cards:
            raise RuntimeError("Blackjack hit transition is inconsistent with its observation")
        if terminated:
            if reward != -1.0 or current_sum <= 21:
                raise RuntimeError("Blackjack bust semantics drifted")
            event = "hit_bust"
        else:
            if reward != 0.0 or current_sum > 21:
                raise RuntimeError("Blackjack continuing-hit semantics drifted")
            event = "hit_continue"
    else:
        if current != previous or not terminated:
            raise RuntimeError("Blackjack stick semantics drifted")
        event = {
            -1.0: "stick_loss",
            0.0: "stick_draw",
            1.0: "stick_win",
            1.5: "stick_natural_win",
        }.get(reward, "")
        if not event:
            raise RuntimeError("Blackjack stick reward drifted")

    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "requested_action": "stick" if action == 0 else "hit",
        "event": event,
        "initial_player_natural": initial_player_natural,
        "player_sum_before": previous_sum,
        "player_sum_after": current_sum,
        "player_sum_change": current_sum - previous_sum,
        "usable_ace_before": previous_usable_ace,
        "usable_ace_after": current_usable_ace,
        "usable_ace_changed": previous_usable_ace != current_usable_ace,
        "player_bust": current_sum > 21,
    }
    if action == 1:
        metrics["possible_drawn_card_values"] = possible_drawn_cards
    reasons: list[str] = []
    if terminated:
        reasons.append(event)
    if truncated:
        reasons.append("time_limit")
    if reasons:
        metrics["terminal_reason"] = "+".join(reasons)
    return metrics


def _after_draw(player_sum: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    if usable_ace:
        hard_sum = player_sum - 10 + card
        return (hard_sum + 10, True) if hard_sum + 10 <= 21 else (hard_sum, False)
    if card == 1 and player_sum + 11 <= 21:
        return player_sum + 11, True
    return player_sum + card, False


def _integer_field(observation: dict[str, PolicyValue], name: str) -> int:
    value = observation.get(name)
    if type(value) is not int:
        raise RuntimeError(f"Blackjack returned invalid {name}")
    return value


def _boolean_field(observation: dict[str, PolicyValue], name: str) -> bool:
    value = observation.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"Blackjack returned invalid {name}")
    return value


__all__ = ["BlackjackEnvironment"]
