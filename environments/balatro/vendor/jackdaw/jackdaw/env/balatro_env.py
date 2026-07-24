"""BalatroEnvironment: GameEnvironment implementation for Balatro.

Wraps a :class:`GameAdapter` together with observation encoding and action
masking behind the game-agnostic :class:`GameEnvironment` protocol.  This is
the bridge between the Balatro engine and the generic RL training loop.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from jackdaw.env.action_space import (
    ActionMask,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import GameActionMask, GameObservation, GameSpec
from jackdaw.env.observation import encode_observation


def _compute_shop_splits(gs: dict[str, Any]) -> tuple[int, int, int]:
    """Compute ``(n_shop_cards, n_vouchers, n_boosters)`` from a game state."""
    return (
        len(gs.get("shop_cards", [])),
        len(gs.get("shop_vouchers", [])),
        len(gs.get("shop_boosters", [])),
    )


class BalatroEnvironment:
    """Game-agnostic environment wrapper for Balatro.

    Implements the :class:`~jackdaw.env.game_spec.GameEnvironment` protocol
    so the generic RL pipeline can interact with Balatro without importing
    any Balatro-specific modules.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter` instance.
    back_keys:
        List of deck back keys to sample from on each reset.
    stakes:
        List of stakes to sample from on each reset.
    max_steps:
        Maximum steps before truncation.
    seed_prefix:
        Prefix for deterministic seed generation.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
    ) -> None:
        self._adapter_factory = adapter_factory
        self._adapter = adapter_factory()
        self._back_keys = back_keys or ["b_red"]
        self._stakes = stakes or [1]
        self._max_steps = max_steps
        self._seed_prefix = seed_prefix
        self._spec = balatro_game_spec()

        self._episode_count = 0
        self._step_count = 0

        # Episode tracking
        self.episode_length: int = 0
        self.episode_won: bool = False
        self.episode_ante: int = 1

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def reset(
        self,
        **kwargs: object,
    ) -> tuple[GameObservation, GameActionMask, dict[str, object]]:
        """Reset the environment and return initial observation + mask."""
        seed = str(kwargs.get("seed", f"{self._seed_prefix}_{self._episode_count}"))
        self._episode_count += 1
        self._step_count = 0
        self.episode_length = 0
        self.episode_won = False
        self.episode_ante = 1

        self._adapter = self._adapter_factory()
        back_key = (
            random.choice(self._back_keys) if len(self._back_keys) > 1 else self._back_keys[0]
        )
        stake = random.choice(self._stakes) if len(self._stakes) > 1 else self._stakes[0]
        self._adapter.reset(back_key, stake, seed)

        gs = self._adapter.raw_state
        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        game_obs = obs.to_game_observation()
        game_mask = _action_mask_to_game(mask)
        info: dict[str, object] = {
            "raw_state": gs,
            "shop_splits": _compute_shop_splits(gs),
            "observation": obs,
            "action_mask": mask,
        }
        return game_obs, game_mask, info

    def step(
        self,
        action: FactoredAction,
    ) -> tuple[GameObservation, bool, bool, GameActionMask, dict[str, object]]:
        """Execute an action and return (obs, terminated, truncated, mask, info)."""
        gs_prev = self._adapter.raw_state
        engine_action = factored_to_engine_action(action, gs_prev)
        self._adapter.step(engine_action)

        gs = self._adapter.raw_state
        self.episode_length += 1
        self._step_count += 1

        terminated = self._adapter.done
        truncated = self._step_count >= self._max_steps

        # Track episode stats
        rr = gs.get("round_resets", {})
        self.episode_ante = rr.get("ante", 1)
        self.episode_won = self._adapter.won

        obs = encode_observation(gs)
        mask = get_action_mask(gs)
        game_obs = obs.to_game_observation()
        game_mask = _action_mask_to_game(mask)
        info: dict[str, object] = {
            "raw_state": gs,
            "prev_raw_state": gs_prev,
            "shop_splits": _compute_shop_splits(gs),
            "observation": obs,
            "action_mask": mask,
        }
        return game_obs, terminated, truncated, game_mask, info


def _action_mask_to_game(mask: ActionMask) -> GameActionMask:
    """Convert a Balatro ActionMask to a game-agnostic GameActionMask."""
    return GameActionMask(
        type_mask=mask.type_mask,
        card_mask=mask.card_mask,
        entity_masks=mask.entity_masks,
        min_card_select=mask.min_card_select,
        max_card_select=mask.max_card_select,
    )
