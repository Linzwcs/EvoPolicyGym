"""Gymnasium wrapper for BalatroEnvironment compatible with SB3's MaskablePPO.

Flattens the factored action space into a ``Discrete(MAX_ACTIONS)`` space by
enumerating all legal :class:`FactoredAction` instances each step.  Exposes an
``action_masks()`` method so ``sb3_contrib.MaskablePPO`` can mask illegal slots.

Example::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    obs, info = env.reset()
    mask = env.action_masks()
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.env.action_space import ActionType, get_consumable_target_info
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation

MAX_ACTIONS: int = 500
CARD_COMBO_BUDGET: int = 200

# Pre-compute entity layout from spec
_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
# Action types that take no targets at all
_SIMPLE_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
# Entity-only action types (entity target, no card select) — excludes UseConsumable
_ENTITY_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
# Card-only action types (card select, no entity) — PlayHand, Discard
_CARD_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)


def _subsample(items: list[Any], budget: int, rng: np.random.Generator) -> list[Any]:
    """Return *items* if within budget, else a random subsample."""
    if len(items) <= budget:
        return items
    indices = rng.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]


def _card_combos(
    legal_cards: np.ndarray,
    min_select: int,
    max_select: int,
) -> list[tuple[int, ...]]:
    """Enumerate all card-index combinations within the selection range."""
    upper = min(len(legal_cards), max_select)
    lower = min(min_select, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result


class BalatroGymnasiumEnv(gymnasium.Env):
    """Gymnasium wrapper exposing a flat Discrete action space with action masking.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter`.
    back_keys, stakes, max_steps, seed_prefix:
        Forwarded to :class:`BalatroEnvironment`.
    reward_shaping:
        If True, use dense multi-signal reward: blind-beaten bonuses scaled
        by ante, score-progress within blinds, efficient-clear bonuses, and
        reduced terminal rewards.  When False, use sparse ±1.0 at terminal.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
        reward_shaping: bool = False,
    ) -> None:
        super().__init__()
        self._inner = BalatroEnvironment(
            adapter_factory=adapter_factory,
            back_keys=back_keys,
            stakes=stakes,
            max_steps=max_steps,
            seed_prefix=seed_prefix,
        )
        self._reward_shaping = reward_shaping

        # Observation space
        obs_spaces: dict[str, spaces.Space] = {
            "global": spaces.Box(
                -np.inf, np.inf, shape=(_SPEC.global_feature_dim,), dtype=np.float32
            ),
        }
        max_counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            obs_spaces[name] = spaces.Box(
                -np.inf, np.inf, shape=(max_count, feat_dim), dtype=np.float32
            )
            max_counts.append(max_count)
        obs_spaces["entity_counts"] = spaces.Box(
            low=0,
            high=np.array(max_counts, dtype=np.float32),
            shape=(len(_ENTITY_INFO),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_spaces)

        # Action space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # Internal state
        self._action_table: list[FactoredAction] = []
        self._rng = np.random.default_rng()
        self._prev_ante: int = 1
        self._prev_round: int = 0
        self._prev_chips: int = 0
        self._episode_max_ante: int = 1
        self._episode_max_round: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = str(seed)

        game_obs, game_mask, info = self._inner.reset(**kwargs)
        self._prev_ante = self._inner.episode_ante
        self._prev_round = 0
        self._prev_chips = 0
        self._episode_max_ante = 1
        self._episode_max_round = 0
        self._action_table = self._enumerate_actions(game_mask, info)
        obs = self._build_obs(game_obs)
        return obs, {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        factored = self._action_table[action]
        game_obs, terminated, truncated, game_mask, info = self._inner.step(factored)

        reward = self._compute_reward(info, terminated, truncated)

        # Rebuild action table for next step
        if not (terminated or truncated):
            self._action_table = self._enumerate_actions(game_mask, info)
        else:
            self._action_table = []

        obs = self._build_obs(game_obs)
        step_info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            step_info["balatro/ante_reached"] = self._episode_max_ante
            step_info["balatro/rounds_beaten"] = self._episode_max_round
            step_info["balatro/won"] = self._inner.episode_won
        return obs, reward, terminated, truncated, step_info

    def action_masks(self) -> np.ndarray:
        """Return bool mask of shape ``(MAX_ACTIONS,)`` for MaskablePPO."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: len(self._action_table)] = True
        return mask

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(self, info: dict[str, Any], terminated: bool, truncated: bool) -> float:
        """Compute step reward from game state deltas."""
        if not self._reward_shaping:
            if terminated or truncated:
                return 1.0 if self._inner.episode_won else -1.0
            return 0.0

        gs: dict[str, Any] = info.get("raw_state", {})
        phase = gs.get("phase")

        # Step cost — discourages stalling; doubled in shop phase
        reward = -0.002 if phase == "shop" else -0.001

        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)
        ante_scale = ante / 8.0

        # 1. Blind beaten: round increased → +0.15 * ante_scale
        if round_num > self._prev_round:
            reward += 0.15 * ante_scale
            # 2. Boss blind beaten (ante increased) → extra +0.1 * ante_scale
            if ante > self._prev_ante:
                reward += 0.1 * ante_scale
            # 3. Efficient clear: hands remaining bonus
            hands_left = gs.get("current_round", {}).get("hands_left", 0)
            reward += 0.01 * hands_left

        # 4. Score progress within a blind: chips gained toward target
        blind = gs.get("blind")
        blind_target = getattr(blind, "chips", 0) if blind is not None else 0
        if blind_target > 0 and chips > self._prev_chips:
            chip_delta = chips - self._prev_chips
            reward += 0.02 * min(chip_delta / blind_target, 1.0)

        # 5. Terminal
        if terminated or truncated:
            reward += 0.5 if self._inner.episode_won else -0.2

        # Update trackers
        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

        return reward

    def _build_obs(self, game_obs: GameObservation) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {
            "global": game_obs.global_context.astype(np.float32),
        }
        counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            arr = game_obs.entities.get(name)
            padded = np.zeros((max_count, feat_dim), dtype=np.float32)
            if arr is not None and arr.shape[0] > 0:
                n = min(arr.shape[0], max_count)
                padded[:n] = arr[:n]
                counts.append(n)
            else:
                counts.append(0)
            obs[name] = padded
        obs["entity_counts"] = np.array(counts, dtype=np.float32)
        return obs

    def _enumerate_actions(
        self, mask: GameActionMask, info: dict[str, Any]
    ) -> list[FactoredAction]:
        actions: list[FactoredAction] = []
        type_mask = mask.type_mask

        for t in range(len(type_mask)):
            if not type_mask[t]:
                continue

            # --- Simple (no targets) ---
            if t in _SIMPLE_TYPES:
                actions.append(FactoredAction(action_type=t))

            # --- Entity-only ---
            elif t in _ENTITY_ONLY_TYPES:
                if t in mask.entity_masks:
                    legal = np.nonzero(mask.entity_masks[t])[0]
                    for idx in legal:
                        actions.append(FactoredAction(action_type=t, entity_target=int(idx)))

            # --- Card-only (PlayHand / Discard) ---
            elif t in _CARD_ONLY_TYPES:
                legal_cards = np.nonzero(mask.card_mask)[0]
                combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)
                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                for combo in combos:
                    actions.append(FactoredAction(action_type=t, card_target=combo))

            # --- UseConsumable (entity + optional cards) ---
            elif t == ActionType.UseConsumable:
                if t in mask.entity_masks:
                    legal_entities = np.nonzero(mask.entity_masks[t])[0]
                    raw_consumables = info.get("raw_state", {}).get("consumables", [])
                    for c_idx in legal_entities:
                        c_idx_int = int(c_idx)
                        if c_idx_int < len(raw_consumables):
                            card = raw_consumables[c_idx_int]
                            min_cards, max_cards, needs = get_consumable_target_info(card)
                            if needs:
                                legal_cards = np.nonzero(mask.card_mask)[0]
                                combos = _card_combos(legal_cards, min_cards, max_cards)
                                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                                for combo in combos:
                                    actions.append(
                                        FactoredAction(
                                            action_type=t,
                                            entity_target=c_idx_int,
                                            card_target=combo,
                                        )
                                    )
                            else:
                                actions.append(
                                    FactoredAction(action_type=t, entity_target=c_idx_int)
                                )

        # Subsample to budget if needed
        if len(actions) > MAX_ACTIONS:
            actions = _subsample(actions, MAX_ACTIONS, self._rng)

        return actions
