"""Gymnasium environment for Balatro RL research.

Canonical usage::

    from jackdaw.env import BalatroEnvironment, DirectAdapter, balatro_game_spec

    spec = balatro_game_spec()
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    obs, mask, info = env.reset()
"""

from jackdaw.env.action_space import (
    NUM_ACTION_TYPES,
    ActionMask,
    ActionType,
    FactoredAction,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
    get_consumable_target_info,
)
from jackdaw.env.agents import Agent, RandomAgent
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.consumable_targets import (
    ConsumableTargetSpec,
    get_consumable_target_spec,
    get_valid_target_cards,
    validate_card_targets,
)
from jackdaw.env.game_interface import (
    BridgeAdapter,
    DirectAdapter,
    GameAdapter,
    GameState,
)
from jackdaw.env.game_spec import (
    GameActionMask,
    GameObservation,
    GameSpec,
)
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
    Observation,
    encode_observation,
)

__all__ = [
    "ActionMask",
    "ActionType",
    "Agent",
    "BalatroEnvironment",
    "BalatroGymnasiumEnv",
    "BridgeAdapter",
    "ConsumableTargetSpec",
    "D_CONSUMABLE",
    "D_GLOBAL",
    "D_JOKER",
    "D_PLAYING_CARD",
    "D_SHOP",
    "DirectAdapter",
    "FactoredAction",
    "GameActionMask",
    "GameAdapter",
    "GameObservation",
    "GameSpec",
    "GameState",
    "NUM_ACTION_TYPES",
    "NUM_CENTER_KEYS",
    "Observation",
    "RandomAgent",
    "balatro_game_spec",
    "encode_observation",
    "engine_action_to_factored",
    "factored_to_engine_action",
    "get_action_mask",
    "get_consumable_target_info",
    "get_consumable_target_spec",
    "get_valid_target_cards",
    "validate_card_targets",
]
