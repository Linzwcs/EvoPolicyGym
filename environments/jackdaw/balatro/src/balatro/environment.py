"""One fresh deterministic Jackdaw game state per Episode."""

from __future__ import annotations

from typing import Any

from evopolicygym.authoring import EpisodeSpec, Step
from evopolicygym.policy import PolicyValue
from jackdaw.engine import GamePhase, initialize_run
from jackdaw.engine import step as jackdaw_step
from jackdaw.engine.card import reset_sort_id_counter

from .actions import decode_action
from .observation import encode_observation

MAX_EPISODE_STEPS = 2_048
WIN_BONUS = 1_000
_BACK_KEY = "b_red"
_STAKE = 1
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

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario != {"back": _BACK_KEY, "stake": _STAKE}:
            raise ValueError("episode scenario is not supported")
        self._environment_seed = episode.environment_seed
        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        reset_sort_id_counter()
        self._state = initialize_run(
            _BACK_KEY,
            _STAKE,
            f"EPG{self._environment_seed:016X}",
            challenge=_CONTENT_PROFILE_CHALLENGE,
        )
        self._started = True
        return encode_observation(self._state, step_count=0)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._state is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        engine_action = decode_action(action, self._state)
        previous_score = _run_score(self._state)
        jackdaw_step(self._state, engine_action)
        self._steps += 1
        score = _run_score(self._state)
        terminated = _is_terminal(self._state)
        truncated = self._steps >= MAX_EPISODE_STEPS and not terminated
        self._done = terminated or truncated
        return Step(
            observation=encode_observation(
                self._state,
                step_count=self._steps,
            ),
            reward=float(score - previous_score),
            terminated=terminated,
            truncated=truncated,
            metrics=_metrics(self._state),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._state = None
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


__all__ = [
    "CONTENT_PROFILE",
    "EXCLUDED_TAG_KEYS",
    "EXCLUDED_VOUCHER_KEYS",
    "BalatroEnvironment",
    "MAX_EPISODE_STEPS",
    "WIN_BONUS",
]
