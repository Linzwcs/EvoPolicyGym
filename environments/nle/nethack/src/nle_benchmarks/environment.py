"""One fresh, seeded NLE NetHackScore instance per Episode."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping, Sequence
from typing import Protocol, SupportsFloat, SupportsIndex, cast

import nle
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from nle import nethack
from nle.env.tasks import TASK_ACTIONS, NetHackScore

from .config import NetHackConfig
from .constants import (
    ACTION_MEANINGS,
    AGGREGATE_FEEDBACK_SCOPE,
    CHARACTER,
    FEEDBACK_SCOPE_KEY,
    NETHACK_OPTIONS,
    OBSERVATION_KEYS,
    PENALTY_MODE,
    PENALTY_STEP,
    PENALTY_TIME,
    PUBLIC_FEEDBACK_SCOPE,
    RAW_ACTIONS,
    UPSTREAM_VERSION,
)
from .observation import project_observation

_SEED_DOMAIN = b"evopolicygym-nle-nethack/upstream-seeds/v1\0"


class _ActionSpace(Protocol):
    n: object


class _UpstreamEnvironment(Protocol):
    actions: Sequence[object]
    action_space: _ActionSpace

    def seed(
        self,
        core: int,
        disp: int,
        reseed: bool,
        lgen: int,
    ) -> tuple[int, int, bool, int]: ...

    def reset(self) -> tuple[object, object]: ...

    def step(self, action: int) -> tuple[object, object, object, object, object]: ...

    def close(self) -> None: ...


class NetHackEnvironment:
    """Strict adapter around the canonical NLE score task."""

    def __init__(self, episode: EpisodeSpec, *, config: NetHackConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not NetHackConfig:
            raise TypeError("config must be NetHackConfig")
        _validate_scenario(episode.scenario)
        seeds = _upstream_seeds(episode.environment_seed)
        self._environment = _make_upstream(config, seeds=seeds)
        self._max_episode_steps = config.max_episode_steps
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._max_game_score = 0
        self._max_depth = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, information = self._environment.reset()
        public = project_observation(observation)
        ascended, end_status = _information(information)
        if ascended or end_status != 0:
            raise RuntimeError("NLE reset into a terminal state")
        self._update_progress(public)
        self._started = True
        return public

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or not 0 <= action < len(ACTION_MEANINGS):
            raise InvalidAction()

        observation, reward, terminated, truncated, information = (
            self._environment.step(action)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("NLE returned invalid termination flags")
        public = project_observation(observation)
        self._update_progress(public)
        ascended, end_status = _information(information)
        self._steps += 1
        reached_horizon = self._steps >= self._max_episode_steps
        if reached_horizon and end_status == -1:
            # NLE 1.3.0 increments its private counter before computing
            # end_status, but returns the truncation flag computed before that
            # increment. Its own horizon therefore arrives as
            # (terminated=True, truncated=False, ABORTED). Translate that
            # upstream quirk into the authoring SDK's termination semantics.
            terminated = False
            truncated = True
        elif reached_horizon and not terminated:
            truncated = True
        self._done = terminated or truncated
        stats = _stats(public)
        return Step(
            observation=public,
            reward=_number(reward),
            terminated=terminated,
            truncated=truncated,
            metrics={
                "game_score": stats["score"],
                "max_game_score": self._max_game_score,
                "depth": stats["depth"],
                "max_depth": self._max_depth,
                "experience_level": stats["experience_level"],
                "dungeon_level": stats["dungeon_level"],
                "hit_points": stats["hit_points"],
                "turn": stats["turn"],
                "ascended": ascended,
                "end_status": end_status,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._environment.close()
        finally:
            self._closed = True

    def _update_progress(self, observation: PolicyValue) -> None:
        stats = _stats(observation)
        score = stats["score"]
        depth = stats["depth"]
        if type(score) is not int or type(depth) is not int:
            raise RuntimeError("NLE public progress values are invalid")
        self._max_game_score = max(self._max_game_score, score)
        self._max_depth = max(self._max_depth, depth)


def _make_upstream(
    config: NetHackConfig,
    *,
    seeds: tuple[int, int, int],
) -> _UpstreamEnvironment:
    if nle.__version__ != UPSTREAM_VERSION:
        raise RuntimeError("installed NLE version changed incompatibly")
    if tuple(nethack.NETHACKOPTIONS) != NETHACK_OPTIONS:
        raise RuntimeError("NLE default options changed incompatibly")
    if tuple(operator.index(cast(SupportsIndex, action)) for action in TASK_ACTIONS) != RAW_ACTIONS:
        raise RuntimeError("NLE task Actions changed incompatibly")

    environment = cast(
        _UpstreamEnvironment,
        NetHackScore(
            save_ttyrec_every=0,
            savedir=None,
            character=CHARACTER,
            max_episode_steps=config.max_episode_steps,
            observation_keys=OBSERVATION_KEYS,
            actions=TASK_ACTIONS,
            options=NETHACK_OPTIONS,
            wizard=False,
            allow_all_yn_questions=False,
            allow_all_modes=False,
            spawn_monsters=True,
            render_mode=None,
            fix_moon_phase=True,
            penalty_mode=PENALTY_MODE,
            penalty_step=PENALTY_STEP,
            penalty_time=PENALTY_TIME,
        ),
    )
    try:
        if tuple(
            operator.index(cast(SupportsIndex, action))
            for action in environment.actions
        ) != RAW_ACTIONS:
            raise RuntimeError("NLE Environment Actions changed incompatibly")
        if environment.action_space.n != len(ACTION_MEANINGS):
            raise RuntimeError("NLE action space changed incompatibly")
        applied = environment.seed(seeds[0], seeds[1], False, seeds[2])
        if applied != (seeds[0], seeds[1], False, seeds[2]):
            raise RuntimeError("NLE did not apply the requested seeds")
    except Exception:
        environment.close()
        raise
    return environment


def _upstream_seeds(environment_seed: int) -> tuple[int, int, int]:
    seeds: list[int] = []
    for label in (b"core", b"disp", b"level"):
        digest = hashlib.sha256()
        digest.update(_SEED_DOMAIN)
        digest.update(label)
        digest.update(b"\0")
        digest.update(environment_seed.to_bytes(8, "big"))
        seeds.append(int.from_bytes(digest.digest()[:8], "big") & (2**63 - 1))
    return (seeds[0], seeds[1], seeds[2])


def _validate_scenario(value: PolicyValue) -> None:
    if value is None:
        return
    if type(value) is not dict or set(value) != {FEEDBACK_SCOPE_KEY}:
        raise ValueError("NetHack configuration belongs in NetHackConfig")
    scope = value[FEEDBACK_SCOPE_KEY]
    if type(scope) is not str or scope not in {
        PUBLIC_FEEDBACK_SCOPE,
        AGGREGATE_FEEDBACK_SCOPE,
    }:
        raise ValueError("NLE Episode Feedback scope is invalid")


def _stats(observation: PolicyValue) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise RuntimeError("NLE public observation is invalid")
    stats = observation.get("stats")
    if type(stats) is not dict:
        raise RuntimeError("NLE public stats are invalid")
    return stats


def _information(value: object) -> tuple[bool, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "end_status",
        "is_ascended",
    }:
        raise RuntimeError("NLE returned invalid information")
    ascended = value["is_ascended"]
    end_status = value["end_status"]
    if type(ascended) is not bool or isinstance(end_status, bool):
        raise RuntimeError("NLE returned invalid status information")
    try:
        status = operator.index(cast(SupportsIndex, end_status))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("NLE returned invalid end status") from error
    if status not in {-1, 0, 1}:
        raise RuntimeError("NLE returned unknown end status")
    return ascended, status


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("NLE returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("NLE returned non-finite reward")
    return number


__all__ = ["NetHackEnvironment"]
