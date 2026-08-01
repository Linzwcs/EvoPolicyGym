"""Host-owned deterministic training Episode pool construction."""

from __future__ import annotations

import hashlib

from ..authoring import EpisodeSpec
from ..benchmark import Benchmark
from ..errors import EvaluationError
from ..evaluation._inputs import EpisodeInput
from . import RunConfig

TRAINING_POOL_DERIVATION = "evopolicygym/training-pool/v1"
TRAINING_POLICY_DERIVATION = "evopolicygym/training-policy/v1"

_TRAINING_POOL_SEED_DOMAIN = b"evopolicygym/training-pool/v1\0"
_TRAINING_POLICY_SEED_DOMAIN = b"evopolicygym/training-policy/v1\0"


def build_training_episode_pool(
    benchmark: Benchmark,
    config: RunConfig,
) -> tuple[EpisodeInput, ...]:
    """Plan the fixed Run-local training pool before Agent execution."""

    pool_size = config.episode_pool_size
    assert pool_size is not None
    training_seed = _derive_seed(
        config.seed,
        _TRAINING_POOL_SEED_DOMAIN,
    )
    try:
        episodes = tuple(
            benchmark.episodes(
                config.split,
                seed=training_seed,
                count=pool_size,
            )
        )
    except Exception:
        raise EvaluationError(
            "Benchmark could not plan the training Episode pool"
        ) from None
    if len(episodes) != pool_size:
        raise EvaluationError(
            "Benchmark returned the wrong training Episode count"
        )
    if any(type(episode) is not EpisodeSpec for episode in episodes):
        raise EvaluationError(
            "Benchmark returned an invalid training Episode pool"
        )
    return tuple(
        EpisodeInput(
            spec=episode,
            policy_seed=_derive_seed(
                config.seed,
                _TRAINING_POLICY_SEED_DOMAIN,
                index,
            ),
        )
        for index, episode in enumerate(episodes)
    )


def _derive_seed(
    run_seed: int,
    domain: bytes,
    index: int | None = None,
) -> int:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(run_seed.to_bytes(8, "big"))
    if index is not None:
        digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


__all__: list[str] = []
