"""Map env_instance ID -> hidden real seed; load static seed pools.

Each env ships:
- train.json  : agent-addressable instances. Format `{"real_seeds": [int, ...]}`,
                array index = env_instance ID, value = real seed used by env.reset().
- heldout.json: held-out evaluation pool. Same format. Agent NEVER sees these.

The mapping is read-only and bound to env_version. See SPEC.md §6.

Anti-cheating: real seeds are server-internal. Agents address by integer ID
in [0, n_env_instances).
"""

from __future__ import annotations

import json
from pathlib import Path


class SeedManager:
    """Loads train.json and heldout.json for one env."""

    def __init__(self, train_path: Path, heldout_path: Path) -> None:
        self._train: list[int] = json.loads(train_path.read_text())["real_seeds"]
        self._heldout: list[int] = json.loads(heldout_path.read_text())["real_seeds"]

    @property
    def n_env_instances(self) -> int:
        """Number of agent-addressable instances (length of train pool)."""
        return len(self._train)

    @property
    def n_held_out(self) -> int:
        """Held-out pool size (server-internal; not exposed to agent)."""
        return len(self._heldout)

    def real_seed_for_instance(self, env_instance_id: int) -> int:
        """Resolve agent-facing ID to the underlying real seed.

        Raises ValueError on out-of-range ID (caller maps to invalid_env_instance verdict).
        """
        if not (0 <= env_instance_id < len(self._train)):
            raise ValueError(
                f"env_instance {env_instance_id} out of range [0, {len(self._train)})"
            )
        return self._train[env_instance_id]

    def held_out_seeds(self) -> list[int]:
        """Server-internal access for finalize/held-out evaluation."""
        return list(self._heldout)
