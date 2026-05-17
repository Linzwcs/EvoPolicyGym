"""Policy rollout entrypoint."""

from hlbench.rollout.cli import main
from hlbench.rollout.engine import file_sha256, load_policy, run_episode, run_rollout

__all__ = ["file_sha256", "load_policy", "main", "run_episode", "run_rollout"]

if __name__ == "__main__":
    raise SystemExit(main())
