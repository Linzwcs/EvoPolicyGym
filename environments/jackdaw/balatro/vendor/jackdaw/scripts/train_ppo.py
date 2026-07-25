"""Train MaskablePPO on the Balatro gymnasium environment.

Requires the ``train`` optional dependency group::

    uv pip install -e ".[train]"

Usage::

    python scripts/train_ppo.py --total-timesteps 500000
    python scripts/train_ppo.py --total-timesteps 50000 --log-dir runs/exp1 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


class BalatroMetricsCallback(BaseCallback):
    """Log Balatro-specific episode metrics to tensorboard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._antes: list[int] = []
        self._rounds: list[int] = []
        self._wins: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "balatro/ante_reached" in info:
                self._antes.append(info["balatro/ante_reached"])
                self._rounds.append(info["balatro/rounds_beaten"])
                self._wins.append(info["balatro/won"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._antes:
            return
        self.logger.record("balatro/mean_ante_reached", np.mean(self._antes))
        self.logger.record("balatro/max_ante_reached", np.max(self._antes))
        self.logger.record("balatro/mean_rounds_beaten", np.mean(self._rounds))
        self.logger.record("balatro/win_rate", np.mean(self._wins))
        self._antes.clear()
        self._rounds.clear()
        self._wins.clear()


def make_env(seed: int = 0, max_steps: int = 10_000) -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"PPO_{seed}",
        reward_shaping=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="runs/balatro_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=args.seed, max_steps=args.max_steps)

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_path),
        ent_coef=0.05,
        learning_rate=1e-4,
        n_steps=4096,
        clip_range=0.15,
    )

    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=BalatroMetricsCallback())

    save_path = log_path / "balatro_ppo"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
