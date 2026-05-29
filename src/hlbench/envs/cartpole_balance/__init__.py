"""CartPole-Balance env registration (Classic Control, EASY anchor).

Standard `CartPole-v1`: balance a pole hinged on a cart by applying
left/right force. Episode terminates on fall (pole angle > 12°) or
cart leaving the track (|x| > 2.4). Reward +1 per step alive; max
500 per episode (episode horizon).

Why include in the v1 scored suite:

  - **EASY anchor for score-distribution diagnostics.** Most other
    envs in the suite are medium-hard or hard, by design. Having one
    env where even a naive PD controller scores near 100 gives the
    paper's per-env score histogram a clean baseline point — useful
    for verifying that low scores on hard envs aren't an artifact of
    the protocol but reflect actual difficulty.
  - **Different action space than Pendulum.** Pendulum is continuous
    Box(1); CartPole is Discrete(2). The v1 suite needs at least one
    Discrete-action Classic Control env to test both action types in
    that category.
  - **Classic baseline.** Almost every RL textbook starts with
    CartPole. Including it makes cross-paper comparison easier
    ("our score on CartPole is X; theirs is Y").

Side effect: importing this module registers ``cartpole_balance``.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("CartPole-v1", render_mode=None)


register_env(
    env_id="cartpole_balance",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [4],
        # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        "low": [-4.8, -float("inf"), -0.42, -float("inf")],
        "high": [4.8, float("inf"), 0.42, float("inf")],
        "dtype": "float32",
    },
    action_space={
        "type": "Discrete",
        "n": 2,
    },
    max_episode_steps=500,
    # Per Gymnasium docs: random ≈ 22 (falls in ~22 steps); "solved"
    # at +475 over 100 episodes. Hand-crafted angle-based PD
    # (push left if pole leans right, right if leans left) reaches
    # near the 500 ceiling immediately.
    expert_baseline=500.0,
    random_baseline=22.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
