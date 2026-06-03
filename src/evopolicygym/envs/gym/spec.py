"""Gymnasium environment specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

Json = Any


@dataclass(frozen=True, slots=True)
class TaskDoc:
    """Human-readable task semantics for one Gymnasium environment."""

    title: str
    objective: str
    observation: str
    action: str
    reward: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("title", "objective", "observation", "action", "reward"):
            if not getattr(self, name):
                raise ValueError(f"task doc {name} must be non-empty")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))


@dataclass(frozen=True, slots=True)
class GymSpec:
    """Static EvoPolicyGym metadata for one Gymnasium registry id."""

    name: str
    id: str
    steps: int
    cases: int = 64
    valid_size: int = 64
    final_size: int = 256
    expert: float = 0.0
    random: float = 0.0
    version: str = "0.1"
    kwargs: Mapping[str, Json] = field(default_factory=dict)
    doc: TaskDoc | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("spec.name must be non-empty")
        if not self.id:
            raise ValueError("spec.id must be non-empty")
        if self.steps <= 0:
            raise ValueError("spec.steps must be positive")
        if self.cases <= 0:
            raise ValueError("spec.cases must be positive")
        if self.valid_size <= 0:
            raise ValueError("spec.valid_size must be positive")
        if self.final_size <= 0:
            raise ValueError("spec.final_size must be positive")
        object.__setattr__(self, "kwargs", dict(self.kwargs))


ACROBOT = GymSpec(
    name="gym/acrobot",
    id="Acrobot-v1",
    steps=500,
    expert=-80.0,
    random=-500.0,
    doc=TaskDoc(
        title="Acrobot",
        objective="Swing the two-link underactuated pendulum up until the tip reaches the target height.",
        observation=(
            "A six-value Box: cos(theta1), sin(theta1), cos(theta2), sin(theta2), "
            "angular velocity of joint 1, and angular velocity of joint 2. Angles are encoded with "
            "sin/cos pairs to avoid wraparound discontinuities."
        ),
        action="A Discrete action with three torques applied at the actuated joint: negative, zero, or positive torque.",
        reward="Each step gives -1 until the goal is reached. Better policies finish in fewer steps, producing less negative returns.",
        notes=("The task is sparse and delayed; useful policies usually build momentum before the final upswing.",),
    ),
)

CARTPOLE = GymSpec(
    name="gym/cartpole",
    id="CartPole-v1",
    steps=500,
    expert=500.0,
    random=20.0,
    doc=TaskDoc(
        title="CartPole",
        objective="Keep the pole balanced by moving the cart left or right.",
        observation="A four-value Box: cart position, cart velocity, pole angle, and pole angular velocity.",
        action="A Discrete action with two values: push cart left or push cart right.",
        reward="Each balanced step gives +1. The maximum episode return is the step limit.",
        notes=("The built-in `cartpole` environment is dependency-free; `gym/cartpole` uses Gymnasium CartPole-v1.",),
    ),
)

MOUNTAINCAR = GymSpec(
    name="gym/mountaincar",
    id="MountainCar-v0",
    steps=200,
    expert=-110.0,
    random=-200.0,
    doc=TaskDoc(
        title="MountainCar",
        objective="Drive the underpowered car up the right hill to reach the goal position.",
        observation="A two-value Box: horizontal position and velocity.",
        action="A Discrete action with three values: accelerate left, coast, or accelerate right.",
        reward="Each step gives -1 until the goal is reached. Better policies reach the goal in fewer steps.",
        notes=("The car must usually build momentum by moving away from the goal before climbing back.",),
    ),
)

CONTINUOUSCAR = GymSpec(
    name="gym/continuouscar",
    id="MountainCarContinuous-v0",
    steps=999,
    expert=90.0,
    random=-100.0,
    doc=TaskDoc(
        title="MountainCar Continuous",
        objective="Drive the underpowered car up the right hill using a continuous throttle.",
        observation="A two-value Box: horizontal position and velocity.",
        action="A one-value Box throttle. Negative values push left, positive values push right, and magnitude controls force.",
        reward="The environment rewards reaching the goal and penalizes squared action magnitude, so efficient control matters.",
        notes=("Actions outside the declared Box are clipped by the adapter before stepping the environment.",),
    ),
)

PENDULUM = GymSpec(
    name="gym/pendulum",
    id="Pendulum-v1",
    steps=200,
    expert=-200.0,
    random=-1200.0,
    doc=TaskDoc(
        title="Pendulum",
        objective="Swing the pendulum upright and keep it near the upright position with low control effort.",
        observation="A three-value Box: cos(theta), sin(theta), and angular velocity. Upright corresponds roughly to cos(theta)=1 and sin(theta)=0.",
        action="A one-value Box torque. Negative and positive values apply torque in opposite directions.",
        reward="Reward is negative cost based on angle error, angular velocity, and torque. Less negative return is better.",
        notes=("Use the sin/cos pair rather than raw angle; angle wraparound is already encoded in the observation.",),
    ),
)

BLACKJACK = GymSpec(
    name="gym/blackjack",
    id="Blackjack-v1",
    steps=200,
    expert=1.0,
    random=-0.4,
    doc=TaskDoc(
        title="Blackjack",
        objective="Play one hand of Blackjack against the dealer and maximize win probability.",
        observation=(
            "A tuple-like observation encoded by Gymnasium: player sum, dealer showing card, "
            "and whether the player has a usable ace."
        ),
        action="A Discrete action with two values: stick or hit.",
        reward="A win gives +1, a draw gives 0, and a loss gives -1 at episode end.",
        notes=("The task is episodic and mostly terminal-reward; intermediate steps usually carry no reward.",),
    ),
)

CLIFF = GymSpec(
    name="gym/cliff",
    id="CliffWalking-v1",
    steps=200,
    expert=-13.0,
    random=-100.0,
    doc=TaskDoc(
        title="Cliff Walking",
        objective="Navigate from the start to the goal on a grid while avoiding the cliff cells.",
        observation="A Discrete integer state representing the agent's grid cell in row-major order.",
        action="A Discrete action with four movement choices: up, right, down, or left.",
        reward="Each normal step gives -1. Stepping into the cliff gives a large negative penalty and resets the agent to start.",
        notes=("The shortest path near the cliff is risky; robust policies account for the cliff penalty.",),
    ),
)

FROZENLAKE = GymSpec(
    name="gym/frozenlake",
    id="FrozenLake-v1",
    steps=100,
    expert=1.0,
    random=0.0,
    doc=TaskDoc(
        title="FrozenLake",
        objective="Move across the frozen grid from start to goal without falling into holes.",
        observation="A Discrete integer state representing the grid cell in row-major order.",
        action="A Discrete action with four movement choices: left, down, right, or up.",
        reward="The goal gives +1. Other transitions give 0, including most failed exploration before termination.",
        notes=("The default Gymnasium environment is slippery, so intended actions may not be the actual movement.",),
    ),
)

TAXI = GymSpec(
    name="gym/taxi",
    id="Taxi-v4",
    steps=200,
    expert=8.0,
    random=-800.0,
    doc=TaskDoc(
        title="Taxi",
        objective="Pick up the passenger at the source location and drop them off at the destination.",
        observation=(
            "A Discrete integer encoding taxi row, taxi column, passenger location, and destination. "
            "Use Gymnasium's documented Taxi encoding or decode by integer arithmetic."
        ),
        action="A Discrete action with six values: move south, north, east, west, pickup, or dropoff.",
        reward="Each step gives -1, successful dropoff gives +20, and illegal pickup/dropoff gives -10.",
        notes=("Good policies decode the integer state into taxi position, passenger status, and destination before acting.",),
    ),
)

P1: tuple[GymSpec, ...] = (
    ACROBOT,
    CARTPOLE,
    MOUNTAINCAR,
    CONTINUOUSCAR,
    PENDULUM,
    BLACKJACK,
    CLIFF,
    FROZENLAKE,
    TAXI,
)

BOX2D: tuple[GymSpec, ...] = (
    GymSpec(name="gym/bipedal", id="BipedalWalker-v3", steps=1600),
    GymSpec(
        name="gym/racing",
        id="CarRacing-v3",
        steps=1000,
        expert=900.0,
        random=-50.0,
        doc=TaskDoc(
            title="Car Racing",
            objective="Drive a top-down race car around the generated track and maximize lap progress while staying on the road.",
            observation=(
                "A 96x96 RGB image. Feedback stores these observations externally as `observations.npy`; "
                "`trajectory.jsonl` uses `obs: null`, and row `t` corresponds to image frame `observations.npy[t]`."
            ),
            action=(
                "A three-value Box action: steering in [-1, 1], gas in [0, 1], and brake in [0, 1]. "
                "Return a JSON-safe list such as `[0.0, 0.3, 0.0]`."
            ),
            reward=(
                "The environment rewards visiting track tiles and applies a small per-step penalty. "
                "Better policies keep the car on track, avoid unnecessary braking, and complete more of the lap."
            ),
            notes=(
                "Use visual feedback artifacts to inspect road position, grass excursions, and whether actions saturate.",
                "The first frames may show the car near the starting area before meaningful progress is visible.",
            ),
        ),
    ),
    GymSpec(name="gym/lunar", id="LunarLander-v3", steps=1000),
)

MUJOCO: tuple[GymSpec, ...] = (
    GymSpec(name="gym/ant4", id="Ant-v4", steps=1000),
    GymSpec(name="gym/ant5", id="Ant-v5", steps=1000),
    GymSpec(name="gym/halfcheetah4", id="HalfCheetah-v4", steps=1000),
    GymSpec(name="gym/halfcheetah5", id="HalfCheetah-v5", steps=1000),
    GymSpec(name="gym/hopper4", id="Hopper-v4", steps=1000),
    GymSpec(name="gym/hopper5", id="Hopper-v5", steps=1000),
    GymSpec(name="gym/humanoid4", id="Humanoid-v4", steps=1000),
    GymSpec(name="gym/humanoid5", id="Humanoid-v5", steps=1000),
    GymSpec(name="gym/standup4", id="HumanoidStandup-v4", steps=1000),
    GymSpec(name="gym/standup5", id="HumanoidStandup-v5", steps=1000),
    GymSpec(name="gym/doublependulum4", id="InvertedDoublePendulum-v4", steps=1000),
    GymSpec(name="gym/doublependulum5", id="InvertedDoublePendulum-v5", steps=1000),
    GymSpec(name="gym/invertedpendulum4", id="InvertedPendulum-v4", steps=1000),
    GymSpec(name="gym/invertedpendulum5", id="InvertedPendulum-v5", steps=1000),
    GymSpec(name="gym/pusher4", id="Pusher-v4", steps=100),
    GymSpec(name="gym/pusher5", id="Pusher-v5", steps=100),
    GymSpec(name="gym/reacher4", id="Reacher-v4", steps=50),
    GymSpec(name="gym/reacher5", id="Reacher-v5", steps=50),
    GymSpec(name="gym/swimmer4", id="Swimmer-v4", steps=1000),
    GymSpec(name="gym/swimmer5", id="Swimmer-v5", steps=1000),
    GymSpec(name="gym/walker4", id="Walker2d-v4", steps=1000),
    GymSpec(name="gym/walker5", id="Walker2d-v5", steps=1000),
)

DEFAULTS: tuple[GymSpec, ...] = (*P1, *BOX2D, *MUJOCO)

BY_NAME: dict[str, GymSpec] = {spec.name: spec for spec in DEFAULTS}


__all__ = [
    "ACROBOT",
    "BLACKJACK",
    "BY_NAME",
    "CARTPOLE",
    "CLIFF",
    "CONTINUOUSCAR",
    "DEFAULTS",
    "BOX2D",
    "FROZENLAKE",
    "MOUNTAINCAR",
    "MUJOCO",
    "P1",
    "PENDULUM",
    "TAXI",
    "GymSpec",
    "TaskDoc",
]
