"""Environment catalog and built-in task registrations."""

from .cartpole import CartPole, cartpole
from .catalog import Registry
from .gym import (
    Gym,
    acrobot,
    blackjack,
    cliff,
    continuouscar,
    frozenlake,
    gym,
    mountaincar,
    pendulum,
    taxi,
)
from .gym import (
    available as gym_available,
)
from .gym import (
    cartpole as gym_cartpole,
)
from .gym import (
    envs as gym_envs,
)
from .manifest import Entry, Level
from .toy import Toy, toy


def registry(*, bulk: bool = False, filters: tuple[str, ...] = ()) -> Registry:
    """Return the default built-in environment catalog."""

    return Registry.of(cartpole(), toy(), *gym_envs(bulk=bulk, filters=filters))


__all__ = [
    "CartPole",
    "Gym",
    "Registry",
    "Toy",
    "acrobot",
    "blackjack",
    "cartpole",
    "cliff",
    "continuouscar",
    "Entry",
    "frozenlake",
    "gym",
    "gym_available",
    "gym_cartpole",
    "Level",
    "mountaincar",
    "pendulum",
    "registry",
    "taxi",
    "toy",
]
