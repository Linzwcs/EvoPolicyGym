"""Env registry — each env package self-registers at import time.

Convention: importing `hlbench.envs.<env_id>` triggers `register_env(...)`
in its module body. Test code or the Server explicitly imports the envs
it needs.
"""

# v0: import envs eagerly. Once we have many, switch to lazy / entry_points.
from hlbench.envs import (  # noqa: F401  (side effect: register)
    acrobot,
    bipedal_hardcore,
    bipedal_walker,
    cache_replacement,
    k_server,
    lunar_hardcore,
    lunar_lander_continuous,
    mountain_car_continuous,
    online_bipartite_matching,
    pendulum,
    pendulum_hardcore,
)
