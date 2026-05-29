# K-Server — Online Server Dispatch on a 2D Plane

## Goal

`K=3` servers are positioned in 2D Euclidean space. A stream of
request points arrives, one at a time. For each request the policy
must dispatch exactly one server — that server moves from its current
position to the request point. Minimize the total distance moved
across the trace.

## What's different from textbook K-server

Train and held-out **request distributions differ**:

- **Train pool**: two Gaussian clusters at `(±0.4, ±0.4)` with equal
  weight. Greedy "nearest server" works reasonably (~`-30` mean
  return).
- **Held-out pool**: four corners at `(±0.7, ±0.7)`, with **75% of
  requests at the (0.7, 0.7) corner**. Greedy overcommits one server
  to the hot corner and leaves the other servers idle, paying high
  travel costs when the remaining 25% lands at the cold corners.

A policy that learns "always dispatch nearest server" wins train but
loses held-out. To generalize, the policy needs to either:

1. Recognize asymmetric demand and reserve servers for cold regions
   even at short-term cost.
2. Use a strategy with provable competitive ratio (Work Function
   Algorithm — but it's O(k! n), intractable in `act_wall_ms=10ms`).
3. Track recent request frequencies per region and adapt.

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...
```

### `act(obs)`

| Direction | Python type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(8,)` | `float32` | `[s0.x, s0.y, s1.x, s1.y, s2.x, s2.y, req.x, req.y]` |
| **Return** | `int` | scalar | python int | Server index in `[0, K)` to dispatch |

### Observation layout

| Index range | Component | Bounds |
|---|---|---|
| 0..5 | `server_positions` | 3 servers × (x, y), each in `[-1, 1]` |
| 6..7 | `current_request` | (x, y) in `[-1, 1]` |

Server 0 starts at angle 0, server 1 at angle 120°, server 2 at 240°,
all on a circle of radius 0.3 around the origin. (You can read the
exact initial positions from the obs at step 0.)

## Reward

Per step: `-distance(current_server_pos, request_pos)`. Always
non-positive. Episode return = negative sum of total movement.

## Episode structure

- 200 requests per episode.
- Initial server positions are fixed across episodes (deterministic).
- Request sequence is determined by the env's hidden seed.

## Strategies you may take

1. **Greedy nearest** (the textbook baseline). Compute Euclidean
   distance from each server to the request, dispatch closest.
   ~`-30` on train, ~`-80` on held-out.
2. **Greedy with reservation**. Greedy, but with a penalty if
   dispatching a server "abandons" its cluster (no other server
   nearby). Mitigates held-out collapse partially.
3. **Anticipation**. Maintain a running estimate of the request
   distribution from recent history (track recent request centroids),
   pre-position servers near predicted hot spots.
4. **Round-robin with bias**. Always dispatch in cycle; ignore
   geometry. Lower train score but more robust on adversarial
   held-out (paradoxically).

The held-out evaluation rewards finding a strategy that doesn't
collapse on the 75/25 adversarial weighting.

## Configuration (informational; values are server-determined)

| Constant | Value |
|---|---|
| `K_SERVERS` | 3 |
| `N_REQUESTS` per episode | 200 |
| Plane bounds | `[-1, 1]^2` |
| Initial server radius | 0.3 (around origin) |
