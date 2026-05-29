# Online-Bipartite-Matching — Decide Match Or Skip Online

## Goal

A bipartite graph has `N=16` fixed **left vertices** and `M=24`
**right vertices that arrive online**. Each arriving right vertex
carries a set of edges to a subset of left vertices. For each
arrival the policy must decide **immediately and irrevocably**
whether to match the arrival to one of its (still-unmatched) left
neighbors, or skip (leave unmatched). Maximize the total number of
successful matches.

## What's different from textbook online matching

Train and held-out **graph distributions differ**:

- **Train pool**: random bipartite graphs — each (left, right) edge
  independently present with probability `p = 0.25`. The RANKING
  algorithm (Karp-Vazirani-Vazirani 1990) achieves the
  `(1 - 1/e) ≈ 0.632` competitive ratio in expectation here.
- **Held-out pool**: **structured adversarial** graphs designed to
  punish greedy. First `M/2` arrivals connect only to the "left
  half" of left vertices. Second `M/2` arrivals connect both to the
  "right half" AND to a few "honey trap" vertices in the left half.
  Greedy that grabs the honey trap matches strands second-half right
  vertices that needed those left-half slots.

A pure greedy policy ("match to first available neighbor") wins on
train (~12-13 matches) but collapses on held-out (~6-8 matches). To
generalize, the policy must either:

1. Implement RANKING (random permutation of left vertices, match to
   highest-ranked available neighbor).
2. Save left-half vertices for late arrivals if the early arrivals
   are unusually concentrated.
3. Use a stochastic policy (e.g., flip a coin between greedy and
   conservative) — a known robust strategy for adversarial inputs.

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
| **Input** `obs` | `numpy.ndarray` | `(32,)` | `int8` | See layout below |
| **Return** | `int` | scalar | python int | Left vertex `[0, N)` to match, or `N` to skip |

### Observation layout (32 ints = 2 × 16)

| Index range | Component | Encoding |
|---|---|---|
| 0..15 | `left_matched_mask` | 1 = left vertex already matched (unavailable); 0 = still free |
| 16..31 | `current_arrival_neighbors` | 1 = arrival has an edge to this left vertex; 0 = no edge |

Valid match: action `a ∈ [0, N)` where `left_matched_mask[a] == 0`
AND `current_arrival_neighbors[a] == 1`. Invalid attempts (matching
to an already-matched vertex, or to a non-neighbor) have **no
effect** and yield 0 reward — they're not penalized beyond the
opportunity cost of not matching successfully this turn.

Action `a == N` (here: `a == 16`) is the explicit **skip** — same
0 reward but signals intentional non-matching.

## Reward

`+1.0` per successful match, `0.0` otherwise. Episode return = total
matches, bounded above by `N = 16` (perfect matching) and below by
`0`.

## Episode structure

- 24 arrivals per episode.
- `left_matched_mask` starts all-zero; accumulates as matches occur.
- Once a left vertex is matched, it stays matched for the rest of
  the episode (no rematching).

## Strategies you may take

1. **Greedy first-available**: match to the lowest-index unmatched
   neighbor. Simple, ~12 on train, ~7 on held-out.
2. **Greedy last-available**: match to the highest-index unmatched
   neighbor. Performance ~ same as first-available but different
   failure mode on adversarial.
3. **RANKING** (KVV'90): at episode start, sample a random
   permutation `π` of left vertices; for each arrival, match to the
   unmatched neighbor with smallest `π[v]`. `(1 - 1/e)`-competitive
   in expectation; ~13 on train, ~9 on held-out.
4. **Threshold / waterfilling**: reserve a fraction of left
   vertices for late arrivals; only match early arrivals greedily
   from the "unreserved" pool. Better on held-out, worse on train.
5. **Skip-anticipating**: if the current arrival has many neighbors
   (>3), skip it (assume a later arrival with fewer options will
   need one of these neighbors). Aggressive — can backfire.

The held-out evaluation rewards finding a strategy that doesn't
collapse on adversarial structure.

## Configuration (informational)

| Constant | Value |
|---|---|
| `N_LEFT` | 16 |
| `M_RIGHT` (arrivals per episode) | 24 |
| `EDGE_PROB_TRAIN` | 0.25 |
| Action `N` (= 16) | skip |
