# Cache-Replacement — Online Eviction Under Distribution Shift

## Goal

Stream of memory accesses; cache has fixed capacity. On every miss
the policy must choose which cached object to evict. Maximize the
total number of hits across the trace.

## What's different from textbook cache replacement

Train and held-out **access distributions differ**:

- **Train pool**: Zipfian distribution (small set of "hot" objects
  accessed most often). High locality. LRU, LFU, ARC all work well —
  ARC is near-optimal.
- **Held-out pool**: scan-heavy traces (cycles through a permutation
  of all objects). The working set exceeds cache capacity, defeating
  LRU because every access targets the object LRU just evicted.

A policy that implements pure LRU will score well in-loop but
collapse on held-out. To generalize, the policy must either:

1. Recognize the trace pattern from observations and switch strategy.
2. Use a distribution-agnostic strategy (e.g., ARC's adaptive
   recency/frequency mix, or LIRS's reuse-distance tracking).
3. Take advantage of the access history window in obs to detect scans.

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
| **Input** `obs` | `numpy.ndarray` | `(17,)` | `int32` | See layout below |
| **Return** | `int` | scalar | python int | Slot index to evict, in `[0, 8)` |

### Observation layout (17 ints total)

| Index range | Component | Encoding |
|---|---|---|
| 0..7 | `cache_slots` | Object IDs currently in each cache slot. `-1` = empty slot. |
| 8..15 | `history` | The 8 most-recent access IDs (chronological, with `-1` padding at the front before history fills). |
| 16 | `current_access` | The object ID being requested at this step. |

The current access (index 16) is the input you must decide for. If
`current_access` is in `cache_slots[0..7]`, it's a **hit** — your
returned action is ignored (but must still be a valid int in
`[0, 8)`). Otherwise it's a **miss** — your returned action selects
which slot to overwrite with `current_access`.

## Reward

`+1.0` per cache hit, `0.0` per miss. Episode return = total hits over
the trace. Trace length is 500 steps, so the score range is `[0, 500]`.

## Episode structure

- 500 accesses per episode (no natural termination, only truncation
  on the last step).
- Cache starts empty; first 8 misses fill the cache without eviction.
- After that, every miss requires an eviction decision.

## Configuration (informational; values are server-determined)

| Constant | Value | Notes |
|---|---|---|
| Cache capacity | 8 | Number of slots |
| History window | 8 | Recent accesses visible to policy |
| Total object IDs | 64 | The address space |
| Trace length | 500 | Episode horizon |

## Strategies you may take

1. **LRU**: evict the slot whose object was used the longest ago.
   Track per-slot recency in policy state. ~85% hit rate on Zipfian,
   ~5% on held-out scans.
2. **LFU**: evict the slot accessed least often. Needs a counter per
   slot. Works on some Zipfian; poor on scans.
3. **ARC** (Adaptive Replacement Cache): two LRU lists, one for
   recently-used and one for frequently-used items, with a tunable
   weight. Adapts to distribution shifts. Near-optimal on most train
   traces; partial improvement on held-out.
4. **Pattern detection + switch**: monitor `history` for scan
   patterns (long-period repeats with no clustering), switch
   strategy when detected.
5. **MRU** (most-recently-used): the surprising winner on pure scans
   — by evicting the just-accessed item, you preserve older items
   that the scan won't return to soon. Catastrophic on Zipfian.

The held-out evaluation rewards finding a strategy that doesn't
catastrophically fail on either distribution.
