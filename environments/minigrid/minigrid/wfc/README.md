# MiniGrid WFC Benchmark

An independently installable EvoPolicyGym Benchmark family for all 22
MiniGrid Wave Function Collapse presets.

```python
from minigrid_wfc import WFCBenchmark, WFCConfig

benchmark = WFCBenchmark(
    WFCConfig(profile="MazeSimple", size=25)
)
```

Each Episode runs Wave Function Collapse again from the selected packaged PNG
pattern and Episode seed; this is not a fixed map library. The largest
navigable component is retained, then a random start and goal are placed inside
it, guaranteeing that they share a connected component. Deterministic reset
retries handle presets that occasionally fail to collapse.

The Policy must navigate the generated maze and reach its random goal. Only
turning and forward movement are useful. Feedback reports goal discovery,
public-observation-derived known cells, walkable cells, walls, exploration
frontiers and known goal distance, map expansion and stalls, blocked movement,
unused actions, observation novelty, ineffective actions, action mix, exact
terminal reason, and success;
the bounded semantic trace contains no seeds, private identity, or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
The Host selects a preset profile and size (`15` or `25`) before the Run.
Generation-heavy profiles remain explicit rather than being mixed into an
Episode pool.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
