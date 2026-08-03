"""Canonical single-Policy Jumanji profiles with bounded public feedback."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy
from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
    Transition,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import JumanjiConfig
from .environment import JumanjiEnvironment

_SEED_DOMAIN = b"evopolicygym-jumanji/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_SUMMARIZED_EPISODES = 128
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_STEPS_PER_EPISODE = 48
_TRACE_EDGE_STEPS = 12
_MAX_EVENT_STEPS = 12
_MAX_LEGAL_ACTION_SAMPLES = 16
_MAX_ACTION_SUMMARIES = 16
_KNAPSACK_CAPACITY = 12.5
_RUBIK_FACES = ("up", "front", "right", "back", "left", "down")
_RUBIK_ROTATIONS = ("clockwise", "anticlockwise", "half_turn")
_RUBIK_STICKER_COLORS = ("white", "green", "red", "blue", "orange", "yellow")
_RUBIK_FACE_VIEW_ORIENTATIONS = {
    "up": "left face is on the left; back face points up",
    "front": "left face is on the left; up face points up",
    "right": "front face is on the left; up face points up",
    "back": "right face is on the left; up face points up",
    "left": "back face is on the left; up face points up",
    "down": "left face is on the left; front face points up",
}
type _ObservationFieldSpec = tuple[str, tuple[int, ...]]

_RUBIK_FIELDS: dict[str, _ObservationFieldSpec] = {
    "cube": ("int8", (6, 3, 3)),
    "step_count": ("int64", ()),
}
_RUBIK_FIELD_MEANINGS = {
    "cube": (
        "cube[face, row, column] is a sticker color id. Face indices "
        "0/1/2/3/4/5 are up/front/right/back/left/down and sticker values "
        "0/1/2/3/4/5 are white/green/red/blue/orange/yellow. Rows and columns "
        "are read while looking directly at that face using the published face "
        "view orientations. A solved reachable cube has nine copies of face "
        "index i on face i."
    ),
    "step_count": "Number of face turns applied since reset.",
}
_SUDOKU_FIELDS: dict[str, _ObservationFieldSpec] = {
    "action_mask": ("bool", (9, 9, 9)),
    "board": ("int32", (9, 9)),
}
_SUDOKU_FIELD_MEANINGS = {
    "action_mask": (
        "action_mask[row, column, value] is true exactly when that zero-based "
        "cell is empty and internal value 0-8 (human symbol 1-9) is absent from "
        "its row, column, and 3x3 box."
    ),
    "board": (
        "The 9x9 board: -1 is empty and internal values 0-8 represent human "
        "Sudoku symbols 1-9. Initially filled cells are immutable because every "
        "action on them is masked false."
    ),
}
_PROFILE_TASK_DESCRIPTIONS: dict[str, str] = {
    "rubiks-cube": (
        "Restore a 3x3x3 Rubik's Cube after reset applies exactly 100 "
        "independently uniform random legal face turns to a solved cube. Turns "
        "are sampled with replacement, so cancellations are possible and 100 "
        "is not a claim about minimum solution distance. An action is [face, "
        "depth, rotation]; depth has only value 0 and rotates the outer layer, "
        "while rotation is clockwise, anticlockwise, or a half turn as viewed "
        "directly at the selected face. All 18 actions are always legal. Reward "
        "is sparse: a transition that produces six uniform faces rewards 1, "
        "and every other transition rewards 0. The Episode terminates when "
        "solved or after 200 actions."
    ),
    "rubiks-cube-partly-scrambled": (
        "Restore a 3x3x3 Rubik's Cube after reset applies exactly 7 "
        "independently uniform random legal face turns to a solved cube. Turns "
        "are sampled with replacement, so the instance is at most seven moves "
        "from solved and may be closer. An action is [face, depth, rotation]; "
        "depth has only value 0 and rotates the outer layer, while rotation is "
        "clockwise, anticlockwise, or a half turn as viewed directly at the "
        "selected face. All 18 actions are always legal. Reward is sparse: a "
        "transition that produces six uniform faces rewards 1, and every other "
        "transition rewards 0. The Episode terminates when solved or after 20 "
        "actions."
    ),
    "sudoku": (
        "Complete a valid 9x9 Sudoku sampled uniformly from Jumanji 1.1.1's "
        "fixed 10,000-puzzle mixed-difficulty database. Its packaged puzzles "
        "contain 25-77 initial clues. An action is [row, column, value], all "
        "zero-based; internal value 0-8 means human symbol 1-9. Legal values "
        "cannot duplicate a symbol in the row, column, or 3x3 box. Intermediate "
        "reward is 0 and a completely valid board rewards 1, so Episode return "
        "is binary. The Episode terminates when solved or when no legal assignment "
        "remains; the Host rejects masked actions before upstream execution."
    ),
    "sudoku-very-easy": (
        "Complete a valid 9x9 Sudoku sampled uniformly from Jumanji 1.1.1's "
        "fixed 1,000-puzzle very-easy database. Its packaged puzzles contain "
        "46-80 initial clues. An action is [row, column, value], all zero-based; "
        "internal value 0-8 means human symbol 1-9. Legal values cannot duplicate "
        "a symbol in the row, column, or 3x3 box. Intermediate reward is 0 and a "
        "completely valid board rewards 1, so Episode return is binary. The "
        "Episode terminates when solved or when no legal assignment remains; the "
        "Host rejects masked actions before upstream execution."
    ),
    "graph-coloring": (
        "Color nodes 0 through 19 in order so adjacent nodes have different "
        "colors, while minimizing the number of distinct colors. Nonterminal "
        "reward is zero; successful terminal reward is the negative number of "
        "distinct colors, and an invalid color terminates with reward -20."
    ),
    "minesweeper": (
        "Reveal all 90 non-mine cells on the 10x10 board without selecting one "
        "of its 10 mines. Each safe reveal rewards 1; revealing a mine or making "
        "an invalid selection rewards 0. The Episode terminates when the board is "
        "solved, a mine is revealed, or an invalid selection is made."
    ),
    "sliding-tile-puzzle": (
        "Arrange the 5x5 board in row-major order [1, 2, ..., 24, 0], "
        "where 0 is the empty tile. Each action moves the empty tile in the "
        "named direction. Reset starts from the goal and applies 200 random "
        "legal moves, so every instance is solvable. Each transition rewards "
        "newly correct positions minus newly displaced correct positions; the "
        "Episode terminates when solved or after 500 moves."
    ),
    "bin-pack": (
        "Maximize volume utilization of one normalized unit-cube container. A "
        "legal action [empty_maximal_space, item] places that fixed-orientation "
        "item at the selected empty maximal space's lower corner. Each dense "
        "reward is the packed item's volume divided by container volume, so "
        "Episode return equals final volume utilization. The Episode terminates "
        "when every item is packed or no feasible placement remains."
    ),
    "flat-pack": (
        "Pack the 25 block footprints into the 11x11 grid without overlap; "
        "generated instances admit a complete tiling. An action chooses a block, "
        "0/90/180/270-degree clockwise rotation, and the top-left row and column "
        "of its 3x3 window. Each legal placement rewards its nonzero footprint "
        "cells divided by 121, so Episode return equals final grid occupancy. The "
        "Episode terminates after all blocks are placed or when no legal placement "
        "remains."
    ),
    "knapsack": (
        "Choose a subset of 50 items with total weight at most 12.5 to maximize "
        "their total value. Item weights and values are independently sampled in "
        "[0, 1) and remain fixed within an Episode. A legal action selects one "
        "unpacked item that fits the remaining capacity. Its dense reward is that "
        "item's value, so Episode return equals the packed subset's total value. "
        "The Episode terminates when no remaining item fits."
    ),
    "tetris": (
        "Place a uniformly sampled sequence of the seven tetrominoes on a 10x10 "
        "board. To construct rotation r, crop the current 4x4 shape's all-zero "
        "outer rows and columns, rotate the cropped footprint clockwise by r "
        "quarter-turns, then place it at the top-left of a zero-padded 4x4 window. "
        "An action selects r and that window's left column, then lets the piece "
        "fall to its lowest non-overlapping row. "
        "Completed rows are removed. Clearing 0/1/2/3/4 rows rewards "
        "0/40/100/300/1200 respectively. The Episode terminates when the next "
        "piece has no legal placement or after 400 placed pieces."
    ),
    "cvrp": (
        "Serve customers 1-20 from depot 0 while minimizing total Euclidean "
        "route length. Coordinates lie in the unit square. The vehicle starts at "
        "the depot with normalized capacity 1.0 (raw capacity 30); customer raw "
        "integer demands are sampled from [1, 10) and published divided by 30. "
        "Selecting a customer consumes its demand, while selecting depot 0 when "
        "away returns there and restores capacity to 1.0. Each dense reward is "
        "the negative distance from the current node to the selected node, so "
        "maximizing Episode return minimizes route length. The Episode finishes "
        "only after every customer is served and the vehicle returns to depot 0, "
        "within at most 40 actions."
    ),
    "snake": (
        "Collect fruit on a 12x12 grid. Actions 0/1/2/3 move the head one cell "
        "up/right/down/left. The snake starts at length 1. A move is legal when "
        "it stays on the board and avoids body cells that remain occupied after "
        "the tail advances; "
        "moving into the current tail cell is therefore legal when no fruit is "
        "eaten. Eating a fruit rewards 1, grows the snake by one cell, and "
        "uniformly resamples the next fruit from empty cells; every other legal "
        "move rewards 0 and preserves length. Episode return equals fruits "
        "eaten. The Episode terminates when the 144-cell board is filled, no "
        "legal move remains, or the 4000-step limit is reached."
    ),
    "tsp": (
        "Choose an ordering of 20 cities sampled independently and uniformly "
        "from the unit square, visiting every city exactly once. The first "
        "selected city starts the tour and rewards 0. Each later action rewards "
        "the negative Euclidean distance from the previous city; the twentieth "
        "and final action also subtracts the distance back to the first city to "
        "close the cycle. Thus Episode return is the negative closed-tour length, "
        "and maximizing return minimizes that length."
    ),
    "pacman": (
        "Collect all 318 pellets in a fixed 31-row by 28-column maze while "
        "avoiding four seeded heuristic ghosts. Actual Jumanji 1.1.1 dynamics "
        "map actions 0/1/2/3 to up/left/down/right; action 4 is no-op but its "
        "mask is always false, and this adapter rejects every masked action. A "
        "regular pellet rewards 10. Each of four power-up tiles also contains a "
        "regular pellet and adds 50, for 60 total on collection, then makes "
        "ghosts frightened for 30 steps. Eating an eligible frightened ghost "
        "adds 200, at most once per ghost per Episode. Score is cumulative "
        "reward, so Episode return equals final score and should be maximized. "
        "The Episode terminates after all pellets are collected, a non-frightened "
        "ghost collision, or 1,000 actions. The maze is fixed across Episodes; "
        "the Environment seed controls stochastic ghost choices."
    ),
}
_OBSERVATION_FIELD_MEANINGS: dict[str, dict[str, str]] = {
    "rubiks-cube": _RUBIK_FIELD_MEANINGS,
    "rubiks-cube-partly-scrambled": _RUBIK_FIELD_MEANINGS,
    "sudoku": _SUDOKU_FIELD_MEANINGS,
    "sudoku-very-easy": _SUDOKU_FIELD_MEANINGS,
    "graph-coloring": {
        "action_mask": (
            "Index c is true exactly when assigning color c to the current node "
            "is legal."
        ),
        "adj_matrix": (
            "adj_matrix[u, v] is true exactly when nodes u and v are adjacent."
        ),
        "colors": (
            "colors[i] is -1 until node i is assigned, then stores its integer "
            "color id."
        ),
        "current_node_index": "Index of the node colored by the next action.",
    },
    "minesweeper": {
        "action_mask": (
            "action_mask[row, column] is true exactly when that cell is still "
            "unexplored and may be selected."
        ),
        "board": (
            "board[row, column] is -1 for an unexplored cell; a revealed safe "
            "cell contains the number of mines among its up to eight neighbors."
        ),
        "num_mines": "Total number of mines hidden on the board.",
        "step_count": "Number of cells selected so far in the Episode.",
    },
    "sliding-tile-puzzle": {
        "action_mask": (
            "Index d is true exactly when the empty tile can move in direction d."
        ),
        "empty_tile_position": (
            "The empty tile's [row, column] coordinates; its puzzle value is 0."
        ),
        "puzzle": (
            "Tile labels in row-major board coordinates. The solved flattened "
            "ordering is [1, 2, ..., 24, 0]."
        ),
        "step_count": "Number of empty-tile moves taken so far in the Episode.",
    },
    "bin-pack": {
        "action_mask": (
            "action_mask[e, i] is true exactly when unpacked item i fits in "
            "empty maximal space e."
        ),
        "ems.x1": "Normalized lower x coordinate of each empty maximal space.",
        "ems.x2": "Normalized upper x coordinate of each empty maximal space.",
        "ems.y1": "Normalized lower y coordinate of each empty maximal space.",
        "ems.y2": "Normalized upper y coordinate of each empty maximal space.",
        "ems.z1": "Normalized lower z coordinate of each empty maximal space.",
        "ems.z2": "Normalized upper z coordinate of each empty maximal space.",
        "ems_mask": "True exactly for currently active empty maximal spaces.",
        "items.x_len": "Normalized fixed x length of each item.",
        "items.y_len": "Normalized fixed y length of each item.",
        "items.z_len": "Normalized fixed z length of each item.",
        "items_mask": "True exactly for item slots present in this instance.",
        "items_placed": "True exactly for items already packed.",
    },
    "flat-pack": {
        "action_mask": (
            "action_mask[b, r, row, column] is true exactly when unplaced block b "
            "at clockwise quarter-turn r can occupy that 3x3 window without overlap."
        ),
        "blocks": (
            "blocks[b] is block b's 3x3 footprint: zero cells are transparent and "
            "nonzero cells must occupy grid cells."
        ),
        "grid": "The 11x11 board: zero cells are empty and nonzero cells are occupied.",
    },
    "knapsack": {
        "action_mask": (
            "Index i is true exactly when item i is unpacked and weights[i] is no "
            "greater than 12.5 minus the total weight of packed items."
        ),
        "packed_items": "Index i is true exactly when item i has already been packed.",
        "values": (
            "values[i] is item i's fixed nonnegative value and is the immediate "
            "reward for legally selecting that item."
        ),
        "weights": "weights[i] is item i's fixed nonnegative contribution to capacity.",
    },
    "tetris": {
        "action_mask": (
            "action_mask[r, c] is true exactly when clockwise rotation r of the "
            "current tetromino can enter the board with its 4x4 window starting "
            "at column c."
        ),
        "grid": (
            "The settled 10x10 board after completed rows are removed; row 0 is "
            "the top, row 9 is the bottom, zero is empty, and nonzero is occupied."
        ),
        "step_count": (
            "Number of tetrominoes placed so far; the adapter corrects Jumanji "
            "1.1.1's upstream observation bug that otherwise reports constant zero."
        ),
        "tetromino": (
            "The current tetromino's unrotated 4x4 binary footprint. Rotation 0 "
            "uses this matrix exactly. For other rotations, crop all-zero outer "
            "rows and columns, rotate the cropped footprint clockwise, and "
            "top-left-align it in a zero-padded 4x4 matrix."
        ),
    },
    "cvrp": {
        "action_mask": (
            "Index i is true exactly when node i may be selected next. A customer "
            "must be unvisited with demand no greater than capacity; depot 0 is "
            "legal exactly when the vehicle is away from it."
        ),
        "capacity": (
            "Remaining vehicle capacity divided by raw capacity 30; returning to "
            "depot resets this scalar to 1.0."
        ),
        "coordinates": (
            "coordinates[i] is node i's [x, y] point in the unit square; index 0 "
            "is the depot and indices 1-20 are customers."
        ),
        "demands": (
            "demands[0] is 0; each customer demand is a raw integer from [1, 10) "
            "divided by vehicle capacity 30."
        ),
        "position": "Index of the vehicle's current node; 0 is the depot.",
        "trajectory": (
            "Visited node indices in order, beginning with depot 0. Unused suffix "
            "slots are also 0, so use visited customers and current position to "
            "distinguish padding from actual depot returns."
        ),
        "unvisited_nodes": (
            "For customers 1-20, true means not yet served. At index 0 this field "
            "means the depot is available, not that it is an unvisited customer."
        ),
    },
    "snake": {
        "action_mask": (
            "Entries 0/1/2/3 correspond to up/right/down/left and are true "
            "exactly when that next head cell is on-board and is not part of "
            "the body after the tail advances."
        ),
        "grid": (
            "A 12x12x5 float tensor indexed [row, column, channel], with row 0 "
            "at the top. Channels 0/1/2/3 are body occupancy, head one-hot, tail "
            "one-hot, and fruit one-hot; channel 4 is normalized body order. On body "
            "cells channel 4 decreases from 1.0 at the head to 1/length at the "
            "tail; it is 0 outside the body."
        ),
        "step_count": "Number of moves already applied, from 0 through 4000.",
    },
    "tsp": {
        "action_mask": (
            "Index i is true exactly when city i has not yet been selected and "
            "may be visited next."
        ),
        "coordinates": (
            "coordinates[i] is city i's [x, y] point sampled uniformly from "
            "the unit square [0, 1) x [0, 1)."
        ),
        "position": (
            "Index of the most recently selected city; -1 before the first "
            "selection."
        ),
        "trajectory": (
            "Selected city indices in visit order. Unused suffix entries are -1; "
            "the completed 20-city trajectory defines a cycle whose final city "
            "connects back to its first city."
        ),
    },
    "pacman": {
        "action_mask": (
            "Entries 0/1/2/3 mean up/left/down/right and are true when that "
            "move enters a traversable cell, including horizontal tunnel wrap. "
            "Entry 4 is the upstream no-op and is always false. The Host rejects "
            "false-mask actions instead of applying Jumanji's wall no-op behavior."
        ),
        "frightened_state_time": (
            "Raw upstream frightened timer. A power-up sets it to 30; positive "
            "values mean ghosts are edible. It decrements every other step and "
            "continues below zero while inactive."
        ),
        "ghost_locations": (
            "Four ghost coordinates stored as [column, row] pairs; unlike pellet "
            "arrays, these rows are always active and never use zero padding."
        ),
        "grid": (
            "Fixed maze indexed grid[row, column], shape 31x28. Value 1 is a "
            "traversable cell and 0 is a wall."
        ),
        "pellet_locations": (
            "318 [column, row] pairs, initially one for every traversable cell. "
            "A collected pellet's row is replaced by sentinel [0, 0], which is "
            "not a traversable maze cell."
        ),
        "player_locations.x": (
            "Despite the upstream field name, this scalar is the player's grid "
            "row index in [0, 31)."
        ),
        "player_locations.y": (
            "Despite the upstream field name, this scalar is the player's grid "
            "column index in [0, 28)."
        ),
        "power_up_locations": (
            "Four [column, row] pairs. A collected power-up's row is replaced by "
            "sentinel [0, 0]; every active power-up tile is also present in "
            "pellet_locations."
        ),
        "score": "Cumulative reward earned so far; final score equals Episode return.",
    },
}
_OBSERVATION_FIELDS: dict[str, dict[str, _ObservationFieldSpec]] = {
    "game-2048": {
        "action_mask": ("bool", (4,)),
        "board": ("int32", (4, 4)),
    },
    "graph-coloring": {
        "action_mask": ("bool", (20,)),
        "adj_matrix": ("bool", (20, 20)),
        "colors": ("int32", (20,)),
        "current_node_index": ("int64", ()),
    },
    "minesweeper": {
        "action_mask": ("bool", (10, 10)),
        "board": ("int32", (10, 10)),
        "num_mines": ("int64", ()),
        "step_count": ("int64", ()),
    },
    "rubiks-cube": _RUBIK_FIELDS,
    "rubiks-cube-partly-scrambled": _RUBIK_FIELDS,
    "sudoku": _SUDOKU_FIELDS,
    "sudoku-very-easy": _SUDOKU_FIELDS,
    "sliding-tile-puzzle": {
        "action_mask": ("bool", (4,)),
        "empty_tile_position": ("int32", (2,)),
        "puzzle": ("int32", (5, 5)),
        "step_count": ("int64", ()),
    },
    "bin-pack": {
        "action_mask": ("bool", (40, 20)),
        "ems.x1": ("float32", (40,)),
        "ems.x2": ("float32", (40,)),
        "ems.y1": ("float32", (40,)),
        "ems.y2": ("float32", (40,)),
        "ems.z1": ("float32", (40,)),
        "ems.z2": ("float32", (40,)),
        "ems_mask": ("bool", (40,)),
        "items.x_len": ("float32", (20,)),
        "items.y_len": ("float32", (20,)),
        "items.z_len": ("float32", (20,)),
        "items_mask": ("bool", (20,)),
        "items_placed": ("bool", (20,)),
    },
    "flat-pack": {
        "action_mask": ("bool", (25, 4, 9, 9)),
        "blocks": ("int32", (25, 3, 3)),
        "grid": ("int32", (11, 11)),
    },
    "job-shop": {
        "action_mask": ("bool", (10, 21)),
        "machines_job_ids": ("int32", (10,)),
        "machines_remaining_times": ("int32", (10,)),
        "ops_durations": ("int32", (20, 8)),
        "ops_machine_ids": ("int32", (20, 8)),
        "ops_mask": ("bool", (20, 8)),
    },
    "knapsack": {
        "action_mask": ("bool", (50,)),
        "packed_items": ("bool", (50,)),
        "values": ("float32", (50,)),
        "weights": ("float32", (50,)),
    },
    "tetris": {
        "action_mask": ("bool", (4, 10)),
        "grid": ("int32", (10, 10)),
        "step_count": ("int64", ()),
        "tetromino": ("int32", (4, 4)),
    },
    "cvrp": {
        "action_mask": ("bool", (21,)),
        "capacity": ("float64", ()),
        "coordinates": ("float32", (21, 2)),
        "demands": ("float32", (21,)),
        "position": ("int64", ()),
        "trajectory": ("int32", (40,)),
        "unvisited_nodes": ("bool", (21,)),
    },
    "maze": {
        "action_mask": ("bool", (4,)),
        "agent_position.col": ("int64", ()),
        "agent_position.row": ("int64", ()),
        "step_count": ("int64", ()),
        "target_position.col": ("int64", ()),
        "target_position.row": ("int64", ()),
        "walls": ("bool", (10, 10)),
    },
    "snake": {
        "action_mask": ("bool", (4,)),
        "grid": ("float32", (12, 12, 5)),
        "step_count": ("int64", ()),
    },
    "tsp": {
        "action_mask": ("bool", (20,)),
        "coordinates": ("float32", (20, 2)),
        "position": ("int64", ()),
        "trajectory": ("int32", (20,)),
    },
    "pacman": {
        "action_mask": ("bool", (5,)),
        "frightened_state_time": ("int64", ()),
        "ghost_locations": ("int32", (4, 2)),
        "grid": ("int32", (31, 28)),
        "pellet_locations": ("int32", (318, 2)),
        "player_locations.x": ("int64", ()),
        "player_locations.y": ("int64", ()),
        "power_up_locations": ("int32", (4, 2)),
        "score": ("int64", ()),
    },
}


@dataclass(frozen=True, slots=True)
class _TracedEpisode:
    episode_index: int
    record: EpisodeRecord
    step_indices: tuple[int, ...]
    config: JumanjiConfig

    @property
    def observation_artifact_name(self) -> str:
        return f"episode-{self.episode_index:03d}/observations.npz"


class JumanjiBenchmark:
    """Mean return for one fixed Jumanji profile."""

    def __init__(self, config: JumanjiConfig | None = None) -> None:
        if config is None:
            config = JumanjiConfig()
        if type(config) is not JumanjiConfig:
            raise TypeError("config must be JumanjiConfig")
        self._config = config
        self._spec = _spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(self, split: str, *, seed: int, count: int) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(EpisodeSpec(environment_seed=_seed(split, seed, index)) for index in range(count))

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return JumanjiEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        failure_return = -float(self._config.max_episode_steps)
        returns = tuple(
            record.total_reward if record.policy_failure is None else failure_return
            for record in records
        )
        score = statistics.fmean(returns)
        summarized = records[:_MAX_SUMMARIZED_EPISODES]
        traced = tuple(
            _TracedEpisode(
                episode_index=episode_index,
                record=record,
                step_indices=_trace_step_indices(record),
                config=self._config,
            )
            for episode_index, record in enumerate(records[:_MAX_TRACED_EPISODES])
        )
        observation_artifacts: list[Artifact] = []
        observation_manifests: list[PolicyValue] = []
        for episode in traced:
            artifact, fields = _observation_artifact(episode)
            observation_artifacts.append(artifact)
            observation_manifests.append(
                {
                    "episode_index": episode.episode_index,
                    "artifact": episode.observation_artifact_name,
                    "artifact_sha256": hashlib.sha256(artifact.content).hexdigest(),
                    "fields": fields,
                    "stored_transition_pairs": len(episode.step_indices),
                    "step_indices": list(episode.step_indices),
                    "omitted_steps": episode.record.steps - len(episode.step_indices),
                }
            )
        traced_steps = {
            episode.episode_index: len(episode.step_indices) for episode in traced
        }
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} "
                    f"{self._config.profile} Episodes."
                ),
                "profile": self._config.profile,
                "mean_return": score,
                "mean_steps": statistics.fmean(record.steps for record in records),
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "completed_episodes": sum(_completed(record) for record in records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "failure_return": failure_return,
                "episode_summaries": [
                    _episode_summary(
                        record,
                        episode_index=episode_index,
                        failure_return=failure_return,
                        traced_steps=traced_steps.get(episode_index, 0),
                        config=self._config,
                    )
                    for episode_index, record in enumerate(summarized)
                ],
                "summarized_episodes": len(summarized),
                "summary_episodes_omitted": len(records) - len(summarized),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "traced_steps": sum(traced_steps.values()),
                "trace_steps_omitted": sum(
                    episode.record.steps - len(episode.step_indices) for episode in traced
                ),
                "trace_step_cap_per_episode": _MAX_TRACED_STEPS_PER_EPISODE,
                "trace_selection": (
                    "Every short Episode is complete. Long Episodes retain the first and "
                    "last steps, bounded non-zero-reward and terminal events, and an even "
                    "sample of remaining steps."
                ),
                "trace_format": (
                    "trace.jsonl explains actions, action masks, named observation fields, "
                    "and profile progress. Every selected observation is stored losslessly "
                    "in a per-Episode observations.npz artifact."
                ),
                "observation_artifacts": observation_manifests,
            },
            artifacts=(
                _trace(traced, failure_return=failure_return),
                *observation_artifacts,
            ),
        )


def _spec(config: JumanjiConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.action_kind == "discrete":
        size = config.action_num_values[0]
        action_space = {
            "type": "discrete",
            "values": list(range(size)),
            "component": config.action_components[0],
            "meaning": {
                str(index): meaning
                for index, meaning in enumerate(config.discrete_action_meanings)
            },
            "masked": config.has_action_mask,
            "mask_layout": config.action_mask_layout,
        }
    else:
        action_space = {
            "type": "multi_discrete",
            "shape": [len(config.action_num_values)],
            "num_values": list(config.action_num_values),
            "components": list(config.action_components),
            "masked": config.has_action_mask,
            "mask_layout": config.action_mask_layout,
        }
        value_meanings = _multi_action_value_meanings(config)
        if value_meanings:
            action_space["component_value_meanings"] = value_meanings
    task_description = _PROFILE_TASK_DESCRIPTIONS.get(config.profile)
    description = f"Solve Jumanji's {config.environment_id} {config.category} task."
    if task_description is not None:
        description += f" {task_description}"
    description += " Maximize mean upstream return across independently seeded instances."
    environment_parameters: dict[str, PolicyValue] = {
        "profile": config.profile,
        "category": config.category,
        "environment_id": config.environment_id,
        "action_kind": config.action_kind,
        "action_num_values": list(config.action_num_values),
        "action_components": list(config.action_components),
        "has_action_mask": config.has_action_mask,
        "action_mask_layout": config.action_mask_layout,
        "initial_scramble_moves": config.initial_scramble_moves,
    }
    if config.profile == "knapsack":
        environment_parameters["total_capacity"] = _KNAPSACK_CAPACITY
    if config.profile == "cvrp":
        environment_parameters.update(
            {
                "customer_count": 20,
                "depot_index": 0,
                "raw_vehicle_capacity": 30,
                "raw_demand_minimum": 1,
                "raw_demand_maximum_exclusive": 10,
            }
        )
    if config.profile == "tsp":
        environment_parameters.update(
            {
                "city_count": 20,
                "coordinate_minimum": 0.0,
                "coordinate_maximum_exclusive": 1.0,
                "reward_mode": "dense",
                "tour_returns_to_first_city": True,
            }
        )
    if config.profile == "snake":
        environment_parameters.update(
            {
                "grid_rows": 12,
                "grid_columns": 12,
                "grid_cells": 144,
                "initial_snake_length": 1,
                "fruit_reward": 1.0,
                "time_limit": 4_000,
                "grid_channels": [
                    "body",
                    "head",
                    "tail",
                    "fruit",
                    "normalized_body_order",
                ],
            }
        )
    if config.profile == "pacman":
        environment_parameters.update(
            {
                "grid_rows": 31,
                "grid_columns": 28,
                "walkable_cells": 318,
                "initial_pellets": 318,
                "initial_power_ups": 4,
                "ghost_count": 4,
                "time_limit": 1_000,
                "regular_pellet_reward": 10,
                "power_up_extra_reward": 50,
                "eligible_ghost_reward": 200,
                "frightened_duration": 30,
                "maze_layout": "fixed",
            }
        )
    if config.profile.startswith("rubiks-cube"):
        environment_parameters.update(
            {
                "cube_size": 3,
                "face_count": 6,
                "stickers_per_face": 9,
                "face_order": list(_RUBIK_FACES),
                "sticker_value_colors": {
                    str(index): color
                    for index, color in enumerate(_RUBIK_STICKER_COLORS)
                },
                "face_view_orientations": dict(_RUBIK_FACE_VIEW_ORIENTATIONS),
                "scramble_sampling": (
                    "independent_uniform_legal_actions_with_replacement"
                ),
                "outer_layer_depth": 0,
                "legal_action_count": 18,
                "all_actions_always_legal": True,
                "solved_reward": 1,
                "otherwise_reward": 0,
                "misplaced_stickers_is_solution_distance": False,
            }
        )
    if config.profile.startswith("sudoku"):
        if config.profile == "sudoku":
            database_size, minimum_clues, maximum_clues = 10_000, 25, 77
        else:
            database_size, minimum_clues, maximum_clues = 1_000, 46, 80
        environment_parameters.update(
            {
                "board_size": 9,
                "box_rows": 3,
                "box_columns": 3,
                "puzzle_database_size": database_size,
                "minimum_initial_clues": minimum_clues,
                "maximum_initial_clues": maximum_clues,
                "empty_cell_value": -1,
                "internal_symbol_minimum": 0,
                "internal_symbol_maximum": 8,
                "solved_reward": 1,
            }
        )
    return BenchmarkSpec(
        id=f"jumanji/{config.environment_id}/mean-return-v1",
        description=description,
        observation_space=_observation_space(config),
        action_space=action_space,
        metadata={
            "environment": config.environment_id,
            "provider": "Jumanji",
            "upstream_version": "1.1.1",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters=environment_parameters,
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_space(config: JumanjiConfig) -> PolicyValue:
    fields: dict[str, PolicyValue] = {}
    for field, (dtype, shape) in _OBSERVATION_FIELDS[config.profile].items():
        if shape:
            field_spec: dict[str, PolicyValue] = {
                "policy_carrier": "TensorValue",
                "dtype": dtype,
                "shape": list(shape),
            }
            if field == "cube" and config.profile.startswith("rubiks-cube"):
                field_spec.update(
                    {
                        "axes": ["face", "row", "column"],
                        "face_order": list(_RUBIK_FACES),
                        "sticker_value_colors": {
                            str(index): color
                            for index, color in enumerate(_RUBIK_STICKER_COLORS)
                        },
                        "solved_state": (
                            "each face is uniform; in every reachable state its fixed "
                            "center means face index i contains sticker value i"
                        ),
                        "face_cells": "reading order while looking directly at each face",
                        "face_view_orientations": dict(
                            _RUBIK_FACE_VIEW_ORIENTATIONS
                        ),
                    }
                )
        else:
            field_spec = {
                "policy_carrier": _scalar_carrier(dtype),
                "type": _scalar_json_type(dtype),
                "npz_dtype": dtype,
                "npz_shape": [],
            }
        meaning = _OBSERVATION_FIELD_MEANINGS.get(config.profile, {}).get(field)
        if meaning is not None:
            field_spec["meaning"] = meaning
        field_spec["policy_path"] = list[PolicyValue](field.split("."))
        fields[field] = field_spec
    return {
        "type": "object",
        "encoding": (
            "The live Policy observation is a nested dictionary. Field-table keys are "
            "flattened schema names only; traverse each field's policy_path to read it."
        ),
        "fields": fields,
        "policy_path_rule": (
            "For example, policy_path ['ems', 'x1'] means "
            "observation['ems']['x1'], never observation['ems.x1']."
        ),
        "policy_leaf_rule": (
            "Shape-[] values are exact Python scalars. Non-scalar arrays are TensorValue "
            "instances. NPZ feedback materializes both carrier kinds as NumPy arrays."
        ),
        "includes_action_mask": config.has_action_mask,
        "feedback_encoding": (
            "lossless named arrays in observations.npz plus bounded semantic summaries"
        ),
    }


def _scalar_carrier(dtype: str) -> str:
    if dtype == "bool":
        return "bool"
    if dtype.startswith(("int", "uint")):
        return "int"
    if dtype.startswith("float"):
        return "float"
    raise ValueError("Jumanji scalar schema dtype is invalid")


def _scalar_json_type(dtype: str) -> str:
    if dtype == "bool":
        return "boolean"
    if dtype.startswith(("int", "uint")):
        return "integer"
    if dtype.startswith("float"):
        return "number"
    raise ValueError("Jumanji scalar schema dtype is invalid")


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _completed(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.done
    )


def _terminated(record: EpisodeRecord) -> bool:
    return bool(record.transitions and record.transitions[-1].step.terminated)


def _truncated(record: EpisodeRecord) -> bool:
    return bool(record.transitions and record.transitions[-1].step.truncated)


def _episode_summary(
    record: EpisodeRecord,
    *,
    episode_index: int,
    failure_return: float,
    traced_steps: int,
    config: JumanjiConfig,
) -> PolicyValue:
    action_counts: dict[str, int] = {}
    action_values: dict[str, PolicyValue] = {}
    reward_events = 0
    for transition in record.transitions:
        action = _trace_action(transition.action, config=config)
        key = json.dumps(action, separators=(",", ":"), sort_keys=True)
        action_counts[key] = action_counts.get(key, 0) + 1
        action_values[key] = action
        if transition.step.reward != 0.0:
            reward_events += 1
    ranked_actions = sorted(
        action_counts,
        key=lambda key: (-action_counts[key], key),
    )
    summarized_actions = ranked_actions[:_MAX_ACTION_SUMMARIES]
    omitted_actions = ranked_actions[_MAX_ACTION_SUMMARIES:]
    return {
        "episode_index": episode_index,
        "status": "completed" if record.policy_failure is None else "policy_failed",
        "return": record.total_reward if record.policy_failure is None else None,
        "scored_return": (
            record.total_reward if record.policy_failure is None else failure_return
        ),
        "steps": record.steps,
        "terminated": _terminated(record),
        "truncated": _truncated(record),
        "failure": record.policy_failure,
        "nonzero_reward_steps": reward_events,
        "actions": [
            {
                "action": action_values[key],
                "meaning": _action_meaning(action_values[key], config=config),
                "count": action_counts[key],
            }
            for key in summarized_actions
        ],
        "distinct_actions": len(ranked_actions),
        "action_summaries_omitted": len(omitted_actions),
        "action_steps_omitted": sum(action_counts[key] for key in omitted_actions),
        "metrics": _metric_summaries(record),
        "traced_steps": traced_steps,
        "trace_steps_omitted": record.steps - traced_steps,
    }


def _metric_summaries(record: EpisodeRecord) -> PolicyValue:
    values: dict[str, list[PolicyValue]] = {}
    for transition in record.transitions:
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            continue
        for name, value in metrics.items():
            if type(value) in {bool, int, float, str}:
                values.setdefault(name, []).append(value)
    summaries: dict[str, PolicyValue] = {}
    for name, items in sorted(values.items()):
        if items and all(type(item) is bool for item in items):
            summaries[name] = {
                "true_steps": sum(item is True for item in items),
                "final": items[-1],
            }
        elif items and all(type(item) in {int, float} for item in items):
            numeric = [
                float(item)
                for item in items
                if type(item) is int or type(item) is float
            ]
            summaries[name] = {
                "minimum": min(numeric),
                "mean": statistics.fmean(numeric),
                "maximum": max(numeric),
                "sum": sum(numeric),
                "final": numeric[-1],
            }
        elif items:
            summaries[name] = {"final": items[-1]}
    return summaries


def _trace_step_indices(record: EpisodeRecord) -> tuple[int, ...]:
    if record.steps <= _MAX_TRACED_STEPS_PER_EPISODE:
        return tuple(range(record.steps))
    selected = set(range(_TRACE_EDGE_STEPS))
    selected.update(range(record.steps - _TRACE_EDGE_STEPS, record.steps))
    event_steps = tuple(
        step_index
        for step_index, transition in enumerate(record.transitions)
        if _event_transition(transition) and step_index not in selected
    )
    selected.update(_even_sample(event_steps, _MAX_EVENT_STEPS))
    remaining_capacity = _MAX_TRACED_STEPS_PER_EPISODE - len(selected)
    remaining_steps = tuple(
        step_index for step_index in range(record.steps) if step_index not in selected
    )
    selected.update(_even_sample(remaining_steps, remaining_capacity))
    return tuple(sorted(selected))


def _event_transition(transition: Transition) -> bool:
    return bool(
        transition.step.reward != 0.0
        or transition.step.terminated
        or transition.step.truncated
    )


def _even_sample(values: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0 or not values:
        return ()
    if len(values) <= count:
        return tuple(values)
    if count == 1:
        return (values[len(values) // 2],)
    return tuple(
        values[index * (len(values) - 1) // (count - 1)] for index in range(count)
    )


def _trace(
    episodes: Sequence[_TracedEpisode],
    *,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode in episodes:
        record = episode.record
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode.episode_index,
                    "profile": episode.config.profile,
                    "status": (
                        "completed" if record.policy_failure is None else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": record.total_reward if record.policy_failure is None else None,
                    "scored_return": (
                        record.total_reward
                        if record.policy_failure is None
                        else failure_return
                    ),
                    "failure": record.policy_failure,
                    "traced_steps": len(episode.step_indices),
                    "omitted_steps": record.steps - len(episode.step_indices),
                    "initial_observation": _observation_reference(
                        episode,
                        observation=record.initial_observation,
                        kind="initial",
                        trace_index=None,
                    ),
                }
            )
        )
        for trace_index, step_index in enumerate(episode.step_indices):
            transition = record.transitions[step_index]
            action = _trace_action(transition.action, config=episode.config)
            decision_observation = (
                record.initial_observation
                if step_index == 0
                else record.transitions[step_index - 1].step.observation
            )
            decision_fields = _observation_fields(decision_observation)
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode.episode_index,
                        "step_index": step_index,
                        "action": action,
                        "action_meaning": _action_meaning(action, config=episode.config),
                        "action_was_legal": _action_was_legal(
                            action,
                            fields=decision_fields,
                            config=episode.config,
                        ),
                        "reward": transition.step.reward,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": transition.step.metrics,
                        "event": _event_transition(transition),
                        "decision_observation": _observation_reference(
                            episode,
                            observation=decision_observation,
                            kind="decision",
                            trace_index=trace_index,
                        ),
                        "result_observation": _observation_reference(
                            episode,
                            observation=transition.step.observation,
                            kind="result",
                            trace_index=trace_index,
                        ),
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(action: PolicyValue, *, config: JumanjiConfig) -> PolicyValue:
    if config.action_kind == "discrete":
        if type(action) is not int or not 0 <= action < config.action_num_values[0]:
            raise ValueError("Jumanji trace Action is invalid")
        return action
    if type(action) is not list or len(action) != len(config.action_num_values):
        raise ValueError("Jumanji trace Action is invalid")
    traced: list[PolicyValue] = []
    for item, size in zip(action, config.action_num_values, strict=True):
        if type(item) is not int or not 0 <= item < size:
            raise ValueError("Jumanji trace Action is invalid")
        traced.append(item)
    return traced


def _action_meaning(action: PolicyValue, *, config: JumanjiConfig) -> str:
    if type(action) is int:
        if config.discrete_action_meanings:
            return config.discrete_action_meanings[action]
        return f"{config.action_components[0]}={action}"
    if type(action) is not list:
        raise ValueError("Jumanji trace Action meaning is invalid")
    if any(type(item) is not int for item in action):
        raise ValueError("Jumanji trace Action meaning is invalid")
    integer_action = [item for item in action if type(item) is int]
    if config.profile.startswith("rubiks-cube"):
        return (
            f"face={_RUBIK_FACES[integer_action[0]]}({integer_action[0]}),"
            f"depth={integer_action[1]},"
            f"rotation={_RUBIK_ROTATIONS[integer_action[2]]}({integer_action[2]})"
        )
    if config.profile == "job-shop":
        no_op = config.action_num_values[0] - 1
        return ",".join(
            f"{name}={'no_op' if value == no_op else value}"
            for name, value in zip(
                config.action_components, integer_action, strict=True
            )
        )
    if config.profile == "tetris":
        return (
            f"rotation={integer_action[0] * 90}_degrees_clockwise({integer_action[0]}),"
            f"window_left_column={integer_action[1]}"
        )
    if config.profile.startswith("sudoku"):
        return (
            f"row={integer_action[0]},column={integer_action[1]},"
            f"value={integer_action[2]}(human_symbol={integer_action[2] + 1})"
        )
    return ",".join(
        f"{name}={value}"
        for name, value in zip(config.action_components, integer_action, strict=True)
    )


def _multi_action_value_meanings(config: JumanjiConfig) -> PolicyValue:
    if config.profile.startswith("rubiks-cube"):
        return {
            "face": {str(index): value for index, value in enumerate(_RUBIK_FACES)},
            "depth": "zero-based layer depth from the selected face",
            "rotation": {
                str(index): value for index, value in enumerate(_RUBIK_ROTATIONS)
            },
        }
    if config.profile == "job-shop":
        return {
            "each_machine_job": "values 0-19 select a job; value 20 is no_op"
        }
    if config.profile.startswith("sudoku"):
        return {
            "row": "zero-based row 0-8",
            "column": "zero-based column 0-8",
            "value": {
                str(index): f"human_symbol_{index + 1}" for index in range(9)
            },
        }
    if config.profile == "tetris":
        return {
            "rotation": {
                str(index): f"{index * 90}_degrees_clockwise"
                for index in range(4)
            },
            "column": "left column of the tetromino's re-anchored 4x4 window",
        }
    return {}


def _action_was_legal(
    action: PolicyValue,
    *,
    fields: dict[str, NDArray[numpy.generic]],
    config: JumanjiConfig,
) -> bool:
    if not config.has_action_mask:
        return True
    mask = fields.get("action_mask")
    if mask is None or mask.dtype != numpy.dtype(bool):
        raise ValueError("Jumanji trace action mask is invalid")
    if type(action) is int:
        if mask.shape != config.action_num_values:
            raise ValueError("Jumanji trace action mask shape changed")
        return bool(mask[action])
    if type(action) is not list:
        raise ValueError("Jumanji trace Action is invalid")
    if any(type(item) is not int for item in action):
        raise ValueError("Jumanji trace Action is invalid")
    integer_action = [item for item in action if type(item) is int]
    indices = tuple(integer_action)
    if config.action_mask_layout == "joint":
        if mask.shape != config.action_num_values:
            raise ValueError("Jumanji trace action mask shape changed")
        return bool(mask[indices])
    expected = (len(config.action_num_values), config.action_num_values[0])
    if mask.shape != expected:
        raise ValueError("Jumanji trace action mask shape changed")
    return all(bool(mask[index, item]) for index, item in enumerate(integer_action))


def _observation_artifact(
    episode: _TracedEpisode,
) -> tuple[Artifact, list[PolicyValue]]:
    record = episode.record
    initial = _observation_fields(record.initial_observation)
    _check_observation_schema(initial, config=episode.config)
    decisions = tuple(
        initial
        if step_index == 0
        else _observation_fields(record.transitions[step_index - 1].step.observation)
        for step_index in episode.step_indices
    )
    results = tuple(
        _observation_fields(record.transitions[step_index].step.observation)
        for step_index in episode.step_indices
    )
    arrays: dict[str, object] = {
        "step_indices": numpy.asarray(episode.step_indices, dtype=numpy.int32)
    }
    manifests: list[PolicyValue] = []
    for field, initial_array in initial.items():
        initial_name = f"initial__{field}"
        decision_name = f"decision__{field}"
        result_name = f"result__{field}"
        arrays[initial_name] = initial_array
        arrays[decision_name] = _field_array(
            decisions,
            field=field,
            template=initial_array,
            expected_fields=set(initial),
        )
        arrays[result_name] = _field_array(
            results,
            field=field,
            template=initial_array,
            expected_fields=set(initial),
        )
        manifests.append(
            {
                "field": field,
                "policy_path": list[PolicyValue](field.split(".")),
                "policy_carrier": _policy_carrier(initial_array),
                "dtype": initial_array.dtype.name,
                "shape": list(initial_array.shape),
                "initial_array": initial_name,
                "decision_array": decision_name,
                "result_array": result_name,
            }
        )
    buffer = io.BytesIO()
    numpy.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return (
        Artifact(
            name=episode.observation_artifact_name,
            media_type="application/x-npz",
            content=buffer.getvalue(),
        ),
        manifests,
    )


def _field_array(
    observations: Sequence[dict[str, NDArray[numpy.generic]]],
    *,
    field: str,
    template: NDArray[numpy.generic],
    expected_fields: set[str],
) -> NDArray[numpy.generic]:
    arrays: list[NDArray[numpy.generic]] = []
    for observation in observations:
        if set(observation) != expected_fields:
            raise ValueError("Jumanji trace observation fields changed")
        array = observation.get(field)
        if array is None or array.dtype != template.dtype or array.shape != template.shape:
            raise ValueError("Jumanji trace observation field changed")
        arrays.append(array)
    if not arrays:
        return numpy.empty((0, *template.shape), dtype=template.dtype)
    return numpy.stack(arrays)


def _observation_reference(
    episode: _TracedEpisode,
    *,
    observation: PolicyValue,
    kind: str,
    trace_index: int | None,
) -> dict[str, object]:
    fields = _observation_fields(observation)
    _check_observation_schema(fields, config=episode.config)
    references: list[dict[str, object]] = []
    for field in fields:
        reference: dict[str, object] = {
            "field": field,
            "policy_path": field.split("."),
            "array": f"{kind}__{field}",
        }
        if trace_index is not None:
            reference["index"] = trace_index
        references.append(reference)
    return {
        "artifact": episode.observation_artifact_name,
        "fields": references,
        "semantics": _observation_semantics(fields, config=episode.config),
    }


def _observation_fields(value: PolicyValue) -> dict[str, NDArray[numpy.generic]]:
    fields: dict[str, NDArray[numpy.generic]] = {}
    _flatten_observation(value, path="", fields=fields)
    if not fields:
        raise ValueError("Jumanji trace observation has no fields")
    return dict(sorted(fields.items()))


def _check_observation_schema(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    config: JumanjiConfig,
) -> None:
    expected = _OBSERVATION_FIELDS[config.profile]
    if set(fields) != set(expected):
        raise ValueError("Jumanji trace observation fields drifted")
    for field, (dtype, shape) in expected.items():
        array = fields[field]
        if array.dtype.name != dtype or array.shape != shape:
            raise ValueError(f"Jumanji trace observation field {field!r} drifted")


def _flatten_observation(
    value: PolicyValue,
    *,
    path: str,
    fields: dict[str, NDArray[numpy.generic]],
) -> None:
    if type(value) is dict:
        for key in sorted(value):
            if not key or "." in key:
                raise ValueError("Jumanji trace observation field name is invalid")
            child_path = f"{path}.{key}" if path else key
            _flatten_observation(value[key], path=child_path, fields=fields)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"root[{index}]"
            _flatten_observation(item, path=child_path, fields=fields)
        return
    if type(value) is tuple:
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"root[{index}]"
            _flatten_observation(item, path=child_path, fields=fields)
        return
    field = path or "root"
    if field in fields:
        raise ValueError("Jumanji trace observation field is duplicated")
    fields[field] = _observation_array(value, field=field)


def _observation_array(
    value: PolicyValue,
    *,
    field: str,
) -> NDArray[numpy.generic]:
    if type(value) is TensorValue:
        dtype = numpy.dtype(value.dtype)
        expected_bytes = math.prod(value.shape) * dtype.itemsize
        if len(value.data) != expected_bytes:
            raise ValueError(f"Jumanji trace observation field {field!r} is invalid")
        array = numpy.frombuffer(value.data, dtype=dtype).reshape(value.shape)
    elif type(value) is bool:
        array = numpy.asarray(value, dtype=numpy.bool_)
    elif type(value) is int:
        try:
            array = numpy.asarray(value, dtype=numpy.int64)
        except OverflowError:
            raise ValueError(
                f"Jumanji trace observation field {field!r} is out of range"
            ) from None
    elif type(value) is float:
        array = numpy.asarray(value, dtype=numpy.float64)
    elif type(value) is str:
        array = numpy.asarray(value, dtype=f"<U{max(1, len(value))}")
    elif type(value) is bytes:
        array = numpy.frombuffer(value, dtype=numpy.uint8)
    else:
        raise ValueError(f"Jumanji trace observation field {field!r} is invalid")
    if numpy.issubdtype(array.dtype, numpy.floating) and not numpy.isfinite(array).all():
        raise ValueError(f"Jumanji trace observation field {field!r} is non-finite")
    return array


def _observation_semantics(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    config: JumanjiConfig,
) -> dict[str, object]:
    return {
        "kind": config.profile,
        "field_summaries": [
            _field_summary(field, array)
            for field, array in fields.items()
            if field != "action_mask"
        ],
        "action_mask": _action_mask_semantics(fields, config=config),
        "progress": _profile_progress(fields, config=config),
    }


def _field_summary(field: str, array: NDArray[numpy.generic]) -> dict[str, object]:
    summary: dict[str, object] = {
        "field": field,
        "policy_path": field.split("."),
        "policy_carrier": _policy_carrier(array),
        "dtype": array.dtype.name,
        "shape": list(array.shape),
    }
    if array.shape == ():
        summary["value"] = _public_scalar(array.item())
    elif numpy.issubdtype(array.dtype, numpy.bool_):
        summary["true_values"] = int(numpy.count_nonzero(array))
    elif numpy.issubdtype(array.dtype, numpy.number):
        summary["minimum"] = _public_scalar(array.min().item())
        summary["maximum"] = _public_scalar(array.max().item())
        summary["nonzero_values"] = int(numpy.count_nonzero(array))
    else:
        summary["values"] = int(array.size)
    return summary


def _policy_carrier(array: NDArray[numpy.generic]) -> str:
    if array.shape:
        return "TensorValue"
    return _scalar_carrier(array.dtype.name)


def _public_scalar(value: object) -> bool | int | float | str:
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Jumanji trace observation is non-finite")
        return result
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value
    if type(value) is str:
        return value
    raise ValueError("Jumanji trace observation scalar is invalid")


def _action_mask_semantics(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    config: JumanjiConfig,
) -> dict[str, object] | None:
    if not config.has_action_mask:
        return None
    mask = fields.get("action_mask")
    if mask is None or mask.dtype != numpy.dtype(bool):
        raise ValueError("Jumanji trace action mask is invalid")
    if config.action_mask_layout in {"discrete", "joint"}:
        if mask.shape != config.action_num_values:
            raise ValueError("Jumanji trace action mask shape changed")
        coordinates = numpy.argwhere(mask)
        if config.action_mask_layout == "discrete":
            samples: object = [int(index[0]) for index in coordinates[:_MAX_LEGAL_ACTION_SAMPLES]]
        else:
            samples = [
                [int(item) for item in coordinate]
                for coordinate in coordinates[:_MAX_LEGAL_ACTION_SAMPLES]
            ]
        return {
            "layout": config.action_mask_layout,
            "shape": list(mask.shape),
            "legal_action_count": int(coordinates.shape[0]),
            "legal_action_samples": samples,
            "legal_actions_omitted": max(
                0, int(coordinates.shape[0]) - _MAX_LEGAL_ACTION_SAMPLES
            ),
        }
    expected = (len(config.action_num_values), config.action_num_values[0])
    if mask.shape != expected:
        raise ValueError("Jumanji trace action mask shape changed")
    components: list[dict[str, object]] = []
    legal_action_count = 1
    for index, name in enumerate(config.action_components):
        valid = numpy.flatnonzero(mask[index])
        legal_action_count *= int(valid.size)
        components.append(
            {
                "component": name,
                "legal_value_count": int(valid.size),
                "legal_values": [int(item) for item in valid[:_MAX_LEGAL_ACTION_SAMPLES]],
                "legal_values_omitted": max(
                    0, int(valid.size) - _MAX_LEGAL_ACTION_SAMPLES
                ),
            }
        )
    return {
        "layout": "per_component",
        "shape": list(mask.shape),
        "legal_action_count": legal_action_count,
        "components": components,
    }


def _profile_progress(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    config: JumanjiConfig,
) -> dict[str, object]:
    profile = config.profile
    if profile == "game-2048":
        board = _array(fields, "board")
        highest_exponent = int(board.max(initial=0))
        return {
            "highest_tile": 2**highest_exponent if numpy.any(board) else 0,
            "highest_tile_exponent": highest_exponent,
            "occupied_cells": int(numpy.count_nonzero(board)),
        }
    if profile == "graph-coloring":
        colors = _array(fields, "colors").astype(numpy.int64, copy=False)
        return {
            "colored_nodes": int(numpy.count_nonzero(colors >= 0)),
            "total_nodes": int(colors.size),
            "current_node": _integer(fields, "current_node_index"),
        }
    if profile == "minesweeper":
        mines_board = _array(fields, "board").astype(numpy.int64, copy=False)
        return {
            "revealed_cells": int(numpy.count_nonzero(mines_board >= 0)),
            "total_cells": int(mines_board.size),
            "mines": _integer(fields, "num_mines"),
            "step": _integer(fields, "step_count"),
        }
    if profile.startswith("rubiks-cube"):
        cube = _array(fields, "cube")
        solved = numpy.arange(cube.shape[0], dtype=cube.dtype)[:, None, None]
        misplaced_stickers = int(numpy.count_nonzero(cube != solved))
        per_face_correct_stickers = [
            int(numpy.count_nonzero(cube[index] == index))
            for index in range(cube.shape[0])
        ]
        uniform_faces = sum(
            bool(numpy.all(face == face.reshape(-1)[0])) for face in cube
        )
        color_counts = [
            int(numpy.count_nonzero(cube == color))
            for color in range(cube.shape[0])
        ]
        return {
            "uniform_faces": uniform_faces,
            "faces": int(cube.shape[0]),
            "correct_stickers": int(cube.size) - misplaced_stickers,
            "misplaced_stickers": misplaced_stickers,
            "misplaced_stickers_is_solution_distance": False,
            "correct_stickers_by_face": per_face_correct_stickers,
            "sticker_counts_by_color": color_counts,
            "valid_sticker_inventory": all(count == 9 for count in color_counts),
            "solved": uniform_faces == int(cube.shape[0]),
            "step": _integer(fields, "step_count"),
        }
    if profile.startswith("sudoku"):
        sudoku_board = _array(fields, "board").astype(numpy.int64, copy=False)
        action_mask = _array(fields, "action_mask")
        if sudoku_board.shape != (9, 9) or action_mask.shape != (9, 9, 9):
            raise ValueError("Jumanji Sudoku observation is invalid")
        empty = sudoku_board < 0
        candidate_counts = numpy.count_nonzero(action_mask, axis=-1)
        empty_candidate_counts = candidate_counts[empty]
        filled_cells = int(numpy.count_nonzero(~empty))
        solved_board = filled_cells == int(
            sudoku_board.size
        ) and _sudoku_board_is_solved(sudoku_board)
        return {
            "filled_cells": filled_cells,
            "empty_cells": int(numpy.count_nonzero(empty)),
            "total_cells": int(sudoku_board.size),
            "legal_assignments": int(numpy.count_nonzero(action_mask)),
            "forced_empty_cells": int(
                numpy.count_nonzero(empty & (candidate_counts == 1))
            ),
            "zero_candidate_empty_cells": int(
                numpy.count_nonzero(empty & (candidate_counts == 0))
            ),
            "minimum_candidates_per_empty_cell": (
                int(empty_candidate_counts.min()) if empty_candidate_counts.size else 0
            ),
            "maximum_candidates_per_empty_cell": (
                int(empty_candidate_counts.max()) if empty_candidate_counts.size else 0
            ),
            "candidate_counts": [
                [int(count) for count in row] for row in candidate_counts
            ],
            "solved": solved_board,
        }
    if profile == "sliding-tile-puzzle":
        puzzle = _array(fields, "puzzle").reshape(-1)
        solved = numpy.arange(1, puzzle.size + 1)
        solved[-1] = 0
        correctly_positioned = int(numpy.count_nonzero(puzzle == solved))
        return {
            "correctly_positioned_tiles": correctly_positioned,
            "total_tiles": int(puzzle.size),
            "solved": correctly_positioned == puzzle.size,
            "empty_tile_position": [
                int(item) for item in _array(fields, "empty_tile_position").reshape(-1)
            ],
            "step": _integer(fields, "step_count"),
        }
    if profile == "bin-pack":
        return {
            "packed_items": int(numpy.count_nonzero(_array(fields, "items_placed"))),
            "available_items": int(numpy.count_nonzero(_array(fields, "items_mask"))),
            "active_empty_maximal_spaces": int(
                numpy.count_nonzero(_array(fields, "ems_mask"))
            ),
        }
    if profile == "flat-pack":
        grid = _array(fields, "grid")
        occupied_cells = int(numpy.count_nonzero(grid))
        return {
            "occupied_grid_cells": occupied_cells,
            "grid_cells": int(grid.size),
            "grid_occupancy": occupied_cells / int(grid.size),
            "solved": occupied_cells == grid.size,
        }
    if profile == "job-shop":
        return {
            "remaining_operations": int(numpy.count_nonzero(_array(fields, "ops_mask"))),
            "busy_machines": int(
                numpy.count_nonzero(
                    _array(fields, "machines_remaining_times").astype(
                        numpy.float64, copy=False
                    )
                    > 0
                )
            ),
        }
    if profile == "knapsack":
        packed = _array(fields, "packed_items").astype(bool, copy=False)
        weights = _array(fields, "weights")
        values = _array(fields, "values")
        packed_weight = float(weights[packed].sum())
        return {
            "packed_items": int(numpy.count_nonzero(packed)),
            "legal_items": int(numpy.count_nonzero(_array(fields, "action_mask"))),
            "total_capacity": _KNAPSACK_CAPACITY,
            "packed_weight": packed_weight,
            "remaining_capacity": max(0.0, _KNAPSACK_CAPACITY - packed_weight),
            "packed_value": float(values[packed].sum()),
        }
    if profile == "tetris":
        grid = _array(fields, "grid")
        occupied = grid != 0
        heights: list[int] = []
        holes = 0
        for column in occupied.T:
            filled_rows = numpy.flatnonzero(column)
            if filled_rows.size == 0:
                heights.append(0)
                continue
            top = int(filled_rows[0])
            heights.append(int(grid.shape[0]) - top)
            holes += int(numpy.count_nonzero(~column[top:]))
        return {
            "occupied_grid_cells": int(numpy.count_nonzero(grid)),
            "grid_cells": int(grid.size),
            "column_heights": heights,
            "aggregate_height": sum(heights),
            "maximum_height": max(heights),
            "holes_below_surface": holes,
            "surface_bumpiness": sum(
                abs(left - right)
                for left, right in zip(heights, heights[1:], strict=False)
            ),
            "step": _integer(fields, "step_count"),
        }
    if profile == "cvrp":
        unvisited = _array(fields, "unvisited_nodes").astype(bool, copy=False)
        position = _integer(fields, "position")
        route = _cvrp_route(fields, position=position, unvisited=unvisited)
        coordinates = _array(fields, "coordinates").astype(
            numpy.float64, copy=False
        )
        traveled = float(
            numpy.linalg.norm(
                coordinates[route[1:]] - coordinates[route[:-1]],
                axis=1,
            ).sum()
        )
        return {
            "current_node": position,
            "at_depot": position == 0,
            "visited_customers": int(numpy.count_nonzero(~unvisited[1:])),
            "unvisited_customers": int(numpy.count_nonzero(unvisited[1:])),
            "remaining_capacity": _number(fields, "capacity"),
            "remaining_total_demand": float(_array(fields, "demands")[unvisited].sum()),
            "route_visits": len(route) - 1,
            "depot_returns": int(numpy.count_nonzero(route[1:] == 0)),
            "distance_traveled": traveled,
            "route": [int(node) for node in route],
        }
    if profile == "maze":
        agent = (
            _integer(fields, "agent_position.row"),
            _integer(fields, "agent_position.col"),
        )
        target = (
            _integer(fields, "target_position.row"),
            _integer(fields, "target_position.col"),
        )
        return {
            "agent_position": list(agent),
            "target_position": list(target),
            "manhattan_distance": abs(agent[0] - target[0]) + abs(agent[1] - target[1]),
            "step": _integer(fields, "step_count"),
        }
    if profile == "snake":
        snake_grid = _array(fields, "grid").astype(numpy.float64, copy=False)
        if snake_grid.shape != (12, 12, 5):
            raise ValueError("Jumanji Snake grid is invalid")
        body = snake_grid[..., 0] > 0.5
        head = _single_grid_position(snake_grid[..., 1] > 0.5, name="head")
        tail = _single_grid_position(snake_grid[..., 2] > 0.5, name="tail")
        fruit = _single_grid_position(snake_grid[..., 3] > 0.5, name="fruit")
        length = int(numpy.count_nonzero(body))
        normalized_order = snake_grid[..., 4]
        body_positions = numpy.argwhere(body)
        if (
            length < 1
            or not body[tuple(head)]
            or not body[tuple(tail)]
            or numpy.any(normalized_order[~body] != 0.0)
            or numpy.any(normalized_order[body] <= 0.0)
            or numpy.any(normalized_order[body] > 1.0)
        ):
            raise ValueError("Jumanji Snake body encoding is invalid")
        body_path = sorted(
            ([int(row), int(column)] for row, column in body_positions),
            key=lambda position: float(normalized_order[tuple(position)]),
            reverse=True,
        )
        if body_path[0] != head or body_path[-1] != tail:
            raise ValueError("Jumanji Snake body order is invalid")
        return {
            "head_position": head,
            "tail_position": tail,
            "fruit_position": fruit,
            "snake_length": length,
            "fruits_eaten": length - 1,
            "free_cells": int(body.size) - length,
            "head_to_fruit_manhattan_distance": abs(head[0] - fruit[0])
            + abs(head[1] - fruit[1]),
            "body_path_head_to_tail": body_path,
            "board_full": length == int(body.size),
            "step": _integer(fields, "step_count"),
        }
    if profile == "tsp":
        mask = _array(fields, "action_mask")
        trajectory = _array(fields, "trajectory").astype(numpy.int64, copy=False)
        total_nodes = int(mask.size)
        visited_nodes = total_nodes - int(numpy.count_nonzero(mask))
        route = trajectory[:visited_nodes]
        if (
            trajectory.shape != (total_nodes,)
            or numpy.any(trajectory[visited_nodes:] != -1)
            or numpy.any(route < 0)
            or numpy.any(route >= total_nodes)
            or numpy.unique(route).size != visited_nodes
        ):
            raise ValueError("Jumanji TSP trajectory is invalid")
        position = _integer(fields, "position")
        if (visited_nodes == 0 and position != -1) or (
            visited_nodes > 0 and int(route[-1]) != position
        ):
            raise ValueError("Jumanji TSP position is invalid")
        coordinates = _array(fields, "coordinates").astype(
            numpy.float64, copy=False
        )
        path_length = float(
            numpy.linalg.norm(
                coordinates[route[1:]] - coordinates[route[:-1]],
                axis=1,
            ).sum()
        )
        closing_edge_length = (
            float(numpy.linalg.norm(coordinates[route[-1]] - coordinates[route[0]]))
            if visited_nodes > 1
            else 0.0
        )
        tour_complete = visited_nodes == total_nodes
        return {
            "current_node": position,
            "start_node": int(route[0]) if visited_nodes else -1,
            "visited_nodes": visited_nodes,
            "unvisited_nodes": int(numpy.count_nonzero(mask)),
            "total_nodes": total_nodes,
            "path_length": path_length,
            "closing_edge_length": closing_edge_length,
            "tour_length_if_closed_now": path_length + closing_edge_length,
            "rewarded_distance": path_length
            + (closing_edge_length if tour_complete else 0.0),
            "tour_complete": tour_complete,
            "route": [int(node) for node in route],
        }
    if profile == "pacman":
        pellets = _array(fields, "pellet_locations")
        power_ups = _array(fields, "power_up_locations")
        ghosts = _array(fields, "ghost_locations")
        if pellets.shape != (318, 2) or power_ups.shape != (4, 2) or ghosts.shape != (4, 2):
            raise ValueError("Jumanji PacMan coordinate arrays are invalid")
        remaining_pellet_mask = numpy.any(pellets != 0, axis=-1)
        remaining_power_up_mask = numpy.any(power_ups != 0, axis=-1)
        player = [
            _integer(fields, "player_locations.x"),
            _integer(fields, "player_locations.y"),
        ]
        ghost_positions = [
            [int(row), int(column)] for column, row in ghosts
        ]
        active_power_ups = [
            [int(row), int(column)]
            for column, row in power_ups[remaining_power_up_mask]
        ]
        frightened_raw = _integer(fields, "frightened_state_time")
        remaining_pellets = int(numpy.count_nonzero(remaining_pellet_mask))
        remaining_power_ups = int(numpy.count_nonzero(remaining_power_up_mask))
        return {
            "score": _integer(fields, "score"),
            "player_position_row_column": player,
            "ghost_positions_row_column": ghost_positions,
            "remaining_pellets": remaining_pellets,
            "pellets_collected": 318 - remaining_pellets,
            "remaining_power_ups": remaining_power_ups,
            "power_ups_collected": 4 - remaining_power_ups,
            "active_power_up_positions_row_column": active_power_ups,
            "frightened_timer_raw": frightened_raw,
            "frightened_steps_remaining": max(0, frightened_raw),
            "ghosts_edible": frightened_raw > 0,
        }
    raise ValueError("Jumanji semantic profile is invalid")


def _cvrp_route(
    fields: dict[str, NDArray[numpy.generic]],
    *,
    position: int,
    unvisited: NDArray[numpy.bool_],
) -> NDArray[numpy.int64]:
    trajectory = _array(fields, "trajectory").astype(numpy.int64, copy=False)
    visited_customers = int(numpy.count_nonzero(~unvisited[1:]))
    if visited_customers == 0:
        return numpy.asarray([0], dtype=numpy.int64)
    customer_positions = numpy.flatnonzero(trajectory != 0)
    if customer_positions.size != visited_customers:
        raise ValueError("Jumanji CVRP trajectory does not match visited customers")
    end = int(customer_positions[-1]) + 1
    if position == 0:
        route = (
            trajectory[: end + 1]
            if end < trajectory.size
            else numpy.concatenate((trajectory, numpy.asarray([0], dtype=numpy.int64)))
        )
    else:
        route = trajectory[:end]
    if route.size == 0 or route[0] != 0 or int(route[-1]) != position:
        raise ValueError("Jumanji CVRP trajectory is invalid")
    return route


def _single_grid_position(
    mask: NDArray[numpy.bool_],
    *,
    name: str,
) -> list[int]:
    positions = numpy.argwhere(mask)
    if positions.shape != (1, 2):
        raise ValueError(f"Jumanji grid must contain exactly one {name}")
    return [int(positions[0, 0]), int(positions[0, 1])]


def _sudoku_board_is_solved(board: NDArray[numpy.generic]) -> bool:
    symbols = numpy.arange(9, dtype=board.dtype)
    rows_valid = all(numpy.array_equal(numpy.sort(row), symbols) for row in board)
    columns_valid = all(
        numpy.array_equal(numpy.sort(column), symbols) for column in board.T
    )
    boxes_valid = all(
        numpy.array_equal(
            numpy.sort(board[row : row + 3, column : column + 3].reshape(-1)),
            symbols,
        )
        for row in range(0, 9, 3)
        for column in range(0, 9, 3)
    )
    return rows_valid and columns_valid and boxes_valid


def _array(
    fields: dict[str, NDArray[numpy.generic]],
    name: str,
) -> NDArray[numpy.generic]:
    value = fields.get(name)
    if value is None:
        raise ValueError(f"Jumanji trace observation omitted {name!r}")
    return value


def _integer(fields: dict[str, NDArray[numpy.generic]], name: str) -> int:
    value = _array(fields, name)
    if value.shape != () or not numpy.issubdtype(value.dtype, numpy.integer):
        raise ValueError(f"Jumanji trace observation field {name!r} is not an integer")
    return int(value.item())


def _number(fields: dict[str, NDArray[numpy.generic]], name: str) -> float:
    value = _array(fields, name)
    if value.shape != () or not numpy.issubdtype(value.dtype, numpy.number):
        raise ValueError(f"Jumanji trace observation field {name!r} is not numeric")
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError(f"Jumanji trace observation field {name!r} is non-finite")
    return result


def _json(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["JumanjiBenchmark"]
