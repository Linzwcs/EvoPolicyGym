"""Complete, reversible Agent-visible evidence for NLE training Feedback."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Sequence

import numpy as np
from evopolicygym.authoring import Artifact, EpisodeRecord
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .constants import ACTION_MEANINGS, BLSTAT_NAMES, CONDITION_BITS

MAX_PUBLIC_FEEDBACK_EPISODES = 64
OBSERVATION_CHUNK_SIZE = 1_024
_DUNGEON_SHAPE = (21, 79)
_INVENTORY_SIZE = 55
_INVENTORY_STRING_SIZE = 80
_MESSAGE_SIZE = 256
_INPUT_MODE_CODES = {
    "normal": 0,
    "yes_no": 1,
    "get_line": 2,
    "more": 3,
}


def complete_feedback_artifacts(
    records: Sequence[EpisodeRecord],
) -> tuple[tuple[Artifact, ...], dict[str, PolicyValue]]:
    """Encode every Policy-visible observation and transition without sampling."""

    if len(records) > MAX_PUBLIC_FEEDBACK_EPISODES:
        raise ValueError(
            "public NLE training Feedback supports at most "
            f"{MAX_PUBLIC_FEEDBACK_EPISODES} Episodes"
        )

    artifacts: list[Artifact] = []
    trajectory_entries: list[dict[str, object]] = []
    observation_entries: list[dict[str, object]] = []
    total_observations = 0
    total_transitions = 0
    chunk_index = 0
    chunk: list[_EncodedObservation] = []
    episode_indices: list[int] = []
    observation_indices: list[int] = []

    def flush_observations() -> None:
        nonlocal chunk_index
        if not chunk:
            return
        artifact = _observation_artifact(
            chunk,
            episode_indices=episode_indices,
            observation_indices=observation_indices,
            chunk_index=chunk_index,
        )
        artifacts.append(artifact)
        observation_entries.append(
            {
                "artifact": artifact.name,
                "observations": len(chunk),
                "compressed_bytes": artifact.size,
                "first": {
                    "episode_index": episode_indices[0],
                    "observation_index": observation_indices[0],
                },
                "last": {
                    "episode_index": episode_indices[-1],
                    "observation_index": observation_indices[-1],
                },
            }
        )
        chunk.clear()
        episode_indices.clear()
        observation_indices.clear()
        chunk_index += 1

    for episode_index, record in enumerate(records):
        trajectory = _trajectory_artifact(record, episode_index=episode_index)
        artifacts.append(trajectory)
        trajectory_entries.append(
            {
                "episode_index": episode_index,
                "artifact": trajectory.name,
                "steps": record.steps,
                "compressed_bytes": trajectory.size,
            }
        )
        total_transitions += record.steps
        for observation_index, observation in enumerate(_observations(record)):
            chunk.append(_encode_observation(observation))
            episode_indices.append(episode_index)
            observation_indices.append(observation_index)
            total_observations += 1
            if len(chunk) == OBSERVATION_CHUNK_SIZE:
                flush_observations()
    flush_observations()

    bulk_bytes = sum(
        artifact.size for artifact in artifacts if artifact.retention == "bulk"
    )
    manifest = {
        "schema": "nle/complete-policy-observation-feedback/v1",
        "complete": True,
        "visualization_generated": False,
        "source": "Policy-visible observation values only",
        "alignment": "observation[t] -> action[t] -> observation[t + 1]",
        "encoding": {
            "archive": "NPZ/ZIP_DEFLATED",
            "allow_pickle": False,
            "byte_strings": "Latin-1 with explicit lengths",
            "screen_shape": list(_DUNGEON_SHAPE),
            "stats": list(BLSTAT_NAMES),
            "condition_bits": {
                str(bit): name for bit, name in CONDITION_BITS
            },
            "input_mode_codes": {
                str(code): name for name, code in _INPUT_MODE_CODES.items()
            },
        },
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "trajectory_artifacts": trajectory_entries,
        "observation_artifacts": observation_entries,
        "bulk_compressed_bytes": bulk_bytes,
        "retention": {
            "class": "bulk",
            "policy": (
                "complete for the newest submission; oldest bulk Artifacts "
                "are evicted first from Agent and Host records"
            ),
            "agent_control": (
                "the Agent may inspect, decode, transform, or selectively "
                "retain evidence under analysis/"
            ),
        },
    }
    artifacts.append(
        Artifact(
            name="artifact-manifest.json",
            media_type="application/json",
            content=(
                json.dumps(
                    manifest,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8", errors="strict"),
        )
    )
    return tuple(artifacts), {
        "schema": "nle/complete-policy-observation-feedback-summary/v1",
        "complete": True,
        "visualization_generated": False,
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "observation_chunks": len(observation_entries),
        "trajectory_artifacts": len(trajectory_entries),
        "bulk_compressed_bytes": bulk_bytes,
    }


class _EncodedObservation:
    __slots__ = (
        "chars",
        "colors",
        "glyphs",
        "input_mode",
        "inventory_count",
        "inventory_descriptions",
        "inventory_description_lengths",
        "inventory_glyphs",
        "inventory_letters",
        "inventory_object_classes",
        "message_bytes",
        "message_length",
        "stats",
    )

    def __init__(self) -> None:
        self.glyphs = np.empty(_DUNGEON_SHAPE, dtype=np.int16)
        self.chars = np.empty(_DUNGEON_SHAPE, dtype=np.uint8)
        self.colors = np.empty(_DUNGEON_SHAPE, dtype=np.uint8)
        self.stats = np.empty((len(BLSTAT_NAMES),), dtype=np.int64)
        self.message_bytes = np.zeros((_MESSAGE_SIZE,), dtype=np.uint8)
        self.message_length = 0
        self.inventory_count = 0
        self.inventory_letters = np.zeros((_INVENTORY_SIZE,), dtype=np.uint8)
        self.inventory_descriptions = np.zeros(
            (_INVENTORY_SIZE, _INVENTORY_STRING_SIZE),
            dtype=np.uint8,
        )
        self.inventory_description_lengths = np.zeros(
            (_INVENTORY_SIZE,),
            dtype=np.uint8,
        )
        self.inventory_glyphs = np.zeros((_INVENTORY_SIZE,), dtype=np.int16)
        self.inventory_object_classes = np.zeros(
            (_INVENTORY_SIZE,),
            dtype=np.uint8,
        )
        self.input_mode = 0


def _encode_observation(value: PolicyValue) -> _EncodedObservation:
    if type(value) is not dict or set(value) != {
        "screen",
        "stats",
        "message",
        "inventory",
        "input_mode",
    }:
        raise ValueError("NLE Feedback observation is invalid")
    encoded = _EncodedObservation()
    screen = value["screen"]
    if type(screen) is not dict or set(screen) != {"glyphs", "chars", "colors"}:
        raise ValueError("NLE Feedback screen is invalid")
    encoded.glyphs[:] = _tensor_array(screen["glyphs"], "int16")
    encoded.chars[:] = _tensor_array(screen["chars"], "uint8")
    encoded.colors[:] = _tensor_array(screen["colors"], "uint8")

    stats = value["stats"]
    expected_stat_keys = set(BLSTAT_NAMES) | {"conditions"}
    if type(stats) is not dict or set(stats) != expected_stat_keys:
        raise ValueError("NLE Feedback stats are invalid")
    for index, name in enumerate(BLSTAT_NAMES):
        statistic = stats[name]
        if type(statistic) is not int or not -(2**63) <= statistic < 2**63:
            raise ValueError(f"NLE Feedback stat {name} is invalid")
        encoded.stats[index] = statistic
    condition_mask = int(encoded.stats[BLSTAT_NAMES.index("condition_mask")])
    expected_conditions = [
        name for bit, name in CONDITION_BITS if condition_mask & bit
    ]
    if stats["conditions"] != expected_conditions:
        raise ValueError("NLE Feedback conditions are inconsistent")

    message = _latin1(value["message"], maximum=_MESSAGE_SIZE, name="message")
    encoded.message_bytes[: len(message)] = np.frombuffer(message, dtype=np.uint8)
    encoded.message_length = len(message)

    inventory = value["inventory"]
    if type(inventory) is not list or len(inventory) > _INVENTORY_SIZE:
        raise ValueError("NLE Feedback inventory is invalid")
    encoded.inventory_count = len(inventory)
    for index, item in enumerate(inventory):
        if type(item) is not dict or set(item) != {
            "letter",
            "description",
            "glyph",
            "object_class",
        }:
            raise ValueError("NLE Feedback inventory entry is invalid")
        letter = _latin1(item["letter"], maximum=1, name="inventory letter")
        if len(letter) != 1:
            raise ValueError("NLE Feedback inventory letter is invalid")
        description = _latin1(
            item["description"],
            maximum=_INVENTORY_STRING_SIZE,
            name="inventory description",
        )
        glyph = item["glyph"]
        object_class = item["object_class"]
        if type(glyph) is not int or not -(2**15) <= glyph < 2**15:
            raise ValueError("NLE Feedback inventory glyph is invalid")
        if type(object_class) is not int or not 0 <= object_class < 2**8:
            raise ValueError("NLE Feedback inventory object class is invalid")
        encoded.inventory_letters[index] = letter[0]
        encoded.inventory_descriptions[index, : len(description)] = np.frombuffer(
            description,
            dtype=np.uint8,
        )
        encoded.inventory_description_lengths[index] = len(description)
        encoded.inventory_glyphs[index] = glyph
        encoded.inventory_object_classes[index] = object_class

    input_mode = value["input_mode"]
    if type(input_mode) is not str or input_mode not in _INPUT_MODE_CODES:
        raise ValueError("NLE Feedback input mode is invalid")
    encoded.input_mode = _INPUT_MODE_CODES[input_mode]
    return encoded


def _observation_artifact(
    chunk: Sequence[_EncodedObservation],
    *,
    episode_indices: Sequence[int],
    observation_indices: Sequence[int],
    chunk_index: int,
) -> Artifact:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        episode_indices=np.asarray(episode_indices, dtype=np.uint32),
        observation_indices=np.asarray(observation_indices, dtype=np.uint32),
        glyphs=np.stack([item.glyphs for item in chunk]),
        chars=np.stack([item.chars for item in chunk]),
        colors=np.stack([item.colors for item in chunk]),
        stats=np.stack([item.stats for item in chunk]),
        message_bytes=np.stack([item.message_bytes for item in chunk]),
        message_lengths=np.asarray(
            [item.message_length for item in chunk],
            dtype=np.uint16,
        ),
        inventory_counts=np.asarray(
            [item.inventory_count for item in chunk],
            dtype=np.uint8,
        ),
        inventory_letters=np.stack([item.inventory_letters for item in chunk]),
        inventory_descriptions=np.stack(
            [item.inventory_descriptions for item in chunk]
        ),
        inventory_description_lengths=np.stack(
            [item.inventory_description_lengths for item in chunk]
        ),
        inventory_glyphs=np.stack([item.inventory_glyphs for item in chunk]),
        inventory_object_classes=np.stack(
            [item.inventory_object_classes for item in chunk]
        ),
        input_modes=np.asarray(
            [item.input_mode for item in chunk],
            dtype=np.uint8,
        ),
    )
    return Artifact(
        name=f"bulk/observations-{chunk_index:06d}.npz",
        media_type="application/x-npz",
        content=output.getvalue(),
        retention="bulk",
    )


def _trajectory_artifact(record: EpisodeRecord, *, episode_index: int) -> Artifact:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": record.total_reward,
                    "failure": record.policy_failure,
                    "initial_observation_index": 0,
                    "final_observation_index": record.steps,
                }
            )
        )
        for step_index, transition in enumerate(record.transitions):
            action = transition.action
            if type(action) is not int or not 0 <= action < len(ACTION_MEANINGS):
                raise ValueError("NLE trajectory Action is invalid")
            metrics = _trajectory_metrics(transition.step.metrics)
            stream.write(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation_index": step_index,
                        "next_observation_index": step_index + 1,
                        "action": action,
                        "action_name": ACTION_MEANINGS[action],
                        "reward": transition.step.reward,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": metrics,
                    }
                )
            )
    return Artifact(
        name=(
            f"bulk/episodes/episode-{episode_index:06d}/"
            "trajectory-000000.jsonl.gz"
        ),
        media_type="application/gzip",
        content=output.getvalue(),
        retention="bulk",
    )


def _trajectory_metrics(value: PolicyValue) -> dict[str, int | bool]:
    names = {
        "game_score",
        "max_game_score",
        "depth",
        "max_depth",
        "experience_level",
        "dungeon_level",
        "hit_points",
        "turn",
        "ascended",
        "end_status",
    }
    if type(value) is not dict or set(value) != names:
        raise ValueError("NLE trajectory metrics are invalid")
    result: dict[str, int | bool] = {}
    for name in sorted(names):
        item = value[name]
        if name == "ascended":
            if type(item) is not bool:
                raise ValueError("NLE trajectory ascended metric is invalid")
        elif type(item) is not int:
            raise ValueError(f"NLE trajectory metric {name} is invalid")
        result[name] = item
    return result


def _observations(record: EpisodeRecord) -> Sequence[PolicyValue]:
    return (record.initial_observation,) + tuple(
        transition.step.observation for transition in record.transitions
    )


def _tensor_array(value: PolicyValue, dtype: str) -> NDArray[np.generic]:
    if (
        type(value) is not TensorValue
        or value.dtype != dtype
        or value.shape != _DUNGEON_SHAPE
    ):
        raise ValueError("NLE Feedback screen tensor is invalid")
    selected_dtype = np.dtype(dtype)
    expected_bytes = int(np.prod(_DUNGEON_SHAPE)) * selected_dtype.itemsize
    if len(value.data) != expected_bytes:
        raise ValueError("NLE Feedback screen tensor data is invalid")
    return np.frombuffer(value.data, dtype=selected_dtype).reshape(_DUNGEON_SHAPE)


def _latin1(value: PolicyValue, *, maximum: int, name: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"NLE Feedback {name} is invalid")
    try:
        encoded = value.encode("latin-1", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"NLE Feedback {name} is invalid") from error
    if len(encoded) > maximum:
        raise ValueError(f"NLE Feedback {name} is too large")
    return encoded


def _json_line(document: dict[str, object]) -> bytes:
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


__all__ = [
    "MAX_PUBLIC_FEEDBACK_EPISODES",
    "OBSERVATION_CHUNK_SIZE",
    "complete_feedback_artifacts",
]
