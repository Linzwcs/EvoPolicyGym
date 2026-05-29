"""Run a single episode against a Policy.

Pure function — given a policy and an env instance's real seed, produces
a trajectory + return + length. Knows nothing about feedback writing,
sandboxes, or budget — those are handled by SubmitHandler.

Per SPEC.md §4.2, each trajectory entry is:
    {t, obs, action, reward, terminated, truncated, info, [reward_components]}

The recorded `action` is what the policy returned **before any env-side
clipping** (per SPEC.md §4.2 "Pre-clip vs. post-clip"). If the env clips
out-of-bounds actions, the unclipped value is what we record; envs MAY
expose the post-clip value in info["action_clipped"].
"""

from __future__ import annotations

import traceback as _traceback
from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeRecord:
    """Result of running one episode.

    Attributes:
        trajectory: list of per-step dicts, each matching the schema in
            SPEC.md §4.2. `obs` may be `None` for external obs storage.
        return_: undiscounted sum of step rewards.
        length: number of steps actually run (== len(trajectory)).
        terminated: True if the final step ended via env-natural termination.
        truncated: True if the final step ended via time-limit truncation.
        ended_with_error: True if a runtime exception or act() timeout cut
            the episode short. EnvRunner does not raise on these — it returns
            a partial record. Caller is responsible for recording the error
            in `episodes/ep_<XXX>/error.txt`.
        error_category: when ended_with_error, one of SPEC.md §4.4 per-episode
            categories: "act_error", "act_timeout", "reset_error".
            `None` on success.
        error_step_index: 0-based step at which the failure occurred (the step
            that was attempted but not completed). `None` on success.
        error_traceback: Python traceback string for act_error / reset_error.
            `None` for act_timeout (no useful frame — execution was
            interrupted by SIGALRM at an arbitrary point) and on success.
        stdout_captured: text written to sys.stdout during this episode
            (and during Policy.__init__ for the first episode of a submit).
            Always set (may be empty). SubmitHandler writes this to
            ep_<XXX>/stdout.txt per SPEC §4.5.
        stderr_captured: same for sys.stderr.
    """

    trajectory: list[dict[str, Any]]
    return_: float
    length: int
    terminated: bool
    truncated: bool
    ended_with_error: bool = False
    error_category: str | None = None
    error_step_index: int | None = None
    error_traceback: str | None = None
    stdout_captured: str = ""
    stderr_captured: str = ""
    #: Per-step observations as a list of numpy arrays, populated when
    #: ``run_episode`` is called with ``record_obs=False`` (i.e. external
    #: obs storage). ``None`` when ``record_obs=True`` (inline mode) —
    #: the obs are already in ``trajectory[i]["obs"]`` instead.
    #: Length equals ``length`` on success; truncated to step count on
    #: mid-episode failure. SubmitHandler writes this to
    #: ``ep_<XXX>/observations.npy`` (SPEC §4.6) when present.
    observations: list[Any] | None = None


def _action_to_jsonable(action: Any, action_space_type: str) -> Any:
    """Serialize one action for trajectory.jsonl per SPEC.md §4.2."""
    if action_space_type == "Discrete":
        return int(action)
    # Box / MultiDiscrete / MultiBinary all serialize as flat lists.
    if hasattr(action, "tolist"):
        return action.tolist()
    if isinstance(action, (int, float)):
        return action
    if isinstance(action, (list, tuple)):
        return list(action)
    # Dict / Tuple action spaces: out of scope for v0; raise so we notice.
    raise NotImplementedError(
        f"action serialization for action_space.type={action_space_type!r} not implemented; "
        f"got action of type {type(action).__name__}"
    )


def _obs_to_jsonable(obs: Any) -> Any:
    """Serialize one observation for trajectory.jsonl (inline mode only).

    External obs storage (env_meta.obs_storage == "external") is handled by
    the caller: pass `record_obs=False` and the trajectory's `obs` fields
    will be `None`; observations should be written separately to
    observations.npy.
    """
    if hasattr(obs, "tolist"):
        return obs.tolist()
    if isinstance(obs, (int, float, list, tuple)):
        return list(obs) if isinstance(obs, tuple) else obs
    raise NotImplementedError(
        f"obs serialization for type {type(obs).__name__} not implemented"
    )


def _info_to_jsonable(info: dict[str, Any]) -> dict[str, Any]:
    """Convert info dict entries to JSON-friendly types.

    NaN/Inf encoding (SPEC §4.2) is the feedback writer's job — env_runner
    just produces the raw values. Numpy scalars become Python scalars here
    so json.dumps can handle them.
    """
    out: dict[str, Any] = {}
    for k, v in info.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def run_episode(
    policy: Any,
    env: Any,
    *,
    real_seed: int,
    episode_index: int,
    action_space_type: str,
    max_steps: int,
    record_obs: bool = True,
    reward_components: dict[str, str] | None = None,
    act_timeout_exc_class: type[BaseException] | None = None,
) -> EpisodeRecord:
    """Run one reset-to-termination cycle.

    Args:
        policy: object satisfying the Policy interface (see SPEC.md §2):
            `.reset(episode_index: int) -> None`
            `.act(obs) -> action`
        env: gymnasium.Env (or compatible). Must support reset(seed=...) and
            step(action) returning (obs, reward, terminated, truncated, info).
        real_seed: passed to env.reset(seed=...). Determines initial state.
        episode_index: passed to policy.reset(). 0-based local index within
            the current submit (per architecture.md §11).
        action_space_type: one of "Discrete", "Box", "MultiDiscrete",
            "MultiBinary"; used to format `action` in the trajectory.
        max_steps: hard step cap (env_meta.max_episode_steps). Episodes
            ending naturally before this cap have terminated=True or
            truncated=True from the env.
        record_obs: if False, every trajectory entry's `obs` is None
            (used for env_meta.obs_storage == "external").
        reward_components: if non-empty, each step extracts info[<info_key>]
            for each declared component and attaches it as
            trajectory[t]["reward_components"]. See SPEC.md §4.2.
        act_timeout_exc_class: if set, exceptions of this type raised inside
            `policy.act()` are classified as `act_timeout` (no traceback —
            SIGALRM frames are not meaningful); all other exceptions become
            `act_error`. The sandbox passes its internal _ActTimeout class.

    Returns:
        EpisodeRecord. EnvRunner never raises on policy / env exceptions —
        it returns ended_with_error=True so the caller can record an
        error.txt entry. (Programming bugs in env_runner itself still raise.)
    """
    trajectory: list[dict[str, Any]] = []
    observations: list[Any] | None = None if record_obs else []
    total_reward = 0.0
    terminated = False
    truncated = False
    ended_with_error = False
    error_category: str | None = None
    error_step_index: int | None = None
    error_traceback: str | None = None
    t = 0

    # Reset.
    try:
        reset_out = env.reset(seed=real_seed)
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:  # pragma: no cover (defensive — older gym API)
            obs, info = reset_out, {}
        if hasattr(policy, "reset"):
            policy.reset(episode_index)
    except Exception:
        return EpisodeRecord(
            trajectory=[],
            return_=0.0,
            length=0,
            terminated=False,
            truncated=False,
            ended_with_error=True,
            error_category="reset_error",
            error_step_index=0,
            error_traceback=_traceback.format_exc(),
            observations=observations,
        )

    while t < max_steps:
        try:
            action = policy.act(obs)
        except Exception as e:
            ended_with_error = True
            if act_timeout_exc_class is not None and isinstance(e, act_timeout_exc_class):
                error_category = "act_timeout"
                error_traceback = None
            else:
                error_category = "act_error"
                error_traceback = _traceback.format_exc()
            error_step_index = t
            break

        try:
            step_out = env.step(action)
        except Exception:
            # env.step raising is rare; classify as act_error since the
            # action the policy returned was unusable. SPEC has no
            # separate env_error category in MVP.
            ended_with_error = True
            error_category = "act_error"
            error_step_index = t
            error_traceback = _traceback.format_exc()
            break

        # Gymnasium 0.26+ returns (obs, reward, terminated, truncated, info).
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
        else:  # pragma: no cover
            next_obs, reward, done, info = step_out  # legacy
            terminated, truncated = done, False

        entry: dict[str, Any] = {
            "t": t,
            "obs": _obs_to_jsonable(obs) if record_obs else None,
            "action": _action_to_jsonable(action, action_space_type),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _info_to_jsonable(info),
        }
        if reward_components:
            entry["reward_components"] = {
                name: float(info[key]) for name, key in reward_components.items() if key in info
            }
        trajectory.append(entry)
        # Accumulate obs for external storage (SPEC §4.6). The obs we
        # just consumed (input to act()) is what gets written to
        # observations.npy[t]. Convert to numpy for consistent typing
        # downstream; defensive copy so subsequent env mutations don't
        # alias.
        if observations is not None:
            import numpy as _np
            observations.append(_np.asarray(obs).copy())

        total_reward += float(reward)
        obs = next_obs
        t += 1

        if terminated or truncated:
            break

    return EpisodeRecord(
        trajectory=trajectory,
        return_=total_reward,
        length=len(trajectory),
        terminated=bool(terminated),
        truncated=bool(truncated),
        ended_with_error=ended_with_error,
        error_category=error_category,
        error_step_index=error_step_index,
        error_traceback=error_traceback,
        observations=observations,
    )
