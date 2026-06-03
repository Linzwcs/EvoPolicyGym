"""Task document rendering for Gymnasium-backed environments."""

from __future__ import annotations

from .spec import GymSpec


def render(spec: GymSpec) -> str:
    """Render agent-facing task text for one Gymnasium spec."""

    doc = spec.doc
    if doc is None:
        title = spec.id
        objective = "Maximize total Gymnasium reward over each episode."
        observation = "Use the observation schema from `obs_space` and the details in Gymnasium documentation."
        action = "Return an action matching the declared `action_space` schema."
        reward = "Episode return is the sum of Gymnasium rewards."
        notes: tuple[str, ...] = ()
    else:
        title = doc.title
        objective = doc.objective
        observation = doc.observation
        action = doc.action
        reward = doc.reward
        notes = doc.notes

    lines = [
        f"# {title}",
        "",
        f"EvoPolicyGym environment: `{spec.name}`.",
        f"Upstream Gymnasium id: `{spec.id}`.",
        f"Maximum episode steps: `{spec.steps}`.",
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Policy Interface",
        "",
        "Implement `system/policy.py` with `class Policy`. The constructor receives ",
        "`obs_space`, `action_space`, and `env_meta` as dictionaries. `reset(episode_index)` ",
        "is called before each episode, and `act(obs)` must return one action compatible ",
        "with `action_space`.",
        "",
        "## Observation",
        "",
        observation,
        "",
        "The exact machine-readable schema is available as `obs_space` and in `/info.env_meta.obs_space`.",
        "",
        "## Action",
        "",
        action,
        "",
        "Return JSON-safe Python values matching `action_space`. Invalid numeric actions may be clipped ",
        "or repaired by the harness and marked in transition `info`.",
        "",
        "## Reward",
        "",
        reward,
        "",
        "Feedback reports train-case returns for submitted environment instances. Validation and held-out ",
        "cases remain hidden.",
        "",
        "## Feedback",
        "",
        "Read `feedback/submit_NNN/summary.json` for returns and episode lengths. Inspect ",
        "`feedback/submit_NNN/episodes/ep_XXX/trajectory.jsonl` for step-level observations, ",
        "actions, rewards, termination flags, and `info` diagnostics.",
    ]
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines).replace(" \n", "\n").rstrip() + "\n"
