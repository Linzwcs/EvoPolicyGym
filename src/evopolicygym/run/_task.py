"""Host-owned task template shared by all Coding Agent providers."""

from __future__ import annotations

import json

from ..agents import AgentTask
from ..benchmark import BenchmarkSpec
from . import RunConfig
from ._json import encode_public_json_value


def build_agent_task(spec: BenchmarkSpec, config: RunConfig) -> AgentTask:
    """Build the provider-independent instructions for one development Run."""

    public_spec = {
        "id": spec.id,
        "description": spec.description,
        "observation_space": encode_public_json_value(spec.observation_space),
        "action_space": encode_public_json_value(spec.action_space),
        "metadata": encode_public_json_value(spec.metadata),
        "max_episode_steps": spec.max_episode_steps,
        "primary_metric": spec.primary_metric,
        "score_direction": spec.score_direction,
    }
    rendered_spec = json.dumps(
        public_spec,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return AgentTask(
        instructions=f"""\
You are improving one Policy Program for an EvoPolicyGym Benchmark.

Your working directory is the workspace root. The only submitted Program is:

    program/

Edit files only inside program/. The required entrypoint is
program/policy.py:make_policy. A Policy exposes act(observation); it does not
learn inside an Episode. Persistent improvement happens only by editing and
submitting a new Program between evaluations.

The Host publishes authorized evaluation data under:

    feedback/

Do not modify feedback/. Evaluate the current Program with:

    evopolicygym submit program --episodes N

Choose a positive N no greater than {config.max_episodes_per_submission}. The
whole Run has {config.episode_budget} Episode units and at most
{config.max_submissions} submissions. Read feedback/latest.json and the
referenced Feedback and Artifact files after every successful submission.
The Feedback document's content field and all Artifact contents are defined by
the Benchmark. Inspect their structure, names, media types, and contents to
understand the available development evidence.

Iterate by inspecting the Program, editing it, submitting it, and using the
published Feedback. When you have selected the best published submission, end
the Run with:

    evopolicygym finish SUBMISSION_ID

finish accepts only a published submission. Unsubmitted workspace edits are
never the final Program. Do not exit before calling finish successfully.

Benchmark public specification:

{rendered_spec}
"""
    )


__all__: list[str] = []
