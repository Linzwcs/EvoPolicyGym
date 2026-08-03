"""Host-owned task template shared by all Coding Agent providers."""

from __future__ import annotations

import json

from .._protocol.session import SESSION_MAX_EPISODE_INDICES
from ..agents import AgentTask
from ..benchmark import BenchmarkSpec
from ..skills import AgentSkill
from . import RunConfig
from ._json import encode_public_json_value


def build_agent_task(
    spec: BenchmarkSpec,
    config: RunConfig,
    skills: tuple[AgentSkill, ...] = (),
) -> AgentTask:
    """Build the provider-independent instructions for one development Run."""

    submission_limit = config.max_episodes_per_submission
    pool_size = config.episode_pool_size
    assert pool_size is not None
    effective_submission_limit = min(
        SESSION_MAX_EPISODE_INDICES,
        pool_size,
        (
            SESSION_MAX_EPISODE_INDICES
            if submission_limit is None
            else submission_limit
        ),
    )
    if submission_limit is None:
        episode_guidance = (
            "Choose any non-empty, strictly increasing set of indices with no "
            f"more than {effective_submission_limit} entries and no more than "
            "the remaining Episode budget. You decide how to allocate it. "
            "The whole Run has "
            f"{config.episode_budget} Episode units and at most "
            f"{config.max_submissions} submissions."
        )
    else:
        episode_guidance = (
            "Choose any non-empty, strictly increasing set of indices with no "
            f"more than {effective_submission_limit} entries and no more than "
            "the remaining Episode budget. You decide how to allocate it. "
            f"The whole Run has {config.episode_budget} Episode units and at most "
            f"{config.max_submissions} submissions."
        )
    validation = config.validation
    if validation is None:
        finish_guidance = """\
When you have selected the best published submission, end the Run with:

    evopolicygym-session finish SUBMISSION_ID

finish accepts exactly one published submission. A successful finish closes
your authority; the Host selects that sole candidate only after your process
has exited.
"""
    else:
        finish_guidance = f"""\
When you are ready to end search, pass an ordered set of one to
{validation.max_candidates} published candidates:

    evopolicygym-session finish SUBMISSION_ID [SUBMISSION_ID ...]

A successful finish closes your authority. Only after your process has exited,
the Host evaluates every candidate on identical private Validation Episodes and
selects the final Program by {spec.primary_metric} ({spec.score_direction}),
then fewer Policy failures, then your argument order. Validation results are
not returned to this Agent Session. Exceeding the candidate limit, repeating a
candidate, or naming an unpublished submission rejects the whole finish
request, so you may correct it and retry.
"""
    assessment_guidance = ""
    if config.assessment is not None:
        assessment_guidance = """\
After final selection, the Host evaluates only the selected Program on
held-out Assessment Episodes. Assessment never changes the selected candidate,
and its results are not returned to this Agent Session.

"""
    public_spec = {
        "id": spec.id,
        "description": spec.description,
        "observation_space": encode_public_json_value(spec.observation_space),
        "action_space": encode_public_json_value(spec.action_space),
        "metadata": encode_public_json_value(spec.metadata),
        "environment_parameters": encode_public_json_value(
            spec.environment_parameters
        ),
        "environment_digest": spec.environment_digest,
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
    skill_guidance = _skill_guidance(skills)
    return AgentTask(
        instructions=f"""\
You are improving one Policy Program for an EvoPolicyGym Benchmark.

{skill_guidance}\
Network access is forbidden throughout this Run. Do not use web search,
browsers, curl, wget, remote APIs, package registries, Git remotes, or any
other network-capable mechanism. Do not retrieve or consult external game
descriptions, source code, solutions, Action traces, replays, datasets, or
prior results. Derive every improvement only from the files made available in
this workspace and Host-published evaluation Feedback. Installed local
libraries may be used only as Program dependencies or as static tools over
authorized workspace files; they are not an authorized Benchmark interface.

Your working directory is the workspace root. The only submitted Program is:

    program/

Edit Policy source only inside program/. The required entrypoint is
program/policy.py:make_policy. A Policy exposes act(observation); it does not
learn inside an Episode. Persistent improvement happens only by editing and
submitting a new Program between evaluations.

You may write derived diagnostics, selected frames, summaries, and temporary
analysis scripts under:

    analysis/

analysis/ is Agent-owned and is never part of a submitted Program. The Host
does not automatically delete its contents.

The Host publishes authorized evaluation data under:

    feedback/

Do not modify feedback/. The only authorized way to execute, query, or
otherwise interact with the Benchmark Environment is the Session command
below. Do not directly instantiate, import, call, clone, emulate, simulate,
step, or probe the Benchmark Environment through installed libraries,
Benchmark implementations, environment providers, source files, ROMs, data,
assets, executables, or any other local mechanism. Do not inspect Benchmark
implementation files or Host-private Run data outside this workspace. Every
Environment interaction must consume Host-managed Episode budget through:

    evopolicygym-session submit program --episodes "{_selector_example(pool_size)}"

Session commands are synchronous and can remain silent while Episodes run.
Wait for each `evopolicygym-session` command to return before reading its
Feedback or issuing any other Session command. Never start a second submit or
finish concurrently, inspect the control socket, or terminate a Session command
because feedback/latest.json has not appeared yet; that absence is expected
while evaluation is still in progress.

The available Run-local training Episode indices are 0 through
{pool_size - 1}. A singleton like "7" selects one index; START:END is a
half-open range, and comma-separated items form a union. Repeated or overlapping
indices in one submission are rejected. You may deliberately reuse an index in
a later submission; that index has the same hidden Episode specification and
Policy seed, while the Host still creates a fresh Environment and fresh Policy
runtime. Every selected index consumes one Episode budget unit on every use.

{episode_guidance} Small selections support fast iteration; larger selections
provide more evidence. Read feedback/latest.json and the referenced Feedback
and Artifact files after every successful submission. The Feedback document
maps each public Episode result back to its selected Run-local index. Its
content field and all Artifact contents are defined by the Benchmark. Inspect
their structure, names, media types, and contents to understand the available
development evidence.

Large observation and trajectory Artifacts from older submissions may be
evicted after a newer submission is published. Compact Feedback remains, and
the newest submission is always protected. Before starting another submission,
copy any derived evidence you need to retain into analysis/; never copy it into
feedback/.

Iterate by inspecting the Program, editing it, submitting it, and using the
published Feedback.

{finish_guidance}
{assessment_guidance}\
Unsubmitted workspace edits are never candidates or the final Program. Do not
exit before calling finish successfully.

The Environment parameters in the specification are public and fixed for this
entire Run. The Policy receives the same values through
PolicyContext.environment_parameters. Neither you nor the Policy may change
the evaluated Environment configuration.

Benchmark public specification:

{rendered_spec}
"""
    )


def _skill_guidance(skills: tuple[AgentSkill, ...]) -> str:
    if not skills:
        return ""
    paths = "\n".join(
        f"    skills/{skill.name}/SKILL.md" for skill in skills
    )
    return f"""\
This Run explicitly provides the following read-only Agent Skills:

{paths}

Read every listed SKILL.md completely before inspecting the Program or
submitting a candidate. A Skill may reference additional files inside its own
directory. Follow the selected workflows while treating this Host task, the
Benchmark specification, current observations, and legal Actions as
authoritative.

"""


def _selector_example(pool_size: int) -> str:
    if pool_size >= 8:
        return "0:2,4:8"
    if pool_size >= 5:
        return f"0:2,4:{pool_size}"
    if pool_size >= 3:
        return f"0:2,{pool_size - 1}"
    if pool_size == 2:
        return "0:2"
    return "0"


__all__: list[str] = []
