---
name: evopolicygym
description: Operate and extend EvoPolicyGym through its public SDK. Use for Host/operator setup, Policy Program evaluation, Coding Agent Run configuration, CodingAgent provider integration, Benchmark distribution authoring, retained Run diagnostics, or Kernel development. Do not use as a Benchmark-specific Policy-strategy Skill inside an active Run.
---

# EvoPolicyGym

Use this Skill for caller-side EvoPolicyGym work. Keep Host orchestration,
Benchmark behavior, the Coding Agent workspace, and Episode-local Policy
execution as separate authority boundaries.

## Route the task

Load only the references required by the current task:

- **Install or select packages:** read
  [references/setup.md](references/setup.md).
- **Build or directly evaluate a Policy Program:** read
  [references/evaluation.md](references/evaluation.md).
- **Configure and execute a Coding Agent Run:** read
  [references/run.md](references/run.md).
- **Integrate another Coding Agent provider:** read
  [references/providers.md](references/providers.md).
- **Author or modify a Benchmark distribution:** read
  [references/authoring.md](references/authoring.md) completely before editing
  it.
- **Diagnose a retained Run:** read
  [references/diagnostics.md](references/diagnostics.md) and inspect the records
  without repairing them.
- **Change the Kernel:** read the repository `AGENTS.md` and `ARCHITECTURE.md`
  before editing, then follow their ownership and verification rules.

When a task spans workflows, perform setup or authoring first, direct
Evaluation second, and a Coding Agent Run last.

## Preserve the command and Skill boundaries

- Treat `evopolicygym` as the Host/operator command presentation and use the
  public Python SDK for Evaluation, Run, and authoring workflows.
- Treat `evopolicygym-session` as an Agent-only client projected into one
  active Run. The Host-generated task is authoritative for its current
  submit, finish, workspace, and budget instructions.
- Do not invoke Session commands as an external operator or duplicate their
  syntax in a provider integration.
- Pass Benchmark-specific Skills explicitly as immutable
  `AgentSkill.from_directory()` inputs to `run()`. They are independent Run
  inputs, not Benchmark properties, and are never exposed to the Policy.
- Do not use this general Skill as a substitute for a task-specific
  Benchmark-strategy Skill inside a Run.

## Preserve public and security boundaries

- Confirm Python 3.12, the installed EvoPolicyGym version, and the selected
  Environment distribution before constructing imports or commands.
- Treat every Environment as an independently installable distribution. Read
  its package documentation instead of guessing imports, factories, extras, or
  task configuration.
- Use only the supported public SDK. External distributions may import
  `evopolicygym.authoring`; they must not import private `_...` modules.
- Keep Policy-visible values within the bounded `PolicyValue` ABI. Never cross
  the Policy boundary with Host paths, descriptors, credentials, private
  Episode seeds, Case identity, scorer objects, or custom Python objects.
- Treat `ProcessExecution.unsafe()` as an acknowledgement, not isolation. The
  Agent and Policy execute with the current operating-system user's authority.
- Never silently replace a requested virtualization profile with local process
  execution.

Follow the selected reference's workflow-specific verification and reporting
checklist. Report only fields relevant to the operation performed.
