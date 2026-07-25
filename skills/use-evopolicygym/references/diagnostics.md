# Run diagnostics

Inspect generated Run data read-only. Do not repair a Run by hand and do not
copy Host logs into the Agent workspace.

## Record layout

```text
run-root/
├── run.json
├── events.jsonl
├── initial/program/
├── workspace/
│   ├── program/
│   ├── feedback/
│   └── skill/SKILL.md          optional
├── submissions/
│   └── submission-NNNNNN/
│       ├── program/
│       ├── feedback.json
│       └── artifacts/
├── validation/                 only after successful configured Validation
│   └── report.json             aggregate candidate evidence
└── agent/
    ├── invocation.json
    ├── instructions.md
    ├── stdout.log
    └── stderr.log
```

`events.jsonl` is the persisted lifecycle sequence. `run.json` is the terminal
manifest and authoritative summary. A submission appears under `submissions/`
only after its Program and public Feedback commit successfully.

## Diagnose in order

1. Read `run.json`: benchmark ID, config, terminal reason, final submission,
   ordered candidates, published submissions, optional Validation report, and
   Agent exit.
2. Read the final events in `events.jsonl`: determine whether failure occurred
   before an Episode, during an Episode, after all Episodes, or during
   publication.
3. Inspect only the matching submission's public `feedback.json` and Artifact
   sizes.
4. Inspect `agent/stderr.log` and structured `agent/stdout.log` for provider or
   command failures. Treat these as Host-owned execution records.
5. Reproduce the smallest safe direct Evaluation only when the retained
   records do not identify the fault.

## Interpret common outcomes

- `finished`: the Agent handed off candidates and the Host selected a final
  published submission.
- `budget_exhausted`: no Episode authority remains without a successful
  selection.
- `agent_exited`: the Agent process exited before a terminal selection.
- `agent_failed`: the Agent could not start, timed out, or violated its process
  contract.
- `evaluation_failed`: trusted evaluation could not produce a valid result.
- `validation_failed`: post-Agent candidate Validation failed; no partial
  final selection or report is authoritative.

An Episode status of `policy_failed` is a scored Policy outcome, not a trusted
evaluation failure. Its failure code is one of `exception`, `timeout`,
`invalid_action`, or `protocol_error`.

If all requested Episodes completed and `evaluation_failed` follows
immediately, inspect `Benchmark.feedback()`, score validation, PolicyValue
conversion, and Artifact limits. If `publication_failed` appears, inspect Run
directory permissions, disk availability, serialization, and atomic file
commit behavior.

## Keep evidence boundaries intact

- Do not expose Agent logs, Host paths, seeds, Case identity, or execution
  evidence through Feedback.
- Do not reinterpret Environment or Backend faults as zero reward.
- Do not refund reserved Episode budget after a trusted evaluation failure.
- Do not infer that a missing submission directory means its Episodes did not
  execute; publication happens only after Evaluation and Feedback succeed.
- Remember that current local processes are not isolated even though the
  logical workspace and log views are separate.
