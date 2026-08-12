# Coding Agent provider integration

Integrate another command-line Coding Agent through the public structural
`CodingAgent` contract.

The Kernel includes first-party `Codex`, `ClaudeCode`, `KimiCode`, and `Qoder`
translations. They all retain the unchanged Host task and use the same generic
process runner. `ClaudeCode` selects bare, non-persistent print mode. Kimi Code
and Qoder do not currently expose an equivalent no-session-persistence flag,
so callers that require isolated CLI configuration and session history should
set a dedicated `KIMI_CODE_HOME` or `QODER_CONFIG_DIR` before the Run.

Translate the Host-owned `AgentTask` into an `AgentInvocation`. Do not author
Run instructions, duplicate Agent Session syntax, or move process supervision
into the provider.

```python
from evopolicygym.agents import AgentTask, command_invocation


class ExampleAgent:
    def build_invocation(self, task: AgentTask):
        return command_invocation(
            ("example-agent", "--prompt", task.instructions),
            recorded_command=(
                "example-agent",
                "--prompt",
                "@agent/instructions.md",
            ),
            identity={"provider": "example-agent"},
            instructions=task.instructions,
            inherited_environment=("HOME", "EXAMPLE_API_KEY"),
        )
```

- Keep provider-specific CLI translation and identity in the provider module.
- Pass the exact Host task instructions without rewriting Kernel semantics.
- Inherit only explicitly required environment-variable names.
- Never place credentials in `recorded_command`, identity, instructions,
  Feedback, Artifacts, or retained public results.
- Let the Run-owned process mechanism start, supervise, and reap the Agent.
- Keep provider-specific model and reasoning selections out of `RunConfig` and
  `BenchmarkSpec`.

Verify import safety, deterministic argument translation, recorded-command
sanitization, required identity fields, environment allowlisting, and invalid
configuration failures. Report the provider module, translated invocation,
retained identity, inherited environment names, tests run, and any remaining
provider-specific limitation.
