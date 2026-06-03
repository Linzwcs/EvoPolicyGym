# Security Policy

EvoPolicyGym runs agent-authored Python code and may launch external coding-agent
CLIs. Treat every benchmark run as untrusted code execution.

## Supported Versions

The current `main` branch is the only supported development target. Frozen v1
material under `archive/v1/` is kept for historical reference and is not
maintained.

## Reporting Security Issues

If this repository is hosted publicly, report vulnerabilities through the
project's private security advisory channel when available. If no private
channel exists, contact the maintainers directly before publishing exploit
details.

Include:

- affected commit or release;
- reproduction steps;
- expected impact;
- whether agent-authored code, local credentials, subprocess execution, or
  network access is involved.

## Operational Guidance

- Run live agent experiments in isolated workspaces.
- Avoid exposing production credentials to agent sessions.
- Prefer sandboxed rollout execution for untrusted policies.
- Treat `runs/`, `experiment/`, `workspace/`, `feedback/`, `logs/`, and
  checkpoints as potentially sensitive artifacts.
- Review generated `policy.py` and auxiliary files before reusing them outside a
  sandbox.
- Browser and simulator environments may download or execute additional runtime
  assets; pin versions and keep those assets outside source control.

EvoPolicyGym's server controls rollout budget and hidden evaluation, but it does
not make arbitrary agent-authored code safe by itself.
