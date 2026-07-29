# EvoPolicyGym setup

Establish the exact package and version context before building commands.

## Select the context

- In the EvoPolicyGym repository, use the active source under
  `src/evopolicygym/` and the project-managed `uv` environment.
- In another project, use the installed public package and its matching
  documentation. Do not apply procedures from a different release.
- Treat every package under `environments/` as an independently installable
  Benchmark distribution. The Kernel wheel does not include them.
- Read the selected distribution's README and `pyproject.toml` before choosing
  imports, extras, task configuration, or build commands.

## Verify the repository environment

From the Kernel repository:

```console
uv sync --extra dev
uv run evopolicygym --version
uv lock --check
```

EvoPolicyGym 0.3 requires Python 3.12. If the installed version differs from
the documentation or selected Benchmark requirement, stop and resolve that
compatibility mismatch before authoring or evaluation.

For an Environment distribution, enter its own project directory and use its
lock and dependency configuration:

```console
uv sync --extra dev
uv lock --check
```

Do not assume a successful Kernel import proves that the selected Environment
distribution or its optional simulator dependencies are installed.

## Confirm public imports

Inspect the distribution's documented public module and verify the exact
Benchmark class, configuration values, and baseline factory before using them.
Do not guess an import name from a distribution name or environment folder.

Report the Python version, EvoPolicyGym version, selected distribution and
version, public import used, commands run, and any unresolved dependency or
version mismatch.
