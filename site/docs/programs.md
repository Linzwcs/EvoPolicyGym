---
locale: en
page: programs
section: api
title: "Programs"
navTitle: "Programs"
description: "Create and inspect immutable Policy source snapshots."
lead: "A Program is an immutable, content-addressed snapshot of one Policy directory."
index: D3
order: 3
docsVersion: v0.3
status: draft
---

## Directory layout

A Program directory must contain `policy.py`:

```text
my-policy/
├── policy.py
└── model.json
```

`policy.py` must define `make_policy(context)`. Other files are optional and
may be imported or read by the Policy.

## Create a Program

```python
from evopolicygym import Program

program = Program.from_directory("my-policy/")

print(program.digest)
print(program.files)
```

`from_directory()` reads a stable snapshot. Later changes to `my-policy/` do
not change `program`.

| Property | Value |
| --- | --- |
| `digest` | SHA-256 content identity, including the fixed entry point and Policy ABI version. |
| `entrypoint` | `policy.py:make_policy` |
| `policy_abi` | Policy ABI version required by the snapshot. |
| `files` | Relative file paths in deterministic order. |
| `file_count` | Number of files in the snapshot. |
| `total_bytes` | Total uncompressed file size. |

## Snapshot limits

Default limits are 1,000 files, 64 MiB total, and 16 MiB per file. Override
them with `ProgramLimits`:

```python
from evopolicygym import Program
from evopolicygym.program import ProgramLimits

program = Program.from_directory(
    "my-policy/",
    limits=ProgramLimits(
        max_files=100,
        max_total_bytes=8 * 1024 * 1024,
        max_file_bytes=2 * 1024 * 1024,
    ),
)
```

The source must be a real directory. Symbolic links are rejected. `.git` and
`__pycache__` directories are ignored.

## Read or materialize a snapshot

```python
source = program.read_bytes("policy.py")
program.write_to("saved-policy")
```

`write_to()` requires a destination that does not already exist.

## Next

- [Write the Policy entry point](./policy.md)
- [Evaluate the Program](./evaluation.md)
- [Use Programs in a Run](./runs.md)
