---
locale: zh
page: programs
section: api
title: "Programs"
navTitle: "Programs"
description: "创建并检查不可变的 Policy 源码快照。"
lead: "Program 是一个 Policy 目录的不可变、内容寻址快照。"
index: D3
order: 3
docsVersion: v0.3
status: draft
---

## 目录结构

Program 目录必须包含 `policy.py`：

```text
my-policy/
├── policy.py
└── model.json
```

`policy.py` 必须定义 `make_policy(context)`。其他文件是可选的，Policy 可以导入
或读取它们。

## 创建 Program

```python
from evopolicygym import Program

program = Program.from_directory("my-policy/")

print(program.digest)
print(program.files)
```

`from_directory()` 读取一个稳定快照。之后修改 `my-policy/` 不会改变 `program`。

| 属性 | 值 |
| --- | --- |
| `digest` | SHA-256 内容标识，包含固定入口和 Policy ABI 版本。 |
| `entrypoint` | `policy.py:make_policy` |
| `policy_abi` | 快照要求的 Policy ABI 版本。 |
| `files` | 按确定顺序排列的相对文件路径。 |
| `file_count` | 快照中的文件数。 |
| `total_bytes` | 文件未压缩总大小。 |

## 快照限制

默认上限为 1,000 个文件、总计 64 MiB、单个文件 16 MiB。可使用
`ProgramLimits` 修改：

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

源码必须来自真实目录。符号链接会被拒绝，`.git` 和 `__pycache__` 目录会被忽略。

## 读取或写出快照

```python
source = program.read_bytes("policy.py")
program.write_to("saved-policy")
```

`write_to()` 要求目标目录尚不存在。

## 下一步

- [编写 Policy 入口](./policy.md)
- [评估 Program](./evaluation.md)
- [在 Run 中使用 Program](./runs.md)
