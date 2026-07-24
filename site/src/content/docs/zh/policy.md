---
locale: zh
page: policy
section: core
title: "Policy ABI"
navTitle: "Policy ABI"
description: "policy/v1 entry point、PolicyContext、PolicyValue carrier、状态生命周期与故障语义。"
lead: "一个固定 factory、一个动作方法，以及提交代码可见的有界值表面。"
index: D3
order: 3
docsVersion: v0.3
status: draft
---

## Program 入口

每个 Program 目录必须包含 `policy.py`。固定 entry point 为
`policy.py:make_policy`：

```python
from evopolicygym.policy import PolicyContext, PolicyValue


class MyPolicy:
    def __init__(self, context: PolicyContext):
        self._seed = context.policy_seed

    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0


def make_policy(context: PolicyContext) -> MyPolicy:
    return MyPolicy(context)
```

每个 Episode 调用一次 `make_policy(context)`。返回对象必须提供
`act(observation)`。

Policy 不可见 `learn()`、`reset()`、`update()`、Submission 或 Feedback 方法。

## PolicyContext

| 字段 | 含义 |
| --- | --- |
| `observation_space` | Policy 可见 Observation 的公开描述。 |
| `action_space` | 合法 Action 的公开描述。 |
| `metadata` | string-keyed、与 Case 无关的 Benchmark metadata。 |
| `policy_seed` | 当前全新 Policy instance 使用的 unsigned 64-bit seed。 |

`PolicyContext` 绝不包含 Case identity、Environment seed、Host path、文件描述符、
凭据、pool identity、分数、runtime evidence 或 scorer object。

## PolicyValue

跨越 Policy 边界的值只能由以下 carrier 组成：

```text
None | bool | int | float | str | bytes
| TensorValue
| list[PolicyValue]
| tuple[PolicyValue, ...]
| dict[str, PolicyValue]
```

规则是严格的：

- float 必须有限；
- integer 必须适配 signed 或 unsigned 64-bit；
- mapping key 必须是 exact string；
- container 会递归校验并与原值脱离；
- 自定义 Python object 与 pickle object graph 会被拒绝。

## TensorValue

`TensorValue` 传递 canonical dense tensor：

```python
from evopolicygym.policy import TensorValue

pixels = TensorValue(
    dtype="uint8",
    shape=(84, 84, 3),
    data=raw_rgb_bytes,
)
```

支持 `bool`、8 到 64 位的 signed/unsigned integers，以及 `float16`、
`float32`、`float64`。多字节值使用 little-endian；浮点值必须有限；byte
长度必须与 shape 和 dtype 完全一致。

## Episode 内状态

```text
Episode 0  new process → make_policy() → act() × N → destroy
Episode 1  new process → make_policy() → act() × N → destroy
Episode N  new process → make_policy() → act() × N → destroy
```

Policy 可以在同一 Episode 的 `act()` 调用之间保留历史、循环状态、搜索树、
cache 或临时参数。任何状态都不能进入下一个 Episode。

跨 Episode 学习属于外层 Coding Agent；它通过编写并提交新的不可变 Program
完成改进。

## Action 语义

Environment 接收 `Policy.act()` 返回的完整、未修改 Action。

```text
Policy.act(observation)
        ↓
PolicyValue validation
        ↓
Environment.step(action)
        ├── valid   → trusted Step
        └── invalid → InvalidAction
                       Policy failure; no fallback step
```

非法 Action 绝不会被裁剪、修复、采样或替换。

## Policy failure

公开 Policy failure code 包括：

| Code | 含义 |
| --- | --- |
| `exception` | `make_policy()` 或 `act()` 抛出异常。 |
| `timeout` | Policy 超过 Episode operation timeout。 |
| `invalid_action` | Environment 拒绝完整 Action。 |
| `protocol_error` | Policy 进程返回 malformed 或不可编码值。 |

Policy failure 发生后，evaluator 不会再次调用 `Environment.step()`。
可信 Environment 与 execution fault 保持分离并中止 Evaluation。

## 下一步

- [Evaluation 与 Runs →](../evaluation/)
- [执行与安全 →](../runtime/)
- [阅读 `policy.py` 源码 ↗](https://github.com/Linzwcs/EvoPolicyGym/blob/main/src/evopolicygym/policy.py)
