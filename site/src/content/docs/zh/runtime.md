---
locale: zh
page: runtime
section: core
title: "执行与安全"
navTitle: "执行与安全"
description: "ProcessExecution 生命周期保证、故障归属与隔离限制。"
lead: "全新 Episode 生命周期分离已经实现；恶意代码 containment 尚未实现。"
index: D5
order: 5
docsVersion: v0.3
status: draft
---

## 显式 unsafe selection

本地评估需要显式确认：

```python
from evopolicygym.execution import ProcessExecution

execution = ProcessExecution.unsafe()
```

它选择本地进程执行，并不会启用沙箱、移除权限或让不可信代码变得安全。

## 生命周期保证

对于每个 Episode，evaluator 都会：

1. 创建全新的 Benchmark-owned Environment；
2. 将不可变 Program 写入全新 scratch；
3. 启动全新的 Python Policy 进程；
4. 调用一次 `make_policy(context)`；
5. 在该 Episode 的多次 `act()` 调用之间保留同一 instance；
6. 校验并应用完整、未修改的 Actions；
7. 在所有退出路径上关闭 Environment 并回收 Policy 进程。

Policy 状态与 scratch 不会被有意带入其他 Episode。

## 隔离限制

`ProcessExecution` 不提供：

- kernel、namespace、seccomp、cgroup、container 或 microVM isolation；
- CPU、内存、PID、descriptor、磁盘或网络 confinement；
- 对 Host 文件、凭据、进程或系统时间的保护；
- 第三方 Agent 或 Policy 代码的对抗性执行；
- 任意 Python Program 的 bit-for-bit 确定性。

Agent 进程同样在本地运行且没有隔离。当前 Codex integration 具有当前操作系统
用户的权限。

## 故障域

| 故障 | 结果 |
| --- | --- |
| Policy exception、timeout、非法 Action 或 protocol error | 经过净化的 `policy_failed` Episode；Evaluation 可以继续执行计划。 |
| Environment reset、step、评分或 cleanup fault | 中止 Evaluation；绝不成为 Policy penalty。 |
| 进程准备、控制、framing 或 cleanup fault | 作为 execution fault 中止 Evaluation。 |
| Run publication 或 projection fault | 关闭 Run；绝不伪造 committed Feedback。 |

完整 malformed guest frame、partial frame 与 trusted-input error 在内部保持不同
分类，即使公开 Policy feedback 仍经过净化。

## 可选 native 产品

独立构建的 `native/bootstrap` distribution 包含 formal-manager ownership
primitives。仅仅存在该 package 不会使 formal profile 可用。

可独立安装的 `policy-backend-firecracker` package 是 alpha 基础设施。
安装或构建它不会提供沙箱、production backend 或经过资格认证的 formal
execution profile。

请求的 virtualization setting 绝不能静默 fallback 到 `ProcessExecution`。

## 操作建议

仅在满足以下条件时使用本地进程执行：

- Agent、Program 与 Benchmark 均可信；
- caller 接受 Host-level authority；
- 保留的 Run 文件不包含凭据；
- Evaluation 用于开发或 conformance，而不是对抗性准入。

## 下一步

- [Evaluation 与 Runs →](../evaluation/)
- [Benchmark 编写 →](../authoring/)
- [架构说明 →](../../research/)
