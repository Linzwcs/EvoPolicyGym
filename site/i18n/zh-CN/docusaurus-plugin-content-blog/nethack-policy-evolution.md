---
locale: zh
page: nethack-policy-evolution
title: "深入地下城：为 NetHack 构建探索系统"
description: "Coding Agent 如何将完整的 NetHack 轨迹转化为能够导航、处理障碍并向地下城深处推进的可执行 Policy。"
lead: "Coding Agent 利用 Policy 可见的 NetHack 轨迹构建空间记忆、从移动失败中恢复，并在地下城中取得可测量的进展。"
publishedAt: "2026-08-03"
date: "2026-08-03"
authors: [evopolicygym]
tags:
  - Benchmark
  - NetHack
  - Experiment
  - Policy Evolution
status: published
---

## NetHack 是什么？

NetHack 是一款回合制 Roguelike 游戏，舞台是一座程序生成的地下城。完整游戏要求
玩家深入地下城，取得 Amulet of Yendor，返回地面并完成 ascension。要实现这一
目标，远不只是赢下几场战斗：玩家需要探索未知布局、理解消息、管理资源、记住
重要位置，并在永久死亡的威胁下生存。

<!-- truncate -->

一个简化的推进循环如下：

```text
探索当前地下城楼层
        ↓
处理生物、障碍与资源
        ↓
寻找向下的楼梯
        ↓
下楼并重复这一过程
```

每个 Episode 的具体情况都会变化。房间与走廊重新排列，物品和生物出现在不同
位置，而 Policy 每次只能看到当前楼层的一部分。一个局部看来合理的 Action，
可能浪费数百回合、消耗稀缺食物，或者让角色偏离原本要前往的路线。

这些特点使 NetHack 成为研究可执行策略的合适 Environment。Policy 必须将即时
反应、记忆和长期目标结合起来，同时能够判断游戏是否按预期响应了刚才的 Action。

## 将 NetHack 接入 EvoPolicyGym

该 Benchmark 接入了 NLE 1.3.0 的 `NetHackScore-v0`，底层使用 NetHack 3.6.7。
Policy 接收终端地图的语义表示，以及状态值、当前消息、公开背包条目和输入模式。
它可以从 23 个 Actions 中选择，包括移动、奔跑、上下楼梯、等待、踢、进食、
搜索和处理消息提示。

这套 Action profile 有意比完整 NetHack 的命令集合更窄。在 5,000 步的 Episode
限制内，实验主要考察游戏前期的探索、障碍处理、生存与下楼推进，而不是完整
通关。

分数奖励 NetHack 认可的游戏进展，同时惩罚角色持续冻结在原地的重复步骤。它为
Agent 提供主要优化信号，而地下城深度、游戏分数和冻结步比例则用于解释这些分数
对应了怎样的行为。

## 从完整轨迹中学习

许多较弱的 NetHack Policy 并不会崩溃。它们会继续运行，却可能不断推撞一块
巨石、反复尝试穿过铁栏、在两个格子之间来回移动，或者站在楼梯上却不下楼。
聚合分数能说明 Policy 表现不佳，却不能指出行为究竟在哪里出了问题。

因此，每次训练 Submission 之后，Environment 都会返回所有评测 Episodes 的完整
Policy 可见轨迹。Agent 可以检查位置、Actions、消息、状态变化与重复状态，再将
这些模式对应回源代码。

```text
提交可执行 Policy
        ↓
检查完整训练轨迹
        ↓
定位行为失败
        ↓
重写记忆、寻路或交互规则
        ↓
提交新的 Program
```

Environment 提供证据，而不是诊断。选择检查哪些 Episodes、判断哪些模式值得关注，
仍然是 Coding Agent 工作的一部分。持久的改进保存在可执行 Program 中，而不是
模型权重或跨 Episode 继承的隐藏状态中。

## 实验

我们让三个 GPT-5.6 模型变体——Luna、Terra 和 Sol——通过 Codex 运行。每个 Agent
都从同一个 packaged baseline 开始，并可以利用训练分数与轨迹改进它。实验关闭了
可选的 NetHack 优化 Skill，因此 Agent 必须自行形成分析与修改流程。

Baseline 已经能够进行简单的局部探索。它记录位置访问次数，通常优先选择访问较少
的相邻格子，在存在其他选择时避免立刻折返，定期搜索，踢开可见的关闭房门，并在
饥饿时吃掉可识别的食物。但它不会构建显式地图、规划前往远处目标的路线、记忆
失败的边，也不会把下楼作为一个持续目标。

| 设置 | 值 |
| --- | --- |
| Environment | NLE 1.3.0 · NetHack 3.6.7 |
| 任务 | `NetHackScore-v0` · 23 个 Actions |
| Episode 限制 | 5,000 个 Policy steps |
| 训练额度 | 最多 128 个 Episodes |
| 最终 Assessment | 256 个 held-out Episodes |
| 可选 NetHack Skill | 关闭 |

训练额度是上限，并不要求必须用完。Sol 使用了全部 128 个 Episodes，Terra 使用了
68 个，Luna 使用了 40 个，随后结束运行。

## 结果

Sol 在这次实验中产生了最强的最终 Policy。它的平均 Assessment return 为
`204.026`，平均游戏分数为 `208.230`，平均地下城深度为 `2.867`。其最深的
held-out Episode 到达了第 11 层。

| Agent 路线 | 使用的训练额度 | Submissions | Assessment return | 平均游戏分数 | 平均 / 最大深度 | 冻结步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GPT-5.6 Sol + Codex** | 128 / 128 | 8 | **204.026** | **208.230** | **2.867 / 11** | **27.53%** |
| GPT-5.6 Terra + Codex | 68 / 128 | 4 | 80.237 | 87.094 | 1.082 / 4 | 33.77% |
| GPT-5.6 Luna + Codex | 40 / 128 | 4 | 63.773 | 70.777 | 1.094 / 4 | 35.75% |

三个最终选中的 Policy 都在 held-out Assessment 中完成了全部 Episodes，没有出现
Policy 执行失败。它们都没有完成 ascension。因此这些结果衡量的是游戏前期探索、
生存和地下城推进，而不是对完整 NetHack 的掌握。

![Sol 最终选择的 Policy 在一个 NetHack 训练 Episode 中的完整语义回放。回放覆盖
全部 1,269 个 Policy steps，到达地下城第 11 层，最高游戏分数为 860，最终以死亡
结束。](/images/blog/nle-sol-policy-training-replay.gif)

*第 000008 次 Submission 的第 16 个训练 Episode。这段回放展示了 Agent 编写的
Policy 如何从第一次 observation 开始自主行动，直至 Episode 结束。它是一个具有
代表性的训练轨迹，不属于表格中报告的 held-out Assessment。*

## Baseline 的能力边界

Baseline 能够回答一个有用的局部问题：

> 哪一个可见的相邻格子访问次数最少？

但它还无法回答更完整的导航问题：

> 如何在一座充满不确定性的多层地下城中构建并维护一条路线？

两者之间的差距，正是 Policy evolution 的主要机会。要超越 baseline，Policy
需要在一个位置离开屏幕后仍然保留相关知识，跨越多个房间追踪目标，并在 Action
失败后修改计划。

## Sol 构建了怎样的策略？

Sol 将局部探索 baseline 转化为一个更有结构的导航系统。它的最终 Policy 结合了
三个相互增强的思路。

### 持久的空间记忆

Policy 记录已经发现的地形、路线、目标和移动结果。它不再只是反复选择相邻格子，
而是能够利用先前的 observation，在已知房间和走廊中导航。角色离开当前位置后，
之前获得的信息仍然有用。

### 目标导向的探索

Sol 将地下城推进变成了一个显式目标。当 Policy 已经知道向下楼梯的位置时，它会
保留这个目标、规划路线返回并使用楼梯；当还不知道楼梯位置时，它会寻找尚未探索
的空间，而不只是选择当前可见范围内访问最少的格子。

这使地下城深度不再只是 Episode 结束后观察到的指标，而成为写在可执行策略中的
目标。

### 失败检测与恢复

NetHack 中的移动可能因为墙壁、巨石、铁栏、生物、房门或对地图的过期理解而失败。
Sol 会将计划中的移动与下一个 observation 进行比较。如果预期的状态转移没有发生，
Policy 就可以把路线标记为阻塞、丢弃无效目标，并选择其他路径，而不是继续重复
同一个 Action。

最终得到的能力并不是一组互不相关的特殊处理，而是一个更通用的循环：观察结果、
更新内部地图，然后重新规划。

## Agent 如何使用 Environment Feedback

**Luna——记住障碍。** Luna 发现一些轨迹主要由反复撞击巨石或尝试穿过铁栏组成。
它加入了对失败方向的记忆，减少了对静态障碍的立即重试。

**Terra——摆脱循环并使用楼梯。** Terra 引入了 anti-loop 行为和显式下楼逻辑。
最终选中的最强 candidate 来自一个较早的版本，这说明继续编辑并不一定会产生更好
的 Policy。

**Sol——将局部修复组织成导航系统。** Sol 结合记忆路线、面向楼梯的推进、阻塞边
处理和目标恢复。它没有把每次失败都看作孤立补丁，而是通过位置、路线、目标和
移动结果的共享模型将这些经验连接起来。

在每个案例中，持久的结果都不是 Agent 对轨迹的解释。这些经验必须成为代码，才能
在 held-out evaluation 中独立接收 observations 并返回 Actions。

## 发现与边界

完整语义轨迹足以让 Agent 定位行为失败并完成有效的 Policy 修改，不需要
Environment 预先编写诊断。这项实验也说明，解释 NetHack 行为需要多个指标：
return 用于排序 candidates，而深度、游戏分数和冻结步比例则揭示探索与推进的不同方面。

这些结果仍然只是初步研究。Agent 使用的训练额度并不相同，每条模型路线也只有
一次主要 Run。在最多 128 个 Episodes 的短训练上限下，Coding Agent 搜索可能因
随机性产生波动：相同 Environment 配置下，另一次 Terra Run 得分为 `49.464`，
而这里报告的结果为 `80.237`。

因此，表格应当被理解为这些 Agent 在对应 Runs 中构建出了更好的 NetHack Policy，
而不是 Luna、Terra 和 Sol 的普遍排名。

## 代码与说明

- [NLE NetHack Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/nle/nethack)
- [Evaluation 与 Runs](/docs/evaluation/)
- [Policy 边界](/docs/policy/)

EvoPolicyGym adapter 使用 MIT 许可证。NLE 与 NetHack 是独立依赖，并分别受各自的
许可证约束。
