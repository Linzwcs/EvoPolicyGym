---
locale: zh
page: crafter-policy-evolution
title: "生存、建造、循环：Crafter 中的 Policy 演化与奖励塑形"
description: "成就、生存与重复生产 Feedback 如何改变 Crafter 中可执行 Policy 的演化方向。"
lead: "GPT-5.6 Luna、Terra 与 Sol 的奖励消融实验，揭示了延长生存与推进 Crafter 科技树之间的张力。"
publishedAt: "2026-08-09"
date: "2026-08-09"
authors: [evopolicygym]
tags:
  - Benchmark
  - Crafter
  - Experiment
  - Policy Evolution
status: published
---

## Crafter 是什么？

Crafter 是一款围绕探索、资源采集、合成、战斗与生存构建的开放世界游戏。每个
Episode 都发生在随机生成的世界中。玩家必须寻找食物和水，躲避或对抗生物，收集
材料，并逐步从木石工具发展到铁和钻石。

<!-- truncate -->

一个简化的推进循环如下：

```text
探索世界
    ↓
收集食物、水和材料
    ↓
存活到足以制作工具
    ↓
解锁新的资源与能力
    ↓
完成更困难的成就
```

Crafter 评测 22 项有明确语义的成就，覆盖资源、生物、工具、种植与建筑。标准分数
对各项成就的成功率计算平移后的几何平均，因此奖励科技树上的广泛推进，而不是
反复完成某个简单任务。

这形成了一个策略难题：推进需要生存，但只优化生存又可能产生过度保守、始终无法
进入科技树深处的行为。

## 将 Crafter 接入 EvoPolicyGym

该 Benchmark 使用 Crafter 1.8.3，Episode 上限为 10,000 步。Policy 只接收
`64 × 64 × 3` RGB observation，并从 17 个离散 Actions 中选择。生命、食物、水和
背包只能通过渲染画面读取；玩家坐标、语义图和 Environment seed 不会作为结构化
Policy 输入暴露。

标准 Crafter 分数能够衡量成就广度，却不能直接区分“形成了可持续的长期生存循环”
与“短暂解锁几项成就后立即死亡”。因此，我们比较了四种聚合 Feedback profile：

| Profile | Feedback score | 目的 |
| --- | --- | --- |
| **M1 · 成就** | `C` | 标准 Crafter 成就推进 |
| **M2 · 生存** | `C + L / 100` | 加入较弱的生存激励 |
| **M3 · 生存 + 重复** | `C + L / 100 + R` | 平衡推进、生存与可重复生产 |
| **M4 · 强生存 + 重复** | `C + L / 20 + R` | 显著提高生存优先级 |

`C` 是标准 Crafter 成就分，`L` 是平均有效存活长度，`R` 奖励经过环境确认的重复
采集、喝水、进食、建造和击败生物等活动。某项成就的第一次成功仍只属于 `C`，
之后的成功才进入 `R`。重复分按对数增长，并受到随 Episode 长度连续增长的约束，
因此短命 Policy 不能依靠一个简单循环取得任意高分。

这四种 profile 都**不会修改 Crafter 原始的逐步 reward**，改变的只是用于比较
submitted Programs 的聚合 Feedback。

## 从视觉轨迹中学习

聚合分数可以说明 Policy 是否提升，但许多 Crafter 失败更容易通过视觉理解。Policy
可能在一小片区域绕圈，忽略食物或水，长期停留在低级资源阶段，或者发现一个生产
循环后不再继续发展。

因此，每次训练 Submission 后，Agent 都会获得完整的 Policy 可见 Action 轨迹和
无损 RGB observations；Validation 与 held-out test 轨迹保持私有。Coding Agent
可以使用 NumPy、Pillow 和图片查看工具检查 Policy 实际看到和执行的内容，再修改
可执行策略。Environment 只提供证据；选择检查什么、如何解释证据，仍属于 Agent
自身的工作。

## 实验

我们通过 Codex 分别运行 GPT-5.6 Luna、Terra 和 Sol。每个 Agent 都从同一个
packaged baseline 开始，并在四种 reward profile 下独立优化。12 个 Runs 使用相同
的 Environment、split 构造和 Run seed；在每条模型路线内，实验变量是 reward
profile。可选的 Benchmark Skill 保持关闭。

| 设置 | 值 |
| --- | --- |
| Environment | Crafter 1.8.3 |
| Policy 输入 | `64 × 64 × 3` RGB |
| Actions | 17 |
| Episode horizon | 10,000 steps |
| 训练额度 | 最多 256 Episodes |
| 单次 Submission 上限 | 16 Episodes |
| Validation | 每个 candidate 32 Episodes |
| 最终 Assessment | 64 个 held-out Episodes |
| Agent harness | Codex，high reasoning effort |
| 可选 Benchmark Skill | 关闭 |

训练额度是上限，不要求必须用完，因此 Agent 可以在使用完 256 Episodes 之前结束。

在固定 held-out pool 上，baseline 的 Crafter 成就分为 `1.025`，平均有效存活
`163.4` 步，仅有 `5/22` 种成就取得非零成功率。其行为主要集中在获取木材、喝水
和收集树苗。

## 结果

三个 Agent 在各自 Run 所使用的聚合目标上都超过了 packaged baseline，但不同
Agent 和目标产生的最终 Policy 有明显差异。

| Metric | Agent | 最终分数 | 成就 `C` | 生存 `L` | 重复 `R` | 成就覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **M1 · 成就** | **Sol** | **12.748** | **12.748** | 183.8 | — | **17/22** |
|  | Luna | 3.461 | 3.461 | 163.3 | — | 11/22 |
|  | Terra | 1.655 | 1.655 | 176.8 | — | 7/22 |
| **M2 · + 生存** | **Sol** | **10.110** | **8.308** | **180.2** | — | **15/22** |
|  | Luna | 3.563 | 1.803 | 176.0 | — | 8/22 |
|  | Terra | 3.918 | 2.134 | 178.5 | — | 7/22 |
| **M3 · + 生存 + 重复** | Sol | 7.073 | 4.007 | 177.8 | 1.288 | 10/22 |
|  | Luna | 6.651 | 2.305 | 177.7 | **2.570** | 9/22 |
|  | **Terra** | **7.997** | **4.092** | **186.8** | 2.037 | **11/22** |
| **M4 · 强生存 + 重复** | **Sol** | **15.501** | **4.717** | 176.5 | 1.957 | **14/22** |
|  | Luna | 14.193 | 2.253 | **185.9** | 2.647 | 10/22 |
|  | Terra | 13.055 | 1.023 | 184.7 | **2.799** | 5/22 |

最终分数只能和对应 profile 下的 baseline 比较，不能直接跨 profile 排序：M4 为
每个存活步赋予的分数本来就高于 M1–M3。

M1 产生了最广的科技树推进。Sol 达到 `17/22` 项成就和 **12.748** 的标准 Crafter
分数，而 Luna 和 Terra 分别达到 `11/22` 与 `7/22`。M2 让三个 Agent 的 Policy
都延长了生存，同时 Sol 仍然保持了明显更广的成就推进。

M3 得到了最均衡的结果。在该 profile 内，Terra 以 **7.997** 取得最高最终分数，
同时达到 `C = 4.092`、平均有效存活 `186.8` 步，并在 `11/22` 项成就上获得非零
成功率。它的重复分来自木材和石头采集、放置石头、吃牛与喝水等多种活动，而不是
只依赖一种维持行为。

三个 M4 Run 的平均生存更长，但各 Policy 并不都具备广泛推进能力。Sol 仍达到
`14/22` 项成就，而 Terra 只有 `5/22`，与 baseline 相同。Terra 的标准成就分
`1.023` 也略低于 baseline，尽管其 M4 聚合总分高出很多。

随着生存激励增强，三个 Agent 的宏平均呈现出一致的取舍：

| Metric | 平均成就 `C` | 平均生存 `L` | 平均重复 `R` |
| --- | ---: | ---: | ---: |
| M1 | **5.955** | 174.6 | — |
| M2 | 4.082 | 178.2 | — |
| M3 | 3.468 | 180.8 | 1.965 |
| M4 | 2.664 | **182.4** | **2.468** |

平均存活长度从 `174.6` 单调上升至 `182.4` 步，同时平均标准 Crafter 分数从
`5.955` 降至 `2.664`。这说明奖励塑形改变了 Policy evolution 的方向，而不是
仅仅重新缩放了相同行为的显示分数。

![Validation 选中的 Sol、Luna 与 Terra Policy 在三段代表性 M3 训练 Episode
中的逐步对齐并排回放。三个 Policy 在不同的生成世界中进行探索、采集、制作与
生存。](/images/blog/crafter-m3-policy-replay-comparison.gif)

*三段代表性、非同 seed 的 M3 训练轨迹。Sol 展示 Submission 000010 的 train
index 159，在 296 步内完成 7 项成就；Luna 展示 Submission 000004 的 index 62，
在 194 步内完成 7 项成就；Terra 展示 Submission 000008 的 index 120，在 294
步内完成 8 项成就。回放在共享 step 时间轴上每三个 Policy steps 显示一帧；某个
Episode 结束后，其终止 observation 会继续保留。这些 Episodes 不属于 held-out
Assessment。*

## Baseline 的能力边界

Baseline 能够回答一个游戏前期的问题：

> 如何获取附近资源并满足眼前的生存需求？

但完整的 Crafter 问题是：

> 如何在一个很长的 Episode 中协调生存、探索、生产与科技树推进？

`5/22` 的成就覆盖说明 baseline 可以进入游戏前期，却很少形成广泛的推进链条。
Policy evolution 不仅要改进单个 Action，还要决定何时继续采集熟悉资源、何时探索、
何时投资工具和建筑，以及应当投入多少精力维持生存。

## 奖励塑形如何改变 Policy evolution

**M1 偏向科技树广度。** 没有显式生存奖励时，完成新的成就类型是提升分数的主要
途径。

**M2 在不大幅改变任务的前提下加入生存。** 三个 Agent 的平均存活长度比 baseline
增加约 15 步，同时成就分均高于 baseline。

**M3 奖励可持续推进。** 一项成就首次完成后，重复生产、食物、战斗和建筑仍可提供
反馈；较弱的生存项则减少立即死亡，又没有压过原始成就目标。

**M4 将取舍推得过远。** 相比 M3，它只增加约 `1.6` 个平均存活步，却损失约 `0.8`
个平均成就分。喝水一项就占 Sol、Luna、Terra 重复分的约 65%、86% 和 83%。更强
的生存项使低风险维持循环具有了过高吸引力。

这也是可执行 Policy Benchmark 中一种值得研究的失败模式：Coding Agent 能够发现
并编码利用 Environment 持久激励的行为。

## 发现与边界

在这四种 profile 中，**M3 是最有价值的长生存变体**。它获得了 M4 大部分的生存
提升，同时保留更多成就推进，并支持更丰富的可重复活动。M1 仍适合作为标准 Crafter
对照，M2 是简单的生存 hybrid，M4 则适合作为强生存消融。

这仍是一次初步奖励消融。每个“模型 × profile”组合只有一条主要 Coding Agent
trajectory，而且 Agent 使用的训练额度并不相同。这些结果说明奖励设计在这些 Runs
中系统性地改变了 Policy evolution，但不能视为 Luna、Terra 和 Sol 的普遍排名。

更广泛地说，Crafter 为 EvoPolicyGym 展示了一个关键张力：**好的 metric 应奖励
Policy 活得足够久以形成策略，又不能让生存本身比推进更容易优化。**

## 代码与说明

- [Crafter 上游 Environment](https://github.com/danijar/crafter)
- [EvoPolicyGym Crafter Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/crafter/crafter)
- [Evaluation 与 Runs](/docs/evaluation/)
- [Policy 边界](/docs/policy/)

EvoPolicyGym adapter 使用 MIT 许可证。Crafter 是独立依赖，并受其自身许可证约束。
