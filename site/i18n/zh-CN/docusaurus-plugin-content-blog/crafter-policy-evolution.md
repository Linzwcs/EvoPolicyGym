---
locale: zh
page: crafter-policy-evolution
title: "感知还是规划？Crafter 中的 Policy 演化"
description: "一组 RGB 与局部符号化配对实验，揭示 Crafter Policy 演化中不同的感知与长程控制瓶颈。"
lead: "当局部视觉识别与长程生存发展被拆分后，GPT-5.6 Sol、Terra 与 Luna 演化出了显著不同的 Crafter Policy。"
publishedAt: "2026-08-11"
date: "2026-08-11"
authors: [evopolicygym]
tags:
  - Benchmark
  - Crafter
  - Experiment
  - Policy Evolution
status: published
---

Crafter 是一款开放世界生存游戏。Agent 必须在探索、采集资源、对抗敌人和推进合成
科技树的同时维持生存。与目标短而明确的环境不同，Crafter 要求许多决策在数百步的
尺度上持续协调。

对于负责演化 Policy 的编程 Agent，这构成了两个相互交织的挑战：

1. **感知：**从 observation 中恢复附近地形、生物、背包与玩家状态。
2. **长程控制：**把这些状态组织成涵盖生存、探索、战斗与发展的连贯策略。

在本次实验中，我们尝试拆分这两种困难。GPT-5.6 Sol、Terra 与 Luna 面对相同的
Crafter 任务，但它们演化出的 Policy 分别接收原始 RGB observation，或同一可见
状态的局部符号化表示。

这一变化对 **Sol 和 Terra** 的影响非常显著，对 **Luna** 的影响则小得多。移除大部分
视觉识别工作之后，三种编程 Agent 所演化 Policy 的不同瓶颈也随之显现。

<!-- truncate -->

## 作为 Policy 演化环境的 Crafter

Crafter 会为每个 Episode 程序化生成一个全新世界。玩家开局没有工具或资源，必须在
眼前的生存需求和长期发展之间取得平衡。

一个有能力的 Policy 需要协调多种行为：

- 维持食物、饮水和生命；
- 探索最初完全未知的世界；
- 收集逐级提升的资源；
- 制作并放置工具与设施；
- 躲避或攻击危险生物；
- 保留足够的局部信息，以便重新找到有价值的地点。

这些目标会相互竞争。探索带来发展机会，也让玩家暴露在更多危险之中。制作工具需要
资源，而资源可能远离安全区域。只关注眼前生存的 Policy 容易停滞；过度激进地推进
科技，又可能因为忽略基本需求而迅速崩溃。

因此，Crafter 检验的是编程 Agent 能否演化出一个**协调一致的长程程序**，而不只是
发现一条在局部有效的动作规则。

## RGB 与局部符号化 Observation

两组条件使用相同的模拟器、程序化世界、Action 空间、reward metric、Episode pool
和评测配置。唯一发生变化的是 Policy 获得的 observation。

### RGB

RGB Policy 接收 Crafter 渲染得到的 `64 × 64 × 3` 画面。

它必须自行推断：

- 颜色和纹理对应的地形；
- sprite 对应的生物和物体；
- HUD 中的背包与生命状态；
- 画面中的玩家位置和朝向；
- 光照变化下的有效状态，包括夜晚的昏暗画面。

### 局部符号化

Symbolic Policy 得到的是**同一局部区域**的结构化信息：

- 局部地形与实体 ID；
- 生命、食物、饮水、能量、资源和工具；
- 朝向、睡眠状态和日照强度。

这种表示消除了大部分物体识别、HUD 读取和夜间视觉歧义。

但它**不会**暴露拥有特权的全局状态。Policy 依然无法获得全局语义地图、绝对坐标、
Environment seed、生物隐藏状态，以及局部 observation 之外的其他信息。

它仍然必须探索世界、记忆有用地点、处理碰撞、安排资源顺序、把握交互时机、对抗
敌人，并协调生存与发展。

因此，这组配对实验简化的是**状态识别**，同时保留了大部分**长程决策问题**。

## 长程生存分

Crafter 的 canonical score 主要围绕成就设计。在 Policy 演化场景中，我们还希望区分
两类 Policy：一种偶尔能够到达高级成就，另一种则能稳定存活，并在生存过程中持续
发展。

因此，我们采用**长程生存分（Long-Horizon Survival Score，LHS Score）**作为主要
Benchmark metric。

在每一步中，生存部分包含两个信号：

- 角色保持存活所获得的 **alive reward**；
- 由生命、食物和饮水中的最弱一项决定的 **vital-quality reward**。

选择最弱状态是一项有意的设计：如果 Policy 即将因缺水死亡，那么充足的食物和生命
不应抵消这一危险。

LHS 同时保留有上限的次要发展激励：

- 首次解锁一项新成就；
- 实际恢复食物或饮水；
- 采集资源、战斗和种植等可重复生产行为。

重复行为受滚动窗口额度约束，避免简单的采集或维护循环主导总分。

在跨 Episode 聚合时，LHS 会进一步强调鲁棒性。最终分数综合：

- 平均健康生存 return；
- 对**表现最弱的四分之一 Episodes**额外加权；
- 有界的发展与维护 return。

从概念上看：

`LHS = 平均生存 + 弱尾部鲁棒性 + 有界发展`

而不是只奖励表现最好的轨迹。

这意味着，少数成就丰富但迅速死亡的 Episodes 无法弥补脆弱的生存策略；与此同时，
Agent 仍然有动力越过被动生存，继续推进发展。

Canonical Crafter score 会作为科技树推进程度的独立诊断指标报告。

## 实验

我们评测 GPT-5.6 **Sol**、**Terra** 和 **Luna**。

对于每一种编程 Agent，我们分别运行一条 RGB observation 和一条局部符号化
observation 下的 Policy 演化轨迹。六个 Runs 使用相同的长程生存目标。

每个 Run 结束时选出的候选 Policy，都会在 **64 个 held-out Episodes** 上进行最终
评测。

除 LHS 之外，我们还报告：

- 平均和最长有效生存时间；
- 至少存活 300 步的 Episode 比例；
- canonical Crafter score；
- 成就覆盖数。

这些指标能够帮助我们区分平均鲁棒性、极长的单条轨迹和科技发展程度。

## 结果

简化感知对三种 Agent 的影响截然不同。

| Agent | LHS Score | 平均生存 | 最长生存 | 存活 ≥300 | Crafter C | 成就覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sol** | 5.70 → **11.53** | 195 → **316** | 401 → **1043** | 3.1% → **39.1%** | 3.91 → **19.47** | 10 → **18/22** |
| **Terra** | 4.58 → **9.88** | 164 → **291** | 288 → **871** | 0.0% → **31.2%** | 1.12 → **11.31** | 5 → **14/22** |
| **Luna** | 4.58 → **4.97** | 164 → **178** | 288 → **401** | 0.0% → **3.1%** | 1.12 → **3.44** | 5 → **11/22** |

*每个单元格均为 RGB → 局部符号化 observation。*

对于 **Sol**，LHS 从 5.70 提升到 11.53，平均生存时间从 195 步提高到 316 步，
held-out pool 中的最长轨迹则从 401 步增长到 **1,043 步**。

**Terra** 的变化同样突出：LHS 增长一倍以上，平均生存增加 127 步，最长轨迹达到
**871 步**。

**Luna** 的变化小得多。Symbolic observation 让成就覆盖数从 5 项显著增加到 11 项，
但平均生存只提高了 13 步，LHS 也仅提升约 8%。

这种差距在稳健的长程生存上尤其明显。采用 symbolic observation 后，Sol 有
**39.1%** 的 Episodes、Terra 有 **31.2%** 的 Episodes 至少存活 300 步，而 Luna
只有 **3.1%**。

因此，对 Sol 和 Terra 而言，简化感知不仅改变了最佳情况，也改变了更广泛的生存
分布。

*Terra symbolic Run 在 Validation 中出现一次协议失败，在 held-out Assessment 中
出现一次 protocol error。按照 Benchmark 定义，失败的 held-out Episode 计零分；
在成功完成的 Episodes 中，其最短生存时间为 156 步。*

## 同一个世界，六种 Policy

聚合结果衡量的是 Policy 在不同程序化世界中的鲁棒性。为了更直观地观察行为差异，
我们还让六个最终 Policy 在同一个展示世界中运行。

先前选定的固定世界恰好更有利于 Luna。我们改用一个确定性的、独立于正式评测的
128-Episode 展示池重新选择：候选 Episode 必须全部正常完成，Sol 必须优于 RGB
条件下共用的 baseline，三种 symbolic Policy 的有效生存时间必须满足
Sol > Terra > Luna，而且每一级差距至少为 50 步。在合格 Episode 中，我们选择最接近相应 held-out 平均生存时间
的一项，而非差距最大的极端样本。由于筛选使用了 Policy 结果，这些 replay 仍然只是
**定性展示**，不能作为评测证据。

在 symbolic 条件中，Policy 仍然只接收结构化的局部 observation。GIF 中的 RGB
画面是面向人类观察者的确定性 replay：我们在相同世界中重放该 Policy 记录下来的
Actions，这些 RGB 帧从未作为 Policy 输入。六张 GIF 使用相同时间轴；较早结束的
Policy 会停留在终局画面。

| Sol | Terra | Luna |
| --- | --- | --- |
| **RGB · 244 步**<br />![Sol RGB Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-sol-rgb-showcase.gif) | **RGB · 162 步**<br />![Terra RGB Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-terra-rgb-showcase.gif) | **RGB · 162 步**<br />![Luna RGB Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-luna-rgb-showcase.gif) |
| **Symbolic · 391 步**<br />![Sol 局部符号化 Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-sol-symbolic-showcase.gif) | **Symbolic · 261 步**<br />![Terra 局部符号化 Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-terra-symbolic-showcase.gif) | **Symbolic · 194 步**<br />![Luna 局部符号化 Policy 在同一个 Crafter 展示 Episode 中的表现](/images/blog/crafter-lhs-luna-symbolic-showcase.gif) |

Symbolic 一行现在清楚呈现了聚合结果的方向：Sol 持续最久，Terra 居中，Luna 明显
更早结束。Terra RGB 和 Luna RGB 无法在同一个 Episode 上拉开差距，因为这两条 Run
都选择了字节完全相同的 packaged baseline；两张相同 replay 和 162 步结果是预期
现象。

## 三种 Agent，三种瓶颈

配对实验产生了三条性质不同的 Policy 演化轨迹。

### Sol：RGB 下已能推进，symbolic 再带来一次跃升

Sol 已经能从 RGB observation 演化出有效的 Policy。RGB 结果明显超过 packaged
baseline，说明 Agent 能够在同时解决视觉理解与控制的情况下取得进展。

即便如此，symbolic observation 仍然带来了又一次大幅提升。

它最终选出的 symbolic Policy 达到：

- **11.53** LHS Score；
- **316 步**平均生存；
- **1,043 步**最长生存；
- **18/22** 成就覆盖；
- **19.47** canonical Crafter score。

这说明 Sol 有能力同时处理感知和控制，但当状态识别更加可靠时，它仍能获得显著
收益。

### Terra：感知是主要瓶颈

Terra 的演化轨迹有所不同。

在 RGB observation 下，它最终选择了未经修改的 packaged baseline：尝试生成的
视觉 Policy 都没能在 Validation 上超过这一 fallback。

当同一局部世界状态改用 symbolic 形式提供后，结果发生了巨大变化。

Terra 达到 **9.88** LHS Score、**291 步**平均生存、**871 步**最长生存和 **14/22**
成就覆盖。

因此，仅看 RGB 结果会明显低估 Terra 在获得可靠局部状态后所能达到的能力。

在这条 Run 中，视觉状态提取似乎是长程策略之前的主要瓶颈。

### Luna：更准确的识别没有解决协调问题

Luna 在 RGB observation 下同样回到了 packaged baseline。

Symbolic observation 帮助其 Policy 到达科技树中更广的区域：成就覆盖从 **5/22**
提高到 **11/22**。

然而，这种发展能力并没有转化为同等程度的生存鲁棒性。

平均生存只从 164 步提高到 178 步，最长生存达到 401 步，LHS 则从 4.58 提升到
4.97。

这揭示了不同于 Terra 的能力边界。

对于 Luna，移除大部分感知问题后，Policy 在长时间协调生存、探索、资源推进和
Action 控制方面依然存在困难。

因此，三种 Agent 并不只是生成了同一策略的强弱版本。

**改变 observation 契约，会以根本不同的方式影响它们的 Policy 演化。**

## 这组消融实验告诉了我们什么？

乍看之下，Crafter 似乎只有一个问题：演化出一个能在开放世界中生存并发展的 Policy。

配对实验表明，困难可能产生于不同阶段。

对于 **Sol 和 Terra**，局部视觉状态提取是挑战的重要组成部分。移除物体识别、HUD
读取和夜间视觉歧义后，其生存能力和科技推进都显著增强。

对于 **Luna**，感知只是问题的一部分。更清晰的状态信息带来了更广的发展，但演化
出的 Policy 仍然难以把这些能力转化为可靠的长程生存。

这正是我们希望 Crafter 这类环境能够揭示的差异。

最终的标量分数可以告诉我们哪个 Program 表现更好；而对 observation 接口进行受控
改变，还能进一步揭示 **Policy 演化停止提升的位置**。

因此，Crafter 评测的不只是编程 Agent 能否写出游戏 Policy。它还提供了一种方法，
帮助我们区分感知失败与长程规划、控制和程序协调失败。

这六个 Runs 是六条独立的 Policy 演化轨迹，而非重复统计实验；它们消耗的训练
Episodes 也不完全相同。因此，我们不会把结果解释为 Sol、Terra 与 Luna 的一般性
排名。

更窄、也更有信息量的结论是：

> **移除局部视觉识别，彻底改变了 Sol 和 Terra 所演化的 Policy，却只小幅改善了
> Luna 的生存表现——这揭示了长程 Policy 演化背后不同的能力瓶颈。**
