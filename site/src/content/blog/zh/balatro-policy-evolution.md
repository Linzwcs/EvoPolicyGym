---
locale: zh
page: balatro-policy-evolution
title: "让 Coding Agent 编写打《小丑牌》的策略系统"
description: "我们接入了《小丑牌》环境，并比较Luna、Terra 与 Sol 在1024的环境交互budget下的策略优化结果引出的一些启发和思考。"
lead: ""
publishedAt: "2026-07-29"
author: "EvoPolicyGym contributors"
tags:
  - Benchmark
  - Balatro
  - Experiment
  - Policy Evolution
status: published
---

## 《小丑牌》是什么

《小丑牌》（Balatro）是一款围绕扑克牌型计分的 roguelike 牌组构筑游戏。玩家
从一副基础扑克牌开始一局 Run，通过出牌获得分数、通过商店强化构筑，最终击败
Ante 8 的 Boss Blind 完成通关。每局的抽牌、商店和奖励都会变化，失败后重新
开始一局。

一局游戏反复执行同一个循环：

```text
选择 Blind → 出牌或弃牌 → 达到目标分数 → 结算金钱 → 商店构筑 → 下一个 Blind
```

- **关卡**：一局分为 8 个 Ante，每个 Ante 包含 Small Blind、Big Blind 和
  Boss Blind。Small Blind 与 Big Blind 可以跳过并换取 Tag；Boss Blind 会
  加入一条改变本轮玩法的规则。
- **出牌**：玩家每次从手牌中选择 1–5 张牌。对子、两对、顺子、同花等牌型决定
  基础 Chips 和 Mult，计分牌与其他效果继续修改两者，这一手最终获得
  `Chips × Mult` 分数。
- **过关**：同一个 Blind 中，多次出牌的分数会累积。达到目标 Chips 后通过
  Blind；出牌次数耗尽仍未达标则结束本局。
- **弃牌**：discard 可以丢弃不需要的牌并抽取新牌，用有限次数改善后续手牌。
- **构筑**：通过 Blind 后获得金钱并进入商店。Joker 改变计分方式，Planet
  提升牌型等级，Tarot 和 Spectral 改造牌组，Voucher 与 Booster 提供长期
  强化，reroll 用金钱刷新商店。

游戏的核心是在当前过关与长期成长之间分配资源：选择哪一手牌、何时弃牌、购买和
排列哪些 Joker、是否保留现金、要不要跳过 Blind，以及怎样处理 Boss 规则。这些决策共同组成一套完整的《小丑牌》策略。

## 《小丑牌》接入EvoPolicyGym

在 EvoPolicyGym v0.3.0 中，我们以 vendored 方式接入了非官方 Balatro 引擎
[Jackdaw](https://github.com/TylerFlar/jackdaw-balatro)，并基于它实现了 Balatro
评测环境。

Policy每步可以获取公开 observation中包括：

- Ante、Blind、目标分数、剩余手数与弃牌次数；
- 手牌、Joker、Consumable、牌组公开统计和已有牌型等级；
- 当前商店、Booster、Voucher、Tag 与现金；
- 当前 phase 下严格枚举的合法 Actions；
- 可见对象在固定引擎版本中的规则说明。

Policy 返回语义化 Action，例如出牌、弃牌、购买、出售、reroll、开包或调整
Joker 顺序。无效 Action 不会被环境“修好”，而是直接记为 Policy failure。
同一个 Episode 内可以保存状态，但每个新 Episode 都会创建全新的 Policy 实例。

最终policy得分由下面公式计算。

```text
通过的 Blind 数 + 1000 × 是否通关
```

每通过一个 Blind 得 1 分，完成整局额外获得 1000 分。这个设计一方面保留了
“通关”这一最终目标，另一方面让尚未通关的 Policy 仍能通过平均推进距离获得连续
反馈。

## 简单介绍 EvoPolicyGym 的评测流程

在一次 EvoPolicyGym Run 中，Coding Agent 从初始 Program 出发，反复提交策略、
在训练 Episodes 上评测，并根据分数和 replay 继续优化。Agent 完成后，
Validation 选择最终 Program，Assessment 再使用 held-out test Episodes 得到
最终成绩。每次提交的 Program、反馈、replay 和优化记录都会保存在 Run 数据中。

## 实验

我们比较 packaged baseline，以及 Luna、Terra、Sol 三个 Coding Agent 优化
Run 最终交接的 Program。三个 Agent 都从同一个 baseline 开始，使用相同的训练
Episode pool 和环境交互 budget。

Baseline 是一套确定性的扑克牌型策略。它会穷举手牌中所有 1–5 张组合，按照传统
牌型等级和牌面点数选择出牌；进入商店后购买第一张买得起的 Joker，开包时选择
第一张 Joker。它不使用 discard、Consumable 和 reroll，也不管理 Joker 组合与
经济。

Luna、Terra 和 Sol 的训练 Episode pool 与总 Episode budget 均为 1024。
优化结束后，我们冻结三个 Agent 最终选择的 Program，并使用 `epg2` 引擎在完全
相同的 128 个 held-out test Episodes 上评测。四个 Program 的评测条件如下：

| 实验 | Agent | Reasoning | 训练 Episode budget | Test Episodes |
| --- | --- | --- | ---: | ---: |
| Packaged baseline | — | — | 0 | 128 |
| Luna | `gpt-5.6-luna` | `xhigh` | 1024 | 128 |
| Terra | `gpt-5.6-terra` | `xhigh` | 1024 | 128 |
| Sol | `gpt-5.6-sol` | `xhigh` | 1024 | 128 |

实验使用 Red Deck、White Stake，Run seed 为 `20260729`，单个 Episode 的超时
时间为 60 秒。

## 实验结果

<figure class="blog-result-figure">
  <img
    src="../../images/blog/balatro-heldout-results.svg"
    alt="四个 Program 的 Balatro held-out test 结果。Sol 平均通过 10.45 个 Blind，在 128 局中通关 5 次，最终均分 49.52。"
    loading="lazy"
    decoding="async"
  />
</figure>

Sol 是唯一完成 Run 的策略：它平均通过 10.45 个 Blind，是 baseline 的 2.83
倍，并在 128 局中通关 5 次。Luna 和 Terra 的平均推进距离也超过 baseline 的
两倍，但没有通关。减少早期错误可以走得更远，完成 Run 还需要一套连贯的长期
构筑策略。

## Baseline 的能力边界

Packaged baseline 会穷举当前手牌中所有 1–5 张组合，优先选择传统牌型等级和
牌面点数更高的组合；进入商店和开包后，则购买第一张满足条件的 Joker。

它解决了“当前哪组牌型更高”，却没有把实际得分、弃牌、Joker 组合、经济和 Boss
规则连接起来。

## Sol 写出了什么策略

Sol 建立了一个近似局面模型，再按游戏阶段执行相互配合的启发式策略。最终 Policy
的关键变化可以归纳为三个方面。

### 估分与手牌规划

Sol 仍然穷举 1–5 张组合，但评价对象变成了预计的 `Chips × Mult`。估算同时考虑
牌型等级、计分牌、Enhancement、Edition、持有牌效果和 Joker 组合。

Policy 再结合当前目标分数、剩余手数和弃牌次数，决定立即出牌还是继续找牌。它会
保护具有持有价值的牌，也会在没有 discard 时用低价值牌换取新手牌。

### Joker 构筑与经济管理

Sol 会估算 Joker 的价值，组合稳定的 Chips、Mult 和 X Mult，并根据效果依赖调整
顺序。槽位满时，只有候选 Joker 明显更强才会替换已有组件。

商店决策也与构筑相连：Policy 根据 Ante、现金和关键缺口决定购买、保留利息或
reroll。现金不再只影响当前商店，而是用于提升后续回合的强度。

### 跨回合状态管理

Policy 会记录商店、开包、牌型和跳过 Blind 等状态，并根据 Boss 规则调整行为。
得分估算影响弃牌，弃牌影响过关概率，过关后的经济又改变下一轮构筑，这些决策
由此形成了一个完整循环。

## 模型如何利用环境反馈

在相同的 baseline、训练数据和 budget 下，Sol 是唯一完成 Run 的模型。它更有效
地利用了 replay 和分数反馈，把局部经验组织成涵盖出牌、构筑和经济的完整策略，
更像一名经验丰富的玩家。

Sol 的 Policy 也暴露了工程问题：约 1860 行逻辑集中在一个文件中，耦合度较高。
下一步是拆分策略模块，让它更容易测试、校准和继续优化。

## 下一步可以做的工作

第一个方向是 Skill。有效的 Skill 不只补充领域知识，还可以提供模块划分、replay
分析和测试方法。使用相同的 Agent 和 budget 对比有无 Skill，可以同时观察最终
得分与 Policy 的工程结构。

第二个方向是跨环境 RL。目标不是记住某个环境的规则，而是学习定位失败、提出假设、
设计实验和更新策略的方法，并将这种能力迁移到未见过的环境。

除此之外，在接入各个环境的过程中，我们发现了另一个有价值、但还欠研究的问题：
Agent 能否通过观察外部系统，自行构建适合训练和评测的环境？例如，Agent 能否
理解一款游戏的核心规则，实现一个行为等价的环境引擎，剥离美术、音频等与策略
无关的信息，只保留状态、动作和反馈？

这需要衡量 Agent 能否正确抽象状态、动作、反馈和评测规则。目前我们主要评测
Agent 使用环境优化策略的能力，对构建环境本身还缺少系统度量。如果 Agent 能做好
这一步，它构建的环境就可以接入 EvoPolicyGym，再由 Agent 继续优化其中的策略。
这种从观察、建模到优化的完整能力，将帮助未来的 Agent 更快适应新环境，并构建
有效的决策系统。

## 代码与说明

- [EvoPolicyGym Balatro Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/jackdaw/balatro)
- [EvoPolicyGym](https://github.com/Linzwcs/EvoPolicyGym)
- [Balatro 实验数据](https://huggingface.co/datasets/linzw/EvoPolicyGym-Exp-data/tree/main/v0.3.0/balatro)

本 Benchmark 与 LocalThunk、Playstack 及 Balatro 官方项目无关联，也不包含官方
卡面、美术、音乐、字体或其他游戏资源。
