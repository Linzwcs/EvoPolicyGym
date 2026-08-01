# Balatro epg2 留出集实验结果

## 实验目的

本报告比较 Balatro 仓库内置的 packaged baseline，以及 Luna、Terra、Sol
三组无 Skill 优化实验最终选出的 Program。所有结果均在修复 Ceremonial
Dagger 的 `destroy_joker` 选盲注 mutation 后，使用 vendored Jackdaw
`epg2` 引擎重新评测。

本文对名称作如下限定：

- **Packaged baseline**：`balatro.baseline_program()` 返回的原始起始
  Program，没有经过 Agent 优化。
- **Luna、Terra、Sol**：对应 Agent 优化 Run 最终选中的 Program。
- 之前名称为 `balatro-skill-*` 和 `balatro-no-skill-*` 的 Run 也是从
  packaged baseline 出发的优化实验，不能把它们的最终 Program 当作
  baseline。

## 评测设置

| 设置 | 取值 |
|---|---|
| Benchmark | `jackdaw/Balatro/red-deck-white-stake/run-score-v2` |
| 引擎版本 | `c84dca9+aaf24f9+8e807df+8dd6616+a785574+epg2` |
| 环境摘要 | `sha256:26ad26f704731977d73ef333ff7c7e365a5452d31b19bc1d39a59ed44ea254af` |
| 牌组 / 难度 | Red Deck / white stake |
| 内容配置 | `jackdaw-active-content-v1` |
| Assessment split | `test` |
| Assessment Episodes | 128 |
| Run seed | `20260729` |
| Assessment seed domain | `evopolicygym/assessment-seed/v1` |
| 单 Episode 超时 | 60 秒 |

四个 Program 使用完全相同的 128 个 held-out Episodes。Assessment seed
由 `evopolicygym.run._assessment._assessment_seed` 根据 Run seed
确定性派生。

Luna、Terra、Sol 都是不加载 Agent Skill 的优化 Run，Episode pool 和
Episode budget 均为 1024。Packaged baseline 没有优化阶段，训练预算记为
0。

## Program 来源

| 实验 | 模型 | Reasoning | 最终 Program | Program digest |
|---|---|---|---|---|
| Packaged baseline | 无 | 无 | [`baseline/policy.py`](../environments/jackdaw/balatro/src/balatro/programs/baseline/policy.py) | `sha256:685507392d70a9681170217a7671c70e5115bb99065aa11f0e5368dc7982c0c4` |
| Luna | `gpt-5.6-luna` | `xhigh` | [`submission-000022`](../runs/gpt-luna-balatro-no-skill-train1024-heldout128-seed20260729/submissions/submission-000022/program/policy.py) | `sha256:780e5a80eb407c4d932b50b25f9c1e082ace0c4f2b1af52e370331e2546b1781` |
| Terra | `gpt-5.6-terra` | `xhigh` | [`submission-000021`](../runs/gpt-terra-balatro-no-skill-train1024-heldout128-seed20260729/submissions/submission-000021/program/policy.py) | `sha256:f5ca7db4eadd83aae36dafc62e137d91f989c297cbbad9ac10532d1d4ed4ec9a` |
| Sol | `gpt-5.6-sol` | `xhigh` | [`submission-000039`](../runs/gpt-sol-balatro-no-skill-train1024-heldout128-seed20260729/submissions/submission-000039/program/policy.py) | `sha256:21ee6f811a5dbd0bebfa1f1965e8df0bae559e8e310f0e6e987e07a8e8a710ff` |

## 实验结果

| 排名 | 实验 | 平均分 | 相对 baseline | 平均通过 Blind | 平均到达 Ante | 胜局 | 胜率 | 最高单局总分 | Policy 失败 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Sol | **49.515625** | +45.820313 | 10.453125 | 4.296875 | **5/128** | 3.90625% | 1028 | 0 |
| 2 | Terra | **8.890625** | +5.195313 | 8.890625 | 3.460938 | 0/128 | 0% | 20 | 0 |
| 3 | Luna | **7.867188** | +4.171875 | 7.867188 | 3.062500 | 0/128 | 0% | 18 | 0 |
| 4 | Packaged baseline | **3.695313** | — | 3.695313 | 1.882813 | 0/128 | 0% | 16 | 0 |

主指标的计算方式是：

```text
mean(通过的 Blind 数 + 胜局奖励 1000)
```

Sol 在 128 个 Episodes 中共通过 1338 个 Blinds，并赢得 5 局，因此：

```text
(1338 + 5 * 1000) / 128 = 49.515625
```

去掉胜局奖励后，Sol 的平均进度是 10.453125 个 Blinds。最高单局通过
28 个 Blinds，获得 `28 + 1000 = 1028` 分。

## Sol 最终 Policy 策略解读

Sol 生成的是一个 1860 行、单文件、确定性的规则 Policy。它不是搜索未来
随机状态的规划器，也不读取隐藏 seed；核心思路是利用当前 observation
建立一个近似的局面价值模型，然后按阶段执行启发式决策。

### 1. Episode 内状态记忆

Policy 会在同一个 Episode 内记录：

- 当前商店属于第几轮；
- 本商店已经 reroll 和开包多少次；
- 当前 Blind 已打过哪些牌型；
- 每个 Ante 是否已经跳过一次 Blind。

这些状态用于限制消费频率，并处理依赖历史的效果，例如 Card Sharp、
The Mouth 和 The Eye。它不会把状态跨 Episode 保留。

### 2. 选 Blind 和跳 Blind

默认行为是接受 Blind，但在前期会为明确的高价值 Tag 跳过一次：

- 前 3 Ante 的 Investment、Polychrome、Buffoon、Top-up；
- 有足够现金时的 Economy；
- 前 6 Ante 的 Negative；
- 早期且 Joker 槽位有空间时的 Holographic、Foil；
- 资金较少时的 Coupon。

每个 Ante 最多主动跳一次，避免为了 Tag 连续牺牲过多普通 Blind 收益。

### 3. 手牌枚举和近似得分模型

Policy 穷举当前手牌所有 1–5 张组合，而不是只比较扑克牌型。每个候选组合
都会估算：

```text
chips × mult
```

估算中包含：

- 当前牌型等级的基础 chips 和 mult；
- 计分牌的点数、Enhancement、Edition；
- 手牌中未打出的 Steel 等 held-card 效果；
- Joker 的固定 chips、固定 mult、条件 mult 和 xMult；
- Blueprint、Brainstorm 的复制目标；
- Photograph + Hanging Chad、Card Sharp、Blackboard、Bloodstone、
  Acrobat、Bootstraps、Bull、Supernova 等组合或条件效果；
- Boss Blind 对基础 chips 和 mult 的削弱。

因此它选择的是“预计本手实际得分最高”的组合，而不一定是传统牌型等级最高
的组合。这是它与 packaged baseline 的主要差别之一。

### 4. 出牌和弃牌

Policy 先计算距离 Blind 目标还差多少 chips，再用剩余手数计算每手需要承担
的平均分：

- 当前最优手预计足够时直接打出；
- 预计不足且仍有 discard 时，追逐对子、两对、三条、顺子或同花 draw；
- 构筑中有指定牌型 xMult Joker 时，优先追该牌型；
- 保留 Steel、Gold 和 Blue Seal 等有持有价值的牌；
- Photograph + Hanging Chad 构筑会额外保护并优先打出高价值人头牌；
- 没有 discard 但仍有多手时，可能用低价值牌主动“过牌”换新牌；
- Square Joker 前期会尽量补到 4 张，要求正好 5 张的 Boss 则强制补到
  5 张。

这套逻辑会根据“过关所需分数”决定是否继续找牌，减少有能力过关时仍然盲目
追求大牌的浪费。

### 5. Boss Blind 适配

Policy 对部分 Boss 规则作了显式处理：

- **The Mouth**：首手在 High Card 与 Pair 中选定一种，后续保持同牌型；
- **The Eye**：尽量不重复已经打过的牌型；
- **The Psychic**：出牌不足 5 张时补足 5 张；
- 会根据 Blind rule 中“正好五张”或“基础 chips 和 mult 减半”等文本调整
  枚举和估分。

这种适配减少了“普通 Blind 策略在 Boss 上直接失效”的情况。

### 6. Joker 构筑和排序

商店中会给 Joker 计算近似价值，重点偏向：

- Blueprint、Brainstorm 等复制类；
- Cavendish、Card Sharp、Blackboard、Hologram、Constellation 等稳定
  xMult；
- 能持续成长的 Campfire、Madness、Red Card、Runner、Spare Trousers；
- 当前构筑缺失的第一张可靠 xMult Joker。

有空槽时购买达到阈值的 Joker；满槽时只有新 Joker 比最弱的可出售 Joker
高出足够 margin 才替换。Rental 和 Perishable 会扣分，Negative 和其他
Edition 会加分。

它还会主动调整 Joker 顺序：

- Blueprint 移到最值得复制 Joker 的左边；
- Brainstorm 把最值得复制的 Joker 移到最左侧；
- 通常让加法 chips/mult 在前、乘法 xMult 在后。

这避免了“买到了强 Joker，但因为排列错误没有发挥组合价值”。

### 7. 商店经济

策略不是把钱全部花光，而是使用动态现金下限：

- 早期优先用 Buffoon Pack 建立基础 Joker 构筑；
- 有稳定构筑后才购买相关 Planet、Voucher 和其他 Booster；
- 多数消费后至少保留 5–10 美元；
- 缺少第二个计分 Joker或可靠 xMult 时才积极 reroll；
- 钱越多、Ante 越高，允许的 reroll 次数越多；
- Bull 和 Bootstraps 等金钱缩放 Joker 会改变 reroll 倾向；
- 每个商店通常最多开一个包，防止连续消费失控。

Campfire 有专门策略：在 xMult 尚未成长到约 3 倍时，会出售 Consumable，
购买便宜 Tarot/Planet 再出售，或者出售弱 utility Joker，为 Campfire
累积 xMult。

### 8. Pack 和 Consumable

开包后会统一比较 Joker、Planet、Tarot、Spectral 和 Playing Card 的价值，
而不是固定拿第一张。典型偏好包括：

- Soul、Black Hole、Wraith、Immolate 等高价值 Spectral；
- Hermit、Temperance 等经济 Tarot；
- 与已经常用牌型匹配的 Planet；
- 能补全现有构筑的 Joker。

Ankh 和 Hex 在已有多个 Joker 时会被判为负价值，避免摧毁成熟构筑。
Consumable 使用时会读取合法目标范围，并根据牌面价值选择增强、复制或删除
目标。

### 9. Sol 能赢而 Luna、Terra 未赢的主要原因

从 Policy 结构和 held-out 结果看，Sol 的优势不是单一 Joker 或 Dagger
错误，而是多个决策层同时更完整：

1. 用近似实际得分代替单纯牌型排序；
2. 把过关目标、剩余手数和 discard 联合起来；
3. 显式建模大量 Joker 以及关键组合；
4. 会替换、排序 Joker，并寻找稳定 xMult；
5. 会管理现金、reroll、Voucher、Booster 和 Consumable；
6. 对多个 Boss Blind 有专门处理。

这些机制让它既能提高普通 Episode 的平均进度，也能在少量优质商店和抽牌
轨迹上形成足以完成 Run 的构筑。它仍是启发式近似模型，存在估分误差和大量
硬编码阈值；5/128 的胜率说明它能抓住部分强轨迹，但稳定性仍然有限。

## 总体结论

Packaged baseline 的 held-out 平均分为 3.695313，且没有胜局。Terra 和
Luna 都将平均进度提高到 baseline 的两倍以上，但仍未完成任何 held-out
Run。Sol 同时取得最高的非奖励进度和唯一的 5 个胜局。

此前约 7.63 分的 skill A/B Run 和约 8.69 分的 no-skill A/B Run 都是
Agent 优化后的最终 Program。把它们称为 baseline 会混淆共同的起始 Program
和优化结果。

## 历史结果处理

`runs/` 中原始 Run record 与 Assessment report 描述的是修复前的 `epg1`
引擎，其环境摘要为
`sha256:b998b1067991a3e5425576402ece11351eaec111727340882f9d05e634e8c5f4`。
这些历史文件没有被覆盖。

本报告记录的是冻结 final Programs 在 `epg2` 上的直接重评结果。由于引擎
版本现在属于环境参数，`epg1` 和 `epg2` 具有不同环境身份，不能当作来自
同一个 Benchmark 环境的结果合并统计。
