# Balatro human-in-the-loop policy lab

本目录用于从已有 Agent 策略出发，按公开 replay 证据共同迭代出可稳定通关
Ante 8 的策略。原始策略保持只读：

`runs/balatro-skill-ab-gpt-5.6-sol-retry-20260725-164423/workspace/program`

## 当前基线

这里的“基线”指最初复制进来的 Agent 策略，不是当前候选：

- 最终 digest：`sha256:8ac1c1b...`
- 相同 digest 的 68 个公开 Episode：0 次通关，平均约 7.37 个 Blind
- 最佳历史 digest：`sha256:60862b...`，20 个公开 Episode 平均约 8.55 个 Blind
- 私有 Validation 选择 `submission-000010`，8 个 Episode 平均 6.75 个 Blind
- 所有候选在公开评估和 Validation 中均为 0 次通关

因此当前策略尚不能依赖小样本均值做局部调参，必须补全中后期成长能力。

## 当前候选

当前 `program/` digest 为
`sha256:c61ac881f7e642f639335c6391698975fdaf8102c68b97dc6495878aee3febe6`。

早期冻结版本在 64 个未参与调参的 validation Episode 上达到平均
9.34 个 Blind，并在最终 64 个 test Episode 上达到平均 10.84 个
Blind、中位数 11、1 次通关。当前 H19 版本在另一批 64 个未见
validation Episode 上达到平均 9.84、中位数 11。当前 H21 在另一批
64 个未见 validation Episode 上达到平均 10.33、1 次通关；封存的
test 未被重新使用。H22 进一步修正 The Needle 的单手弃牌规划，并在
另一批 64 个未见 validation Episode 上达到平均 9.81、1 次通关。当前
H24 修复背面牌被错误当成同点数/同花色保留的问题；在全新 64 局 paired
validation 上从同排程 H22 的 10.63 提升至 10.70，双方均通关 2 次，
4 个 seed 中 3 个相同、1 个改善、没有下降。当前 H25 在 Ante 2+、弃牌
耗尽且当前手不致命时，用安全 kicker 增加下一手抽牌吞吐；另一批全新
64 局 paired validation 上从同排程 H24 的 10.23 提升至 10.45，早死
从 15 降至 14，双方均未通关。该结果支持平均生存改善，尚不能证明独立
validation 通关率提高。当前 H26 进一步将主阶段 +Mult 排在主阶段 XMult
之前；另一批 64 局 paired validation 上从同排程 H25 的 9.78 提升至
9.89，`≥18` 从 3 增至 5，双方均通关 2 次，4 个 seed 中 2 个改善、
2 个相同。
在两个公开确定性 Validation 排程、共 32 个 Episode 上：

- 1/32 通关，胜率 3.125%，Policy failure 为 0；
- 平均清除 8.875 个 Blind；原始配对基线为 6.656，H2 为 7.406；
- 胜局清满 24 个 Blind，`won=True`，进入 Ante 9；
- Ante 8 Boss 最后一手 50,400 分，最终 139,104/100,000。

最终成对评估产物：

- `runs/balatro-human-loop-final-paired16-seed20260726/`
- `runs/balatro-human-loop-final-paired16-seed20260727/`

Run score 对通关局给 1024 奖励，因此最终平均 Run score 40.125 不等同于平均
Blind 数；去掉胜局后，其余 31 局平均清除 8.387 个 Blind。

## 已确认的结构性问题

1. **手牌估值严重低估/漏算 Joker。** 当前模型只覆盖少量直接
   `t_chips`、`t_mult`、`x_mult`，大量依赖打出牌、牌型、顺序、持有牌、
   retrigger 和动态 `extra` 的效果被当作零。
2. **弃牌是“保留当前最佳出牌”的静态规则。** 它没有比较 draw 后的期望分数、
   outs、剩余手数和当前 Blind 缺口，容易弃掉构筑需要的牌或在无望追牌时浪费弃牌。
3. **商店只会填空槽，不会升级构筑。** Joker 满槽后永不出售或替换，也不 reroll；
   除 Buffoon 外忽略所有 booster；最终版本还完全不买 voucher。
4. **消耗牌链路缺失。** 策略从不主动使用已持有的 Tarot/Planet/Spectral，
   pack 选择也没有为需要目标牌的卡生成目标。
5. **没有长期构筑状态。** `primary_hand` 实际只是上一手最高估值牌型，会来回漂移；
   没有角色覆盖（Chips、+Mult、XMult、scaling）、成长速度或切换成本。
6. **没有 Boss、skip 和 Joker 顺序规划。** 永远选择 Blind，且永远不调整 Joker
   顺序，无法规避关键 Boss 或正确放置 XMult。
7. **回放测试只校验 Action 字段。** 它能防非法动作，但无法发现“合法但必输”的
   决策回归。

公开 replay 中的代表性失败：Ante 4 Big Blind，目标 7500，策略有 `$72` 和满
Joker 槽，却保留 Smiley Face、Photograph、Abstract Joker、Fortune Teller、
Seltzer；随后在后两手连续打单张 High Card，最终 4413/7500 失败。这同时暴露了
商店不花钱、构筑不替换和手牌/Joker 联合估值不足。

## 目标结构

```text
program/
  policy.py                 # ABI 与组装
  policy_system/
    state.py                # 规范化 StateView 与 EpisodePlan
    actions.py              # 唯一 Action admission gateway
    hands.py                # 牌型识别与候选枚举
    effects.py              # 可见效果 -> 结构化 effect roles
    scoring.py              # 出牌顺序敏感的分数区间估计
    draws.py                # outs、抽牌概率、弃牌价值
    build.py                # 构筑角色覆盖、协同和替换价值
    economy.py              # 购买、出售、reroll、利息与生存预算
    strategy.py             # 各 phase 候选生成与统一选择
  tests/
    test_actions.py
    test_replays.py
    fixtures/               # 精简的关键公开状态
analysis/
  replay_report.py          # 只读取公开 replay 的诊断工具
  evidence.md               # digest/批次/假设/结果台账
```

核心接口应是：

```text
observation
  -> StateView
  -> BuildPlan
  -> CandidateIntent[]
  -> OutcomeEstimate(score range, survival, growth, economy, confidence)
  -> ActionGateway
```

## 实施顺序

### P0：建立可靠测量

- 把现有策略复制为 `program/` 起点，但不把它误称为最佳候选。
- 生成 replay 报告：每局死亡点、每 Blind 实际/预测分数、购买与跳过机会、
  终局钱、Joker 角色覆盖。
- 固化至少三类 fixture：弱局、当前最强局、满 Joker 槽且有升级机会的商店。

### P1：修复成长闭环

- 实现结构化 Joker effect roles 和边际构筑价值。
- 支持 Joker 替换、普通 booster、voucher、reroll 和利息/危险预算。
- 支持零目标和有目标的 consumable/pack 行为。
- 先让策略能把钱转化为长期分数，再优化细粒度出牌。

### P2：统一出牌与弃牌

- 枚举 1–5 张牌，并依据真实 scoring cards、增强、edition、seal、Joker
  激活条件和顺序估计分数区间。
- 使用 `last_hand.breakdown` 校准模型误差。
- 弃牌比较“现在打”与“弃牌后各牌型的概率加权价值”，并考虑剩余 hands。

### P3：通关规划

- 稳定追踪 primary/secondary build，而不是复制上一手牌型。
- 加入 Blind scaling、生存裕量、Boss constraint、skip tag 和 Joker 排序。
- 优化目标使用“通关率优先，Blinds cleared 次之”，不以单局最高 Ante 选型。

## 第一轮验收标准

- replay 中所有动作严格合法，Policy failure 为 0；
- 不再出现“高额现金 + 明显弱满槽构筑 + 直接离店”；
- 每个被购买/保留/出售的 Joker 都能给出角色、激活率和边际价值；
- 对代表性终局，预测分数误差可从 `last_hand` 被解释和记录；
- 先在固定公开 replay 上通过语义断言，再运行新 Episode 比较分布。

## 本地命令

从仓库根目录生成默认基线报告：

```console
python3.12 human-in-loop/balatro/analysis/replay_report.py
```

也可以分析任意 replay 文件或包含 replay 的目录：

```console
python3.12 human-in-loop/balatro/analysis/replay_report.py \
  path/to/replay.jsonl --output /tmp/balatro-report.md
```

运行 human-in-loop 策略回归：

```console
cd human-in-loop/balatro/program
PYTHONPYCACHEPREFIX=/tmp/evopolicygym-pycache \
  PYTHONPATH=../../../src \
  python3.12 -m unittest discover -s tests -v
```

当前默认报告覆盖 `submission-000019` 的 4 局：38 次离店中有 22 次是在
`$15+` 且仍有可负担库存时直接离开；四局终局现金为 `$49/$89/$31/$72`。
动作中没有 `sell_joker`、`reroll_shop`、`redeem_voucher` 或
`use_consumable`。这使 P1 的第一项实现明确为：构筑边际价值、满槽替换和花钱预算。

### H1 实现状态

`effects.py` 现在只依据当前 observation 中公开的 `ability`、
`rule.parameters`、`rule.summary`、edition 和 debuff，输出 Chips、+Mult、
XMult、retrigger、scaling、economy、generation、enabler、utility 等角色及
置信度。`build.py` 用统一价值尺度评估角色缺口、利息储备、购买和满槽替换。

替换是两步事务：先通过当前 `sell_joker` descriptor 卖出，再按稳定的公开 card
key 记录 Episode-local 购买意图；下一次 observation 重新读取临时 index，并且
只有 `buy_card` gateway 接纳后才购买。若商品消失或动作不合法，意图会被清除。

在 `submission-000019` 的公开状态上做反事实探针，新模型产生 6 个
`sell_joker` 触发点，主要针对 Reserved Parking、Odd Todd 和 Fortune Teller；
这只证明新能力命中了预期状态，不代表反事实轨迹或得分有效。真实收益仍需新
Episode 测量。

随后进行了两组各 16 局的同 seed 配对 Validation。第一次实现因改变开放槽购买
而严重回归；根据 replay 定位后，开放槽购买和 pack 排序恢复为基线行为，只在满槽
时启用 H1 模型。最终候选在 32 个配对 Episode 上从 6.6563 提升至 6.8438，
平均 `+0.1875` Blind，5 局改善、3 局变差、24 局相同，26 次出售均合法且
Policy failure 为 0。两边仍是 0 次通关，因此 H1 只应视为保留的小幅改进。

### H2–H7 实现状态

- H2 在保留利息储备的前提下购买高价值 Voucher 和 Celestial pack，只使用免费
  reroll。32 局平均从 6.656 提升到 7.406，仍未通关。
- H4 用公开 `last_hand` 回放校准统一计分器。620 次可比较出牌上，预测 MAPE
  从 0.890 降到 0.156，±25% 命中率从 33.5% 提升到 79.5%。它覆盖牌型条件、
  逐牌和持牌效果、edition、动态参数、retrigger，并避免把 face-down 牌虚构成
  Flush。
- H4c 把弃牌从固定阈值改为随剩余弃牌数变化的安全边际，并落实 The Psychic
  的“每手正好五张”约束。32 局平均清除 8.719 个 Blind。
- H5–H7 修正周期 XMult 的长期估值，识别公开 copy 规则，购买 Blueprint，并
  在每次出牌前比较可见手牌下的复制目标。补齐 `x3 if ...` 解析后，Blueprint
  会在 Blackboard 激活时移动到它左侧，最终将原本停在 Ante 7 的同一局转化为
  Ante 8 通关。
- H8 将 Boss 规则约束抽到 `boss.py`，覆盖 The Psychic、The Eye、The Mouth、
  首手 face-down 等公开文本，并加入零目标 Planet 的安全使用入口。H8 在相同
  32 局上保持 1/32 通关、8.875 平均 Blind 和 0 Policy failure，说明它是低风险
  结构增强；当前排程没有出现可用的 Planet consumable，因此尚未测出增益。
