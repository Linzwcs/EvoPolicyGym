import Link from "@docusaurus/Link";
import {AcademicPage, SectionHeading} from "../../../../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../../../../components/Localized";

const observations = [
  ["0", "cart_position", "Horizontal cart position", "小车水平位置", "[-4.8, 4.8]"],
  ["1", "cart_velocity", "Horizontal cart velocity", "小车水平速度", "unbounded"],
  ["2", "pole_angle", "Pole angle in radians", "杆的弧度角", "≈ [-0.418, 0.418]"],
  ["3", "pole_angular_velocity", "Pole angular velocity", "杆的角速度", "unbounded"],
];

export default function CartPoleReferencePage() {
  const language = useSiteLanguage();
  return (
    <AcademicPage
      title="CartPole-v1 Benchmark"
      description="The Policy-visible and scoring contract for the EvoPolicyGym CartPole-v1 Benchmark."
      eyebrow={<Localized en="Current reference distribution" zh="当前参考 distribution" />}
      heading="CartPole-v1"
      lead={
        <p>
          <Localized
            en="Balance an upright pole by moving its cart left or right. This is the smallest complete reference for the public Benchmark authoring surface."
            zh="通过左右推动小车来保持杆竖直。这是公开 Benchmark authoring surface 的最小完整参考。"
          />
        </p>
      }
      meta={
        <dl>
          <div><dt>Package</dt><dd><code>evopolicygym-benchmark-cartpole</code></dd></div>
          <div><dt>Benchmark ID</dt><dd><code>gymnasium/CartPole-v1/mean-return-v1</code></dd></div>
          <div><dt><Localized en="Maximum return" zh="最高回报" /></dt><dd>500</dd></div>
        </dl>
      }
      className="environment-reference-page"
    >
      <div className="reference-body epg-wide">
        <aside className="reference-index">
          <strong><Localized en="Reference" zh="参考" /></strong>
          <a href="#contract"><Localized en="Policy contract" zh="Policy 契约" /></a>
          <a href="#lifecycle"><Localized en="Lifecycle" zh="生命周期" /></a>
          <a href="#scoring"><Localized en="Scoring" zh="评分" /></a>
          <a href="#evaluate"><Localized en="Evaluate" zh="评估" /></a>
        </aside>
        <article className="reference-prose">
          <section id="contract">
            <SectionHeading
              index="01"
              title={<Localized en="Policy-visible contract" zh="Policy 可见契约" />}
            />
            <p>
              <Localized
                en="Policy.act() receives one ordered list containing exactly four finite floats and returns one exact integer Action."
                zh="Policy.act() 接收一个严格包含四个有限浮点数的有序列表，并返回一个精确整数 Action。"
              />
            </p>
            <div className="reference-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th><Localized en="Index" zh="索引" /></th>
                    <th><Localized en="Component" zh="分量" /></th>
                    <th><Localized en="Meaning" zh="含义" /></th>
                    <th><Localized en="Bound" zh="范围" /></th>
                  </tr>
                </thead>
                <tbody>
                  {observations.map((row) => (
                    <tr key={row[0]}>
                      <td><code>{row[0]}</code></td>
                      <td><code>{row[1]}</code></td>
                      <td>{language === "zh" ? row[3] : row[2]}</td>
                      <td><code>{row[4]}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="contract-actions">
              <article><code>0</code><strong><Localized en="Push left" zh="向左推动" /></strong></article>
              <article><code>1</code><strong><Localized en="Push right" zh="向右推动" /></strong></article>
            </div>
            <div className="epg-note">
              <strong><Localized en="Strict Action handling" zh="严格 Action 处理" /></strong>
              <p>
                <Localized
                  en="Invalid Actions are never clipped, repaired, converted, or replaced."
                  zh="非法 Action 不会被截断、修复、转换或替换。"
                />
              </p>
            </div>
          </section>

          <section id="lifecycle">
            <SectionHeading
              index="02"
              title={<Localized en="Episode lifecycle" zh="Episode 生命周期" />}
            />
            <ol>
              <li><Localized en="Derive the split-scoped Environment seed." zh="派生按 split 隔离的 Environment seed。" /></li>
              <li><Localized en="Create a fresh Environment and Policy process." zh="创建新的 Environment 与 Policy process。" /></li>
              <li><Localized en="Allow state only across act() calls in this Episode." zh="只允许状态在同一 Episode 的 act() 调用间保留。" /></li>
              <li><Localized en="Stop on termination, 500 steps, or Policy failure." zh="在终止、500 步或 Policy failure 时停止。" /></li>
            </ol>
          </section>

          <section id="scoring">
            <SectionHeading
              index="03"
              title={<Localized en="Reward and score" zh="Reward 与评分" />}
            />
            <p>
              <Localized
                en="Every valid Environment step contributes +1. The Benchmark score is the arithmetic mean of Episode contributions; a Policy failure contributes 0."
                zh="每次合法 Environment step 贡献 +1。Benchmark score 是各 Episode 计分的算术平均值；Policy failure 贡献 0。"
              />
            </p>
          </section>

          <section id="evaluate">
            <SectionHeading
              index="04"
              title={<Localized en="Install and evaluate" zh="安装与评估" />}
            />
            <pre><code>{`uv sync --project environments/gymnasium/classic_control/cartpole --extra dev\nuv build environments/gymnasium/classic_control/cartpole`}</code></pre>
            <div className="reference-actions">
              <a href="https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/cartpole">
                <Localized en="Benchmark source" zh="Benchmark 源码" /> ↗
              </a>
              <Link to="/docs/getting-started/">
                <Localized en="Getting started" zh="快速开始" /> →
              </Link>
            </div>
          </section>
        </article>
      </div>
    </AcademicPage>
  );
}
