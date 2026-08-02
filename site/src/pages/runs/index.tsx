import Link from "@docusaurus/Link";
import {AcademicPage, SectionHeading} from "../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../components/Localized";

export default function RunRecordsPage() {
  const language = useSiteLanguage();
  return (
    <AcademicPage
      title={language === "zh" ? "Run 记录" : "Run records"}
      description="The auditable Program, Evaluation, Feedback, and Artifact sequence produced by an EvoPolicyGym Run."
      eyebrow={<Localized en="Auditable evolution record" zh="可审计的演化记录" />}
      heading={<Localized en="Run records" zh="Run 记录" />}
      lead={
        <p>
          <Localized
            en="A Program-Evolution Run is a bounded sequence of immutable Programs evaluated on Agent-selected indices from one fixed training Episode pool."
            zh="一次 Program-Evolution Run 是一个有界序列：Agent 从固定训练 Episode 池中选择编号，并评估不可变 Programs。"
          />
        </p>
      }
      className="runs-page"
    >
      <section className="epg-wide epg-section">
        <SectionHeading
          index="01"
          title={<Localized en="Run model" zh="Run 模型" />}
          description={
            <Localized
              en="Search, selection, and final measurement remain distinct stages."
              zh="搜索、选择和最终测量保持为不同阶段。"
            />
          }
        />
        <div className="run-flow">
          {[
            ["01", "Program", "immutable source", "不可变源码"],
            ["02", "Selector", "Run-local indices", "Run-local 编号"],
            ["03", "Evaluation", "fresh runtimes", "全新 runtimes"],
            ["04", "Feedback", "public evidence", "公开证据"],
            ["05", "Revision", "next Program", "下一个 Program"],
          ].map((item) => (
            <article key={item[0]}>
              <span>{item[0]}</span>
              <strong>{item[1]}</strong>
              <small>{language === "zh" ? item[3] : item[2]}</small>
            </article>
          ))}
        </div>
        <div className="run-copy">
          <p>
            <Localized
              en="Reusing an index across Submissions preserves the hidden Episode specification for matched comparison. Every evaluation still creates a fresh Environment and Policy runtime and consumes budget."
              zh="在不同 Submission 中复用编号，会为配对比较保持隐藏 Episode specification；但每次评估仍创建新的 Environment 与 Policy runtime，并消耗预算。"
            />
          </p>
          <p>
            <Localized
              en="After the Agent Session closes, Host-only Validation selects a submitted Program and held-out Assessment measures only that selection."
              zh="Agent Session 关闭后，Host-only Validation 选择已提交 Program，held-out Assessment 只测量该选择。"
            />
          </p>
        </div>
      </section>

      <section className="epg-band">
        <div className="epg-wide epg-band-grid">
          <div>
            <p className="epg-eyebrow">
              <Localized en="Current publication state" zh="当前发布状态" />
            </p>
            <h2>
              <Localized
                en="No public v0.3 Run record yet."
                zh="尚未发布 v0.3 Run 记录。"
              />
            </h2>
          </div>
          <div>
            <p>
              <Localized
                en="The Run-record surface is implemented, but active CartPole and Balatro evolution records have not yet been published on this site."
                zh="Run record 能力已经实现，但本站尚未发布当前 CartPole 或 Balatro 演化记录。"
              />
            </p>
            <Link className="epg-text-link" to="/results/">
              <Localized en="Open historical Core16 evidence" zh="打开历史 Core16 证据" /> →
            </Link>
          </div>
        </div>
      </section>
    </AcademicPage>
  );
}
