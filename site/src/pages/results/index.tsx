import Link from "@docusaurus/Link";
import {AcademicPage, ResearchBoundary, SectionHeading} from "../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../components/Localized";
import {paperMeta} from "../../data/project";
import {
  environments,
  formatScore,
  models,
  showcase,
} from "../../lib/showcase";

export default function ResultsPage() {
  const language = useSiteLanguage();
  const agentRows = showcase.leaderboard.filter(
    (row) => row.model_slug !== "random_policy",
  );
  const wins = Object.fromEntries(models.map((model) => [model.slug, 0]));
  for (const environment of environments) {
    const top = Math.max(...agentRows.map((row) => row.scores[environment.id]));
    for (const row of agentRows) {
      if (row.scores[environment.id] === top) wins[row.model_slug] += 1;
    }
  }

  return (
    <AcademicPage
      title={language === "zh" ? "Core16 实验结果" : "Core16 experiment results"}
      description="Historical held-out scores and final-Policy reruns from the Core16 paper experiment."
      eyebrow={<Localized en="Paper companion · v0.1.0" zh="论文伴随材料 · v0.1.0" />}
      heading={<Localized en="Core16 experiment record" zh="Core16 实验记录" />}
      lead={
        <p>
          <Localized
            en="Held-out returns and final-Policy reruns from four coding-agent/model lanes across sixteen interactive tasks."
            zh="四条 Coding Agent / 模型链路在十六个交互任务上的 held-out return 与最终 Policy 重跑。"
          />
        </p>
      }
      meta={
        <dl>
          <div><dt><Localized en="Tasks" zh="任务" /></dt><dd>{environments.length}</dd></div>
          <div><dt><Localized en="Agent lanes" zh="Agent 链路" /></dt><dd>{models.length}</dd></div>
          <div><dt><Localized en="Reruns" zh="重跑" /></dt><dd>{showcase.clips.length}</dd></div>
          <div><dt><Localized en="Budget" zh="预算" /></dt><dd>128 episodes / run</dd></div>
        </dl>
      }
      className="results-page"
    >
      <section className="epg-wide epg-section">
        <ResearchBoundary />
      </section>

      <section className="epg-wide epg-section">
        <SectionHeading
          index="01"
          title={<Localized en="Held-out score matrix" zh="Held-out 分数矩阵" />}
          description={
            <Localized
              en="Scores are comparable only within one Environment column. Highlighted values are column maxima."
              zh="分数只能在同一个 Environment 列内比较；高亮值为列内最大值。"
            />
          }
        />
        <div className="score-matrix-wrap">
          <table className="score-matrix">
            <thead>
              <tr>
                <th>Model</th>
                {environments.map((environment) => (
                  <th key={environment.id}>
                    <Link to={`/results/environments/${environment.id}/`}>
                      {environment.display}
                    </Link>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {showcase.leaderboard.map((row) => (
                <tr key={row.model_slug}>
                  <th>
                    <strong>{row.model_display}</strong>
                    <small>{row.harness}</small>
                  </th>
                  {environments.map((environment) => {
                    const value = row.scores[environment.id];
                    const max = Math.max(
                      ...showcase.leaderboard.map(
                        (candidate) => candidate.scores[environment.id],
                      ),
                    );
                    return (
                      <td
                        key={environment.id}
                        className={value === max ? "is-best" : ""}
                      >
                        {formatScore(value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="epg-wide epg-section">
        <SectionHeading
          index="02"
          title={<Localized en="Agent lanes" zh="Agent 链路" />}
          description={
            <Localized
              en="Column-best counts summarize the matrix; they are not a cross-task aggregate score."
              zh="列内最佳次数只用于概括矩阵，并不是跨任务聚合分数。"
            />
          }
        />
        <div className="lane-grid">
          {models.map((model) => (
            <article key={model.slug}>
              <span>{model.harness}</span>
              <h3>{model.display}</h3>
              <strong>{wins[model.slug]} / {environments.length}</strong>
              <small><Localized en="column bests" zh="列内最佳" /></small>
            </article>
          ))}
        </div>
      </section>

      <section className="epg-band">
        <div className="epg-wide epg-band-grid">
          <div>
            <p className="epg-eyebrow">
              <Localized en="Behavioral evidence" zh="行为证据" />
            </p>
            <h2><Localized en="Inspect the selected Policies." zh="检查被选中的 Policies。" /></h2>
          </div>
          <div>
            <p>
              <Localized
                en={`The archive preserves ${showcase.clips.length} original-environment reruns—one for every model–environment lane.`}
                zh={`档案保留 ${showcase.clips.length} 段原环境重跑，每个模型–环境链路各一段。`}
              />
            </p>
            <Link className="epg-button epg-button--primary" to="/results/core16/">
              <Localized en="Open the rerun archive" zh="打开重跑档案" /> →
            </Link>
            <a className="epg-text-link" href={paperMeta.url}>
              <Localized en="Read the paper" zh="阅读论文" /> ↗
            </a>
          </div>
        </div>
      </section>
    </AcademicPage>
  );
}
