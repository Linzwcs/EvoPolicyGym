import Link from "@docusaurus/Link";
import {AcademicPage, ResearchBoundary, SectionHeading} from "../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../components/Localized";
import {RolloutCard} from "../../components/RolloutCard";
import {environmentMeta} from "../../data/environments";
import {
  clipsForEnvironment,
  environments,
  formatScore,
  scoresForEnvironment,
} from "../../lib/showcase";

export default function EnvironmentResultPage({
  pageData,
}: {
  pageData: {id: string};
}) {
  const language = useSiteLanguage();
  const environment = environments.find((item) => item.id === pageData.id);
  if (!environment) throw new Error(`Unknown result environment: ${pageData.id}`);

  const meta = environmentMeta[environment.id];
  const clips = clipsForEnvironment(environment.id);
  const ranking = [...clips].sort((a, b) => b.score - a.score);
  const scores = scoresForEnvironment(environment.id);
  const position = environments.findIndex((item) => item.id === environment.id);
  const previous = environments[(position - 1 + environments.length) % environments.length];
  const next = environments[(position + 1) % environments.length];

  return (
    <AcademicPage
      title={`${environment.display} · Core16`}
      description={`${environment.display} historical held-out scores and final-Policy reruns.`}
      eyebrow={`${environment.category} · Core16`}
      heading={environment.display}
      lead={<p>{language === "zh" ? meta.shortZh : meta.shortEn}</p>}
      meta={
        <dl>
          <div><dt><Localized en="Best score" zh="最佳分数" /></dt><dd>{formatScore(scores[0].score)}</dd></div>
          <div><dt><Localized en="Best model" zh="最佳模型" /></dt><dd>{scores[0].model_display}</dd></div>
          <div><dt><Localized en="Policy focus" zh="Policy 重点" /></dt><dd>{language === "zh" ? meta.focusZh : meta.focusEn}</dd></div>
        </dl>
      }
      className="environment-result-page"
    >
      <section className="epg-wide epg-section">
        <ResearchBoundary />
      </section>

      <section className="epg-wide epg-section">
        <SectionHeading
          index="01"
          title={<Localized en="Final-Policy evidence" zh="最终 Policy 证据" />}
          description={
            <Localized
              en="Each card reruns the validation-selected checkpoint in the original research Environment."
              zh="每张卡片都在原研究 Environment 中重跑 validation 选出的 checkpoint。"
            />
          }
        />
        <div className="rollout-grid">
          {clips.map((clip) => {
            const rank =
              ranking.findIndex((candidate) => candidate.id === clip.id) + 1;
            return (
              <RolloutCard
                key={clip.id}
                clip={clip}
                rank={rank}
                winner={rank === 1}
              />
            );
          })}
        </div>
      </section>

      <section className="epg-wide epg-section">
        <SectionHeading
          index="02"
          title={<Localized en="Reported scores" zh="报告分数" />}
          description={
            <Localized
              en="Higher is better within this Environment. Raw reward scales are not comparable across tasks."
              zh="在该 Environment 内分数越高越好；不同任务的原始 reward scale 不可比较。"
            />
          }
        />
        <div className="result-ranking">
          {scores.map((row, index) => (
            <div key={row.model_slug}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{row.model_display}</strong>
              <small>{row.harness}</small>
              <code>{formatScore(row.score)}</code>
            </div>
          ))}
        </div>
      </section>

      <nav className="result-pager epg-wide">
        <Link to={`/results/environments/${previous.id}/`}>
          ← {previous.display}
        </Link>
        <Link to="/results/core16/">
          <Localized en="All reruns" zh="全部重跑" />
        </Link>
        <Link to={`/results/environments/${next.id}/`}>
          {next.display} →
        </Link>
      </nav>
    </AcademicPage>
  );
}
