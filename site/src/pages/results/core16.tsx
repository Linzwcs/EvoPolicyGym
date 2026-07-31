import Link from "@docusaurus/Link";
import {AcademicPage, SectionHeading} from "../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../components/Localized";
import {RolloutCard} from "../../components/RolloutCard";
import {environmentMeta} from "../../data/environments";
import {clipsForEnvironment, environments, formatScore} from "../../lib/showcase";

export default function Core16GalleryPage() {
  const language = useSiteLanguage();
  return (
    <AcademicPage
      title={language === "zh" ? "64 段最终 Policy 重跑" : "64 final-Policy reruns"}
      description="The complete qualitative record from the historical Core16 paper experiment."
      eyebrow={<Localized en="Core16 · qualitative archive" zh="Core16 · 定性档案" />}
      heading={<Localized en="64 final-Policy reruns" zh="64 段最终 Policy 重跑" />}
      lead={
        <p>
          <Localized
            en="Sixteen Environments, four coding-agent/model lanes, and one original-environment rerun for every validation-selected checkpoint."
            zh="十六个 Environment、四条 Coding Agent / 模型链路，每个 validation-selected checkpoint 都有一段原环境重跑。"
          />
        </p>
      }
      className="gallery-page"
    >
      <nav className="gallery-index epg-wide" aria-label="Environment index">
        {environments.map((environment) => (
          <a href={`#${environment.id}`} key={environment.id}>
            <span>{String(environment.order + 1).padStart(2, "0")}</span>
            {environment.display}
          </a>
        ))}
      </nav>

      <div className="gallery-sections">
        {environments.map((environment) => {
          const clips = clipsForEnvironment(environment.id);
          const ranking = [...clips].sort((a, b) => b.score - a.score);
          return (
            <section
              className="gallery-environment"
              id={environment.id}
              key={environment.id}
            >
              <div className="epg-wide">
                <SectionHeading
                  index={String(environment.order + 1).padStart(2, "0")}
                  title={environment.display}
                  description={
                    language === "zh"
                      ? environmentMeta[environment.id].shortZh
                      : environmentMeta[environment.id].shortEn
                  }
                />
                <div className="gallery-environment-meta">
                  <span>{environment.category}</span>
                  <span>
                    <Localized en="Best reported" zh="最佳报告分数" />{" "}
                    <strong>{formatScore(ranking[0].score)}</strong>
                  </span>
                  <Link to={`/results/environments/${environment.id}/`}>
                    <Localized en="Result detail" zh="结果详情" /> →
                  </Link>
                </div>
              </div>
              <div className="rollout-grid epg-wide">
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
          );
        })}
      </div>
    </AcademicPage>
  );
}
