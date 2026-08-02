import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Layout from "@theme/Layout";
import {Localized} from "../components/Localized";
import {paperMeta} from "../data/project";
import {formatScore, showcase} from "../lib/showcase";

const notes = [
  {
    date: "2026-07-29",
    path: "/blog/balatro-policy-evolution/",
    titleEn: "Teaching a Coding Agent to write a Balatro policy system",
    titleZh: "让 Coding Agent 编写打《小丑牌》的策略系统",
    tagEn: "Experiment",
    tagZh: "实验",
  },
  {
    date: "2026-07-27",
    path: "/blog/designing-evopolicygym/",
    titleEn: "Why EvoPolicyGym treats policy programs as research artifacts",
    titleZh: "为什么 EvoPolicyGym 将策略 Program 视为研究产物",
    tagEn: "Design",
    tagZh: "设计",
  },
];

export default function HomePage() {
  const mediaBaseUrl = useBaseUrl("/");
  const featuredClips = [
    ["gpt_5_5", "minigrid_doorkey"],
    ["claude_opus_4_7", "bipedal"],
    ["deepseek_v4_pro", "parking"],
    ["claude_opus_4_7", "fetch_push"],
  ].flatMap(([modelSlug, environmentId]) => {
    const clip = showcase.clips.find(
      (candidate) => candidate.model_slug === modelSlug && candidate.env_id === environmentId,
    );
    return clip ? [clip] : [];
  });

  return (
    <Layout
      title="Autonomous policy evolution"
      description="Open-source research infrastructure for evaluating the decision systems that Coding Agents build from environment feedback."
    >
      <main className="home-journal">
        <header className="home-hero epg-wide">
          <div className="home-hero-copy">
            <p className="epg-eyebrow">
              <Localized
                en="Open-source research infrastructure · v0.3"
                zh="开源研究基础设施 · v0.3"
              />
            </p>
            <h1>
              <Localized
                en="Coding agents build better policy systems from environment feedback."
                zh="让 Coding Agent 从环境反馈中构建更好的 Policy 系统。"
              />
            </h1>
            <p className="home-deck">
              <Localized
                en="EvoPolicyGym provides a standardized evaluation protocol and a unified interface to interactive environments, giving Coding Agents the infrastructure to evolve executable Policies from bounded feedback and measure them on held-out Cases."
                zh="EvoPolicyGym 提供标准化的评估协议与统一的交互式 Environment 接口，为 Coding Agent 从有界反馈中演化可执行 Policy，并在 held-out Cases 上进行测量提供基础设施。"
              />
            </p>
            <div className="home-actions">
              <Link className="epg-button epg-button--primary" to="/leaderboard/">
                <Localized en="Open leaderboard" zh="打开排行榜" /> <span>→</span>
              </Link>
              <Link className="epg-text-link" to="/docs/getting-started/">
                <Localized en="Get started" zh="快速开始" />
              </Link>
              <a className="epg-text-link" href={paperMeta.url}>
                <Localized en="Paper" zh="论文" /> ↗
              </a>
            </div>
          </div>

          <div className="home-hero-demo">
            <div className="home-demo-grid">
              {featuredClips.map((clip) => (
                <Link
                  key={clip.id}
                  className="home-demo-tile"
                  to={`/results/environments/${clip.env_id}/`}
                  aria-label={`Open the ${clip.env_display} held-out rerun record`}
                >
                  <img
                    src={`${mediaBaseUrl}${clip.media}`}
                    alt={`${clip.model_display}-authored Policy running in the ${clip.env_display} environment`}
                  />
                  <span>
                    <strong>{clip.env_display}</strong>
                    <small>{clip.model_display} · {formatScore(clip.score)}</small>
                  </span>
                </Link>
              ))}
            </div>
            <div className="home-demo-caption">
              <span>
                <Localized
                  en="Four selected Policies · original-Environment reruns"
                  zh="四个选中 Policy · 原始 Environment 重跑"
                />
              </span>
              <Link to="/results/core16/">
                <Localized en="Open archive" zh="打开档案" /> →
              </Link>
            </div>
          </div>
        </header>

        <section className="home-notes epg-wide">
          <header className="home-section-intro">
            <div>
              <p className="epg-eyebrow"><Localized en="Research journal" zh="研究日志" /></p>
              <h2><Localized en="Recent Blogs" zh="近期博客" /></h2>
            </div>
            <Link className="epg-text-link" to="/blog/">
              <Localized en="All notes" zh="全部文章" /> →
            </Link>
          </header>
          <div className="home-note-list">
            {notes.map((note) => (
              <Link key={note.path} to={note.path} className="home-note">
                <time dateTime={note.date}>{note.date}</time>
                <span><Localized en={note.tagEn} zh={note.tagZh} /></span>
                <h3><Localized en={note.titleEn} zh={note.titleZh} /></h3>
                <b>↗</b>
              </Link>
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}
