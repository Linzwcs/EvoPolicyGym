import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Layout from "@theme/Layout";
import {Localized} from "../components/Localized";
import {paperMeta} from "../data/project";

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

const featuredRuns = [
  {
    id: "balatro-sol",
    kind: "balatro",
    path: "/blog/balatro-policy-evolution/",
    media: "images/blog/balatro-sol-winning-replay.gif",
    titleEn: "Balatro",
    titleZh: "小丑牌",
    metaEn: "Sol final Policy · score 1021",
    metaZh: "Sol 最终 Policy · 得分 1021",
    alt: "Sol-authored Balatro Policy completing a held-out run with score 1021",
  },
  {
    id: "crafter-deep-iron",
    kind: "crafter",
    path: "/environments/",
    media: "images/home/crafter-deep-iron-combat.gif",
    titleEn: "Crafter · Deep iron combat",
    titleZh: "Crafter · 深层铁矿战斗",
    metaEn: "Development submission 15 · Episode 10",
    metaZh: "开发 Submission 15 · Episode 10",
    alt: "Agent-authored Crafter Policy navigating a deep-iron combat development episode",
  },
  {
    id: "nethack-sol",
    kind: "nethack",
    path: "/blog/",
    media: "images/blog/nle-sol-policy-training-replay.gif",
    titleEn: "NetHack · Sol Policy",
    titleZh: "NetHack · Sol Policy",
    metaEn: "1,269 steps · dungeon depth 11",
    metaZh: "1,269 steps · 地下城深度 11",
    alt: "Sol-authored NetHack Policy completing a 1,269-step training episode at dungeon depth 11",
  },
];

export default function HomePage() {
  const mediaBaseUrl = useBaseUrl("/");

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
              {featuredRuns.map((run, index) => (
                <Link
                  key={run.id}
                  className={`home-demo-tile home-demo-tile--${run.kind}`}
                  to={run.path}
                  aria-label={`Open the ${run.titleEn} experiment record`}
                >
                  <img
                    src={`${mediaBaseUrl}${run.media}`}
                    alt={run.alt}
                    loading={index === 0 ? "eager" : "lazy"}
                    decoding="async"
                  />
                  <span>
                    <strong><Localized en={run.titleEn} zh={run.titleZh} /></strong>
                    <small><Localized en={run.metaEn} zh={run.metaZh} /></small>
                  </span>
                </Link>
              ))}
            </div>
            <div className="home-demo-caption">
              <span>
                <Localized
                  en="Three autonomous Policies · real experiment replays"
                  zh="三个自主 Policy · 真实实验回放"
                />
              </span>
              <Link to="/blog/">
                <Localized en="Open experiments" zh="查看实验" /> →
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
