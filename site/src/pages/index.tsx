import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import {Localized} from "../components/Localized";
import {paperMeta, projectMeta} from "../data/project";

const portals = [
  {
    index: "01",
    path: "/docs/",
    titleEn: "Project documentation",
    titleZh: "项目文档",
    copyEn: "Install the Kernel, understand the evaluation lifecycle, and work against the public SDK.",
    copyZh: "安装 Kernel、理解评估生命周期，并通过公开 SDK 使用项目。",
  },
  {
    index: "02",
    path: "/environments/",
    titleEn: "Environment catalog",
    titleZh: "Environment 目录",
    copyEn: "Browse independently installable Benchmark distributions across control, planning, robotics, and games.",
    copyZh: "浏览覆盖控制、规划、机器人与游戏的独立 Benchmark distributions。",
  },
  {
    index: "03",
    path: "/results/",
    titleEn: "Research evidence",
    titleZh: "研究证据",
    copyEn: "Inspect held-out scores and final-Policy reruns from the historical Core16 experiment.",
    copyZh: "检查历史 Core16 实验的 held-out 分数与最终 Policy 重跑。",
  },
];

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
  return (
    <Layout
      title="Autonomous policy evolution"
      description="Open-source research software for evaluating autonomous Policy evolution in interactive Environments."
    >
      <main className="home-journal">
        <header className="home-masthead epg-wide">
          <div className="home-masthead-copy">
            <p className="epg-eyebrow">
              <Localized
                en="Open-source research software · v0.3"
                zh="开源研究软件 · v0.3"
              />
            </p>
            <h1>EvoPolicyGym</h1>
            <p className="home-deck">
              <Localized
                en="Evaluating autonomous Policy evolution in interactive Environments."
                zh="在交互式 Environment 中评估自主 Policy 演化。"
              />
            </p>
            <p className="home-abstract">
              <Localized
                en="A Coding Agent studies an Environment, authors executable decision systems, learns from bounded evaluation evidence, and leaves behind a Program that can be selected, inspected, and measured on held-out Cases."
                zh="Coding Agent 研究 Environment、编写可执行决策系统、从有界评估证据中学习，并留下能够被选择、检查和在 held-out Cases 上测量的 Program。"
              />
            </p>
            <div className="home-actions">
              <Link className="epg-button epg-button--primary" to="/docs/getting-started/">
                <Localized en="Get started" zh="快速开始" /> <span>→</span>
              </Link>
              <Link className="epg-text-link" to="/blog/">
                <Localized en="Read the research journal" zh="阅读研究日志" />
              </Link>
            </div>
          </div>

          <aside className="home-project-record">
            <p className="epg-eyebrow">
              <Localized en="Project record" zh="项目记录" />
            </p>
            <dl>
              <div>
                <dt><Localized en="Release" zh="版本" /></dt>
                <dd>{projectMeta.versionLabel}</dd>
              </div>
              <div>
                <dt><Localized en="Policy interface" zh="Policy 接口" /></dt>
                <dd><code>{projectMeta.protocolVersion}</code></dd>
              </div>
              <div>
                <dt><Localized en="Paper experiment" zh="论文实验" /></dt>
                <dd>{paperMeta.experimentVersion}</dd>
              </div>
              <div>
                <dt><Localized en="Implementation" zh="实现" /></dt>
                <dd>Python 3.12</dd>
              </div>
              <div>
                <dt><Localized en="License" zh="许可证" /></dt>
                <dd>MIT</dd>
              </div>
            </dl>
            <a href={paperMeta.url}>
              <Localized en="Read the paper" zh="阅读论文" /> ↗
            </a>
          </aside>
        </header>

        <section className="home-portals epg-wide">
          <header className="home-section-intro">
            <h2><Localized en="Project index" zh="项目索引" /></h2>
            <p className="home-section-meta">
              <Localized
                en="Documentation · Environments · Evidence"
                zh="文档 · 环境 · 证据"
              />
            </p>
          </header>
          <div className="home-portal-grid">
            {portals.map((portal) => (
              <Link key={portal.path} to={portal.path} className="home-portal">
                <span>{portal.index}</span>
                <h3><Localized en={portal.titleEn} zh={portal.titleZh} /></h3>
                <p><Localized en={portal.copyEn} zh={portal.copyZh} /></p>
                <b><Localized en="Open" zh="打开" /> →</b>
              </Link>
            ))}
          </div>
        </section>

        <section className="home-notes epg-wide">
          <header className="home-section-intro">
            <h2><Localized en="Research notes" zh="研究记录" /></h2>
            <p className="home-section-meta">
              <Localized
                en="Experiments · Design · Findings"
                zh="实验 · 设计 · 发现"
              />
            </p>
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
          <Link className="epg-text-link" to="/blog/">
            <Localized en="Browse all research notes" zh="浏览全部研究文章" /> →
          </Link>
        </section>
      </main>
    </Layout>
  );
}
