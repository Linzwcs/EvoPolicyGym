import Link from "@docusaurus/Link";
import type {ReactNode} from "react";
import type {EnvironmentReference} from "../../data/environmentReferences";
import {AcademicPage, SectionHeading} from "../../components/AcademicPage";
import {Localized, pickLocalized, useSiteLanguage} from "../../components/Localized";

export default function EnvironmentReferencePage({
  reference,
}: {
  reference: EnvironmentReference;
}) {
  const language = useSiteLanguage();
  const sourceUrl = `https://github.com/Linzwcs/EvoPolicyGym/tree/main/${reference.sourcePath}`;

  return (
    <AcademicPage
      title={`${reference.title} Benchmark`}
      description={pickLocalized(language, reference.lead)}
      eyebrow={
        <>
          {reference.ecosystem} · {reference.suite}
        </>
      }
      heading={reference.title}
      lead={<p>{pickLocalized(language, reference.lead)}</p>}
      meta={
        <dl>
          <div><dt>Benchmark ID</dt><dd><code>{reference.benchmarkId}</code></dd></div>
          <div><dt><Localized en="Horizon" zh="时域" /></dt><dd>{pickLocalized(language, reference.horizon)}</dd></div>
          <div><dt><Localized en="Direction" zh="优化方向" /></dt><dd>{pickLocalized(language, reference.direction)}</dd></div>
        </dl>
      }
      className="environment-reference-page"
    >
      <div className="reference-body epg-wide">
        <aside className="reference-index">
          <strong><Localized en="On this page" zh="本页内容" /></strong>
          <a href="#task"><Localized en="Task" zh="任务" /></a>
          <a href="#policy-interface"><Localized en="Policy interface" zh="Policy 接口" /></a>
          <a href="#evaluation"><Localized en="Evaluation" zh="评估" /></a>
          <a href="#feedback">Feedback</a>
          <a href="#using"><Localized en="Using" zh="使用" /></a>
        </aside>

        <article className="reference-prose">
          <section id="task">
            <SectionHeading index="01" title={<Localized en="Task" zh="任务" />} />
            {reference.task.map((paragraph) => (
              <p key={paragraph.en}>{pickLocalized(language, paragraph)}</p>
            ))}
          </section>

          <section id="policy-interface">
            <SectionHeading
              index="02"
              title={<Localized en="Policy interface" zh="Policy 接口" />}
            />
            <p>{pickLocalized(language, reference.observation)}</p>
            <ReferenceTable
              first={<Localized en="Observation field" zh="Observation 字段" />}
              second={<Localized en="Meaning" zh="含义" />}
              rows={reference.observationFields.map((field) => [
                <code key="name">{field.name}</code>,
                pickLocalized(language, field.description),
              ])}
            />
            <p>{pickLocalized(language, reference.action)}</p>
            <ReferenceTable
              first="Action"
              second={<Localized en="Meaning" zh="含义" />}
              rows={reference.actions.map((action) => [
                <code key="action">{action.value}</code>,
                pickLocalized(language, action.description),
              ])}
            />
          </section>

          <section id="evaluation">
            <SectionHeading
              index="03"
              title={<Localized en="Evaluation" zh="评估" />}
            />
            <ReferenceTable
              first={<Localized en="Quantity" zh="量" />}
              second={<Localized en="Definition" zh="定义" />}
              rows={reference.measures.map((measure) => [
                pickLocalized(language, measure.label),
                pickLocalized(language, measure.value),
              ])}
            />
          </section>

          <section id="feedback">
            <SectionHeading index="04" title="Feedback" />
            <p>{pickLocalized(language, reference.feedback)}</p>
            <ReferenceTable
              first={<Localized en="Field" zh="字段" />}
              second={<Localized en="Meaning" zh="含义" />}
              rows={[
                ...reference.feedbackFields.map((field) => [
                  <code key="field">{field.name}</code>,
                  pickLocalized(language, field.description),
                ]),
                [
                  <code key="artifact">{reference.artifact.name}</code>,
                  pickLocalized(language, reference.artifact.description),
                ],
              ]}
            />
          </section>

          <section id="using">
            <SectionHeading
              index="05"
              title={<Localized en="Using the distribution" zh="使用 distribution" />}
            />
            <p>
              <Localized
                en="Build this independently installable leaf project from the repository root:"
                zh="从仓库根目录构建这个可独立安装的叶子 project："
              />
            </p>
            <pre><code>{`uv sync --project ${reference.sourcePath} --extra dev\nuv build ${reference.sourcePath}`}</code></pre>
            <p>
              <Localized en="The package exports:" zh="该 package 导出：" />
            </p>
            <pre><code>{`from ${reference.importName} import ${reference.benchmarkClass}, baseline_program\n\nbenchmark = ${reference.benchmarkClass}()\nprogram = baseline_program()`}</code></pre>
            <div className="reference-actions">
              <a href={sourceUrl}><Localized en="Benchmark source" zh="Benchmark 源码" /> ↗</a>
              <a href={reference.upstreamUrl}><Localized en="Upstream task" zh="上游任务" /> ↗</a>
              <Link to="/docs/authoring/"><Localized en="Authoring guide" zh="编写指南" /> →</Link>
            </div>
          </section>
        </article>
      </div>
    </AcademicPage>
  );
}

function ReferenceTable({
  first,
  second,
  rows,
}: {
  first: ReactNode;
  second: ReactNode;
  rows: ReactNode[][];
}) {
  return (
    <div className="reference-table-wrap">
      <table>
        <thead><tr><th>{first}</th><th>{second}</th></tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}><td>{row[0]}</td><td>{row[1]}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
