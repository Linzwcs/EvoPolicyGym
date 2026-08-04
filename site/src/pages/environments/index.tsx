import Link from "@docusaurus/Link";
import {AcademicPage} from "../../components/AcademicPage";
import {Localized, pickLocalized, useSiteLanguage} from "../../components/Localized";
import {
  environmentCollections,
  environmentDomains,
  environmentDistributionCount,
  environmentTaskProfileCount,
} from "../../data/environmentCatalog";

export default function EnvironmentCatalogPage() {
  const language = useSiteLanguage();
  const domainSummaries = environmentDomains
    .map((domain) => {
      const collections = environmentCollections.filter(
        (collection) => collection.domain === domain.id,
      );
      return {
        ...domain,
        collectionCount: collections.length,
        ecosystems: Array.from(
          new Set(collections.map((collection) => collection.ecosystem)),
        ).sort((left, right) => left.localeCompare(right)),
        taskProfileCount: collections.reduce(
          (total, collection) => total + collection.taskProfiles,
          0,
        ),
      };
    })
    .filter((domain) => domain.collectionCount > 0);

  return (
    <AcademicPage
      title={language === "zh" ? "Environment 目录" : "Environment catalog"}
      description="Independently installable Benchmark distributions for evaluating executable Policy systems."
      eyebrow={<Localized en="Public Benchmark surface" zh="公开 Benchmark surface" />}
      heading={<Localized en="Environment catalog" zh="Environment 目录" />}
      lead={
        <p>
          <Localized
            en="Independently installable Benchmark distributions across control, planning, robotics, driving, and long-horizon games."
            zh="覆盖控制、规划、机器人、驾驶和长时程游戏的独立 Benchmark distributions。"
          />
        </p>
      }
      meta={
        <dl>
          <div>
            <dt><Localized en="Distributions" zh="Distributions" /></dt>
            <dd>{environmentDistributionCount}</dd>
          </div>
          <div>
            <dt><Localized en="Named tasks" zh="具名任务" /></dt>
            <dd>{environmentTaskProfileCount}</dd>
          </div>
          <div>
            <dt><Localized en="Ecosystems" zh="生态数量" /></dt>
            <dd>{new Set(environmentCollections.map((item) => item.ecosystem)).size}</dd>
          </div>
        </dl>
      }
      className="environment-catalog-page"
    >
      <section
        className="epg-wide environment-domain-summary"
        aria-labelledby="environment-domain-title"
      >
        <header>
          <h2 id="environment-domain-title">
            <Localized en="Research domains" zh="研究领域" />
          </h2>
          <p>
            <Localized
              en="Generated from the current catalog."
              zh="根据当前环境目录自动统计。"
            />
          </p>
        </header>
        <div className="environment-domain-grid">
          {domainSummaries.map((domain, index) => (
            <article key={domain.id}>
              <div className="environment-domain-title">
                <span>{String(index + 1).padStart(2, "0")}</span>
                <h3>{pickLocalized(language, domain.title)}</h3>
              </div>
              <p>{domain.ecosystems.join(" · ")}</p>
              <dl>
                <div>
                  <dt><Localized en="Collections" zh="Collections" /></dt>
                  <dd>{domain.collectionCount}</dd>
                </div>
                <div>
                  <dt><Localized en="Task profiles" zh="任务配置" /></dt>
                  <dd>{domain.taskProfileCount}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      </section>

      <section className="epg-wide epg-section environment-catalog-list">
        <div className="catalog-table">
          {environmentCollections.map((collection, index) => {
            const visibleItems = collection.items.slice(0, 6);
            const hiddenItemCount = collection.items.length - visibleItems.length;
            return (
              <article key={collection.id} id={collection.id}>
                <div className="catalog-row-index">
                  {String(index + 1).padStart(2, "0")}
                </div>
                <div className="catalog-entry">
                  <header>
                    <div className="catalog-identity">
                      <span>{collection.ecosystem}</span>
                      <h2>{collection.suite}</h2>
                    </div>
                  </header>

                  <p className="catalog-summary">
                    {pickLocalized(language, collection.summary)}
                  </p>

                  <ul className="catalog-items clean-list">
                    {visibleItems.map((item) => (
                      <li key={item.name}>
                        {item.path ? (
                          <Link to={`/${item.path}`}>{item.name} ↗</Link>
                        ) : (
                          <span>{item.name}</span>
                        )}
                      </li>
                    ))}
                    {hiddenItemCount > 0 && (
                      <li className="catalog-items-more">
                        <span>
                          +{hiddenItemCount}{" "}
                          <Localized en="more" zh="项" />
                        </span>
                      </li>
                    )}
                  </ul>

                  <footer className="catalog-entry-footer">
                    <div className="catalog-counts">
                      <span>
                        <b>{collection.distributions}</b>{" "}
                        <Localized en="distributions" zh="distributions" />
                      </span>
                      <span>
                        <b>{collection.taskProfiles}</b>{" "}
                        <Localized en="task profiles" zh="任务配置" />
                      </span>
                    </div>
                    <div className="catalog-actions">
                      {collection.referencePath ? (
                        <Link to={`/${collection.referencePath}`}>
                          <Localized
                            en={
                              collection.referencePath.startsWith("blog/")
                                ? "Read article"
                                : "Open reference"
                            }
                            zh={
                              collection.referencePath.startsWith("blog/")
                                ? "阅读文章"
                                : "打开参考"
                            }
                          />{" "}
                          →
                        </Link>
                      ) : (
                        <a
                          href={`https://github.com/Linzwcs/EvoPolicyGym/tree/main/${collection.sourcePath}`}
                        >
                          <Localized en="View source" zh="查看源码" /> ↗
                        </a>
                      )}
                    </div>
                  </footer>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="epg-band">
        <div className="epg-wide epg-band-grid">
          <div>
            <p className="epg-eyebrow">
              <Localized en="Authoring boundary" zh="Authoring boundary" />
            </p>
            <h2>
              <Localized
                en="Benchmarks live outside the Kernel."
                zh="Benchmark 独立于 Kernel。"
              />
            </h2>
          </div>
          <div>
            <p>
              <Localized
                en="External distributions use only the public authoring SPI and own their domain semantics, dependencies, Feedback, and conformance tests."
                zh="外部 distribution 只使用公开 authoring SPI，并自行拥有领域语义、依赖、Feedback 与 conformance tests。"
              />
            </p>
            <Link className="epg-text-link" to="/docs/authoring/">
              <Localized en="Read the authoring guide" zh="阅读环境编写指南" /> →
            </Link>
          </div>
        </div>
      </section>
    </AcademicPage>
  );
}
