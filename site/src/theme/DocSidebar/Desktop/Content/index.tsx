import Link from "@docusaurus/Link";
import {useAnnouncementBar, useScrollPosition} from "@docusaurus/theme-common/internal";
import {ThemeClassNames} from "@docusaurus/theme-common";
import {translate} from "@docusaurus/Translate";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import DocSidebarItems from "@theme/DocSidebarItems";
import type {Props} from "@theme/DocSidebar/Desktop/Content";
import clsx from "clsx";
import {type ReactNode, useState} from "react";
import styles from "./styles.module.css";

function useShowAnnouncementBar() {
  const {isActive} = useAnnouncementBar();
  const [showAnnouncementBar, setShowAnnouncementBar] = useState(isActive);

  useScrollPosition(
    ({scrollY}) => {
      if (isActive) setShowAnnouncementBar(scrollY === 0);
    },
    [isActive],
  );

  return isActive && showAnnouncementBar;
}

export default function DocSidebarDesktopContent({
  path,
  sidebar,
  className,
}: Props): ReactNode {
  const showAnnouncementBar = useShowAnnouncementBar();
  const {
    i18n: {currentLocale},
  } = useDocusaurusContext();
  const isChinese = currentLocale === "zh-CN";
  const isCover = /\/docs\/?$/.test(path);
  const navigationItems = sidebar.slice(1);

  return (
    <nav
      aria-label={translate({
        id: "theme.docs.sidebar.navAriaLabel",
        message: "Docs sidebar",
        description: "The ARIA label for the sidebar navigation",
      })}
      className={clsx(
        "menu thin-scrollbar leaderboard-sidebar docs-research-sidebar",
        styles.menu,
        showAnnouncementBar && styles.menuWithAnnouncementBar,
        className,
      )}
    >
      <header className="leaderboard-sidebar-suite docs-sidebar-cover">
        <Link to="/docs/" aria-current={isCover ? "page" : undefined}>
          {isChinese ? "文档" : "Documentation"}
        </Link>
        <span className="leaderboard-sidebar-status">
          <i aria-hidden="true" />
          {isChinese ? "当前文档" : "Current documentation"}
        </span>
      </header>
      <ul className={clsx(ThemeClassNames.docs.docSidebarMenu, "menu__list")}>
        <DocSidebarItems
          items={navigationItems}
          activePath={path}
          level={1}
        />
      </ul>
    </nav>
  );
}
