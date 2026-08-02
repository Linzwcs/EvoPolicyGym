import {translate} from "@docusaurus/Translate";
import IconArrow from "@theme/Icon/Arrow";
import type {ReactNode} from "react";
import type {Props} from "@theme/DocSidebar/Desktop/CollapseButton";

export default function CollapseButton({onClick}: Props): ReactNode {
  const label = translate({
    id: "theme.docs.sidebar.collapseButtonTitle",
    message: "Collapse sidebar",
    description: "The title attribute for collapse button of doc sidebar",
  });

  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      className="doc-sidebar-collapse"
      onClick={onClick}
    >
      <span>{label}</span>
      <IconArrow className="doc-sidebar-collapse-icon" />
    </button>
  );
}
