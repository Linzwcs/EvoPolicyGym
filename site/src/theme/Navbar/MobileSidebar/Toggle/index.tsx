import {translate} from "@docusaurus/Translate";
import {useNavbarMobileSidebar} from "@docusaurus/theme-common/internal";
import type {ReactNode} from "react";

export default function MobileSidebarToggle(): ReactNode {
  const {toggle, shown} = useNavbarMobileSidebar();
  const label = shown
    ? translate({
        id: "theme.docs.sidebar.closeSidebarButtonAriaLabel",
        message: "Close navigation bar",
        description: "The ARIA label for close button of mobile sidebar",
      })
    : translate({
        id: "theme.docs.sidebar.toggleSidebarButtonAriaLabel",
        message: "Toggle navigation bar",
        description:
          "The ARIA label for hamburger menu button of mobile navigation",
      });

  return (
    <button
      onClick={toggle}
      aria-label={label}
      aria-expanded={shown}
      className="navbar__toggle clean-btn epg-mobile-nav-toggle"
      type="button"
    >
      <span className="epg-mobile-nav-toggle__bar" />
      <span className="epg-mobile-nav-toggle__bar" />
      <span className="epg-mobile-nav-toggle__bar" />
    </button>
  );
}
