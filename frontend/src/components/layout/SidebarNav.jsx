import { NavLink } from "react-router-dom";
import { LayoutDashboard, BookOpen, Volume2, Map } from "lucide-react";

const navItems = [
  { to: "/", label: "Live", icon: LayoutDashboard },
  { to: "/prep", label: "Prep", icon: BookOpen },
  { to: "/voices", label: "Voices", icon: Volume2 },
  { to: "/campaign", label: "Campaign", icon: Map },
];

export default function SidebarNav({ inBanner = false }) {
  const navClass = inBanner
    ? "flex flex-wrap items-center justify-center gap-2 w-full"
    : "sidebar-nav";
  const linkClass = inBanner
    ? "nav-glyph-btn flex items-center gap-2 px-3 py-2"
    : "nav-glyph-btn flex items-center gap-2 w-full justify-start md:justify-center xl:justify-start md:px-2";
  return (
    <nav className={navClass}>
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={label}
          className={({ isActive }) =>
            `${linkClass} ${isActive ? "is-active" : ""}`
          }
        >
          <Icon size={18} className="shrink-0 text-[var(--gold)]/90" aria-hidden />
          <span className={inBanner ? "" : "md:sr-only xl:not-sr-only"}>{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
