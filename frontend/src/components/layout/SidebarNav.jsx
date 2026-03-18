import { NavLink } from "react-router-dom";
import { LayoutDashboard, BookOpen, Volume2, Map } from "lucide-react";

const navItems = [
  { to: "/", label: "Live", icon: LayoutDashboard },
  { to: "/prep", label: "Prep", icon: BookOpen },
  { to: "/voices", label: "Voices", icon: Volume2 },
  { to: "/campaign", label: "Campaign", icon: Map },
];

export default function SidebarNav() {
  return (
    <nav className="sidebar-nav">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={label}
          className={({ isActive }) =>
            `nav-glyph-btn flex items-center gap-2 w-full justify-start md:justify-center xl:justify-start md:px-2 ${isActive ? "is-active" : ""}`
          }
        >
          <Icon size={18} className="shrink-0 text-[var(--gold)]/90" aria-hidden />
          <span className="md:sr-only xl:not-sr-only">{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
